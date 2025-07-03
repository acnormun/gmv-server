from typing import Any, Dict, List
from flask import Blueprint, jsonify, request
import threading
import time
import os
import re

rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')

def get_rag():
    try:
        from adaptive_rag import rag_system
        return rag_system, None
    except Exception as e:
        return None, str(e)

def _parse_front_matter(text: str):
    if text.startswith('---'):
        end = text.find('---', 3)
        if end != -1:
            yaml_block = text[3:end]
            metadata = {}
            for line in yaml_block.splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip().strip('"\'')
            body = text[end + 3:].lstrip()
            return metadata, body
    return {}, text


def _load_process_documents(base_path: str, processos: List[str]):
    documentos = []
    for numero in processos:
        numero_sanitizado = numero.replace('/', '-')
        pasta = os.path.join(base_path, numero_sanitizado, 'markdowns')
        if not os.path.isdir(pasta):
            continue
        for nome in os.listdir(pasta):
            if nome.lower().endswith('.md'):
                caminho = os.path.join(pasta, nome)
                try:
                    with open(caminho, 'r', encoding='utf-8') as f:
                        texto = f.read()
                    meta, corpo = _parse_front_matter(texto)
                    meta['numero_processo'] = numero
                    documentos.append({'content': corpo, 'metadata': meta, 'filename': nome})
                except Exception:
                    continue
    return documentos


def _select_relevant_docs(docs: List[Dict[str, Any]], question: str, k: int):
    palavras = re.findall(r'\b\w{3,}\b', question.lower())
    pontuados = []
    for doc in docs:
        texto = doc['content'].lower()
        meta_text = ' '.join(str(v).lower() for v in doc['metadata'].values() if isinstance(v, str))
        score = sum(texto.count(p) for p in palavras) + sum(meta_text.count(p) for p in palavras)
        pontuados.append((score, doc))
    pontuados.sort(key=lambda x: x[0], reverse=True)
    selecionados = [d for s, d in pontuados if s > 0][:k]
    if not selecionados:
        selecionados = [d for _, d in pontuados[:k]]
    return selecionados

@rag_bp.route('/status', methods=['GET'])
def status():
    rag, error = get_rag()
    if error:
        return jsonify({"status": "error", "message": error, "isReady": False})
    from adaptive_rag import get_rag_status
    return jsonify(get_rag_status())

@rag_bp.route('/init', methods=['POST'])
def init():
    rag, error = get_rag()
    if error:
        return jsonify({"success": False, "message": error})
    if rag.is_initialized and rag.vector_store and len(rag.documents) > 0:
        return jsonify({
            "success": True, 
            "message": f"Já inicializado com {len(rag.documents)} documentos",
            "already_initialized": True
        })
    def init_background():
        from adaptive_rag import init_rag_system, load_data_directory
        print("Inicializando RAG...")
        if init_rag_system():
            print("Carregando documentos...")
            docs = load_data_directory()
            print(f"{docs} documentos carregados")
        else:
            print("Falha na inicialização")
    threading.Thread(target=init_background, daemon=True).start()
    return jsonify({
        "success": True, 
        "message": "Inicialização iniciada em background",
        "background": True
    })

@rag_bp.route('/query', methods=['POST'])
def query():
    rag, error = get_rag()
    if error:
        return jsonify({"success": False, "message": error})
    if not rag.is_initialized:
        return jsonify({"success": False, "message": "RAG não inicializado. Execute /init primeiro."})
    if not rag.vector_store or len(rag.documents) == 0:
        return jsonify({"success": False, "message": "Nenhum documento carregado. Execute /init primeiro."})
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"success": False, "message": "Campo 'question' obrigatório"})
    question = data['question'].strip()
    if not question:
        return jsonify({"success": False, "message": "Pergunta não pode ser vazia"})
    processos_ctx = data.get('context')
    if processos_ctx is not None:
        if not isinstance(processos_ctx, list):
            return jsonify({"success": False, "message": "Campo 'context' deve ser uma lista"})
        k = data.get('k', 5)
        try:
            k = int(k)
        except (ValueError, TypeError):
            k = 5
        start_time = time.time()
        result = query_with_specific_context_helper(rag, question, processos_ctx, k)
    else:
        start_time = time.time()
        result = rag.query(question)
    processing_time = time.time() - start_time
    if 'error' in result:
        return jsonify({"success": False, "message": result['error']})
    if 'processing_time' not in result:
        result['processing_time'] = round(processing_time, 2)
    return jsonify({"success": True, "data": result})

@rag_bp.route('/query-with-context', methods=['POST'])
def query_with_context():
    rag, error = get_rag()
    if error:
        return jsonify({"success": False, "message": error})
    if not rag.is_initialized:
        return jsonify({"success": False, "message": "RAG não inicializado. Execute /init primeiro."})
    if not rag.vector_store or len(rag.documents) == 0:
        return jsonify({"success": False, "message": "Nenhum documento carregado. Execute /init primeiro."})
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Dados JSON obrigatórios"})
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"success": False, "message": "Campo 'question' obrigatório"})
    processos_selecionados = data.get('context', [])
    if not isinstance(processos_selecionados, list):
        return jsonify({"success": False, "message": "Campo 'processos_selecionados' deve ser uma lista"})
    k = data.get('k', 5)
    try:
        k = int(k)
        if k < 1 or k > 20:
            k = 5
    except (ValueError, TypeError):
        k = 5
    try:
        start_time = time.time()
        result = query_with_specific_context_helper(rag, question, processos_selecionados, k)
        processing_time = time.time() - start_time
        if 'error' in result:
            return jsonify({"success": False, "message": result['error']})
        if 'processing_time' not in result:
            result['processing_time'] = round(processing_time, 2)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"❌ Erro na consulta com contexto: {e}")
        return jsonify({
            "success": False, 
            "message": f"Erro interno: {str(e)}"
        }), 500

def query_with_specific_context_helper(rag, question: str, processos_selecionados: List[str], k: int = 5) -> Dict[str, Any]:
    try:
        start_time = time.time()
        print("⚡ CONSULTA DIRETA POR PASTAS")
        print(f"   ❓ Pergunta: {question}")
        print(f"   ⚖️ Processos: {processos_selecionados}")
        if not processos_selecionados:
            return rag.query(question, top_k=k)
        docs_raw = _load_process_documents(rag.data_path, processos_selecionados)
        if not docs_raw:
            return {"error": "Nenhum documento encontrado para os processos"}

        selected = _select_relevant_docs(docs_raw, question, k)
        context_parts = []
        for i, doc in enumerate(selected, 1):
            trecho = doc['content'].strip()
            if len(trecho) > 1500:
                trecho = trecho[:1500] + '...'
            context_parts.append(f"DOCUMENTO {i} ({doc['filename']}):\n{trecho}")
        context = "\n\n" + "-"*30 + "\n\n".join(context_parts)
        prompt = f"""Responda baseado nos documentos do processo {', '.join(processos_selecionados)}:

PERGUNTA: {question}

DOCUMENTOS:
{context}

RESPOSTA:"""
        try:
            response = rag.llm.invoke(prompt) if hasattr(rag.llm, 'invoke') else rag.llm(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"❌ Erro no LLM: {e}")
            answer = f"Erro ao gerar resposta: {str(e)}"
        processing_time = time.time() - start_time
        source_documents = [{"filename": doc['filename'], "metadata": doc['metadata']} for doc in selected]
        print(f"⚡ Consulta concluída em {processing_time:.2f}s")
        return {
            "answer": answer,
            "question": question,
            "context_size": len(context),
            "documents_count": len(selected),
            "processing_time": processing_time,
            "search_method": "filesystem",
            "source_documents": source_documents,
            "context_info": {
                "documentos_selecionados": [d['filename'] for d in selected],
                "processos_selecionados": processos_selecionados,
                "total_filtered_docs": len(selected),
                "relevant_chunks": len(selected)
            }
        }
    except Exception as e:
        print(f"❌ Erro na consulta rápida: {e}")
        return {"error": f"Erro interno: {str(e)}"}

@rag_bp.route('/processos-triagem', methods=['GET'])
def get_processos_triagem():
    try:
        from utils.triagem import get_processos
        processos = get_processos()
        print(f"📋 Retornando {len(processos)} processos da triagem")
        return jsonify({
            "success": True,
            "processos": processos,
            "total": len(processos)
        })
    except Exception as e:
        print(f"❌ Erro ao obter processos: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro ao obter processos: {str(e)}"
        }), 500

@rag_bp.route('/reload', methods=['POST'])
def reload():
    rag, error = get_rag()
    if error:
        return jsonify({"success": False, "message": error})
    if not rag.is_initialized:
        return jsonify({"success": False, "message": "RAG não inicializado. Execute /init primeiro."})
    def reload_background():
        from adaptive_rag import load_data_directory
        print("Recarregando documentos...")
        docs = load_data_directory()
        print(f"{docs} documentos recarregados")
    threading.Thread(target=reload_background, daemon=True).start()
    return jsonify({"success": True, "message": "Recarregamento iniciado em background"})

@rag_bp.route('/health', methods=['GET'])
def health():
    rag, error = get_rag()
    if error:
        return jsonify({"status": "error", "message": error}), 503
    if rag.is_initialized and rag.vector_store and len(rag.documents) > 0:
        return jsonify({
            "status": "ok", 
            "documents": len(rag.documents),
            "data_path": rag.data_path
        })
    else:
        return jsonify({
            "status": "not_ready",
            "initialized": rag.is_initialized,
            "has_vector_store": rag.vector_store is not None,
            "documents_count": len(rag.documents)
        }), 503

@rag_bp.route('/stats', methods=['GET'])  
def stats():
    rag, error = get_rag()
    if error:
        return jsonify({"success": False, "message": error})
    data = {
        "initialized": rag.is_initialized,
        "documents_loaded": len(rag.documents),
        "has_vector_store": rag.vector_store is not None,
        "data_path": rag.data_path,
        "cache_path": rag.cache_path
    }
    if os.path.exists(rag.data_path):
        try:
            files = [f for f in os.listdir(rag.data_path) if f.lower().endswith(('.txt', '.md'))]
            data["files_in_folder"] = len(files)
            data["sample_files"] = files[:5]
        except:
            data["files_in_folder"] = 0
    else:
        data["files_in_folder"] = 0
    return jsonify({"success": True, "data": data})