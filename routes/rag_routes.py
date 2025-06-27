from typing import Any, Dict, List
from flask import Blueprint, jsonify, request
import threading
import time
import os

rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')

def get_rag():
    try:
        from adaptive_rag import rag_system
        return rag_system, None
    except Exception as e:
        return None, str(e)

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
    start_time = time.time()
    result = rag.query(question)
    processing_time = time.time() - start_time
    if 'error' in result:
        return jsonify({"success": False, "message": result['error']})
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
        import time
        start_time = time.time()
        print(f"⚡ CONSULTA RÁPIDA:")
        print(f"   ❓ Pergunta: {question}")
        print(f"   ⚖️ Processos: {processos_selecionados}")
        if not processos_selecionados:
            return rag.query(question, top_k=k)
        relevant_docs = []
        for processo in processos_selecionados:
            print(f"🔍 Busca direta: {processo}")
            try:
                docs_encontrados = rag.vector_store.similarity_search(processo, k=3)
                print(f"   📄 Encontrados: {len(docs_encontrados)} docs")
                for doc in docs_encontrados:
                    metadata = getattr(doc, 'metadata', {})
                    content = getattr(doc, 'page_content', '')
                    if processo in metadata.get('numero_processo', ''):
                        relevant_docs.append(doc)
                        print(f"   ✅ Doc relevante adicionado (metadata)")
                        continue
                    if processo in content:
                        relevant_docs.append(doc)
                        print(f"   ✅ Doc relevante adicionado (conteúdo)")
            except Exception as e:
                print(f"   ❌ Erro na busca: {e}")
                continue
        print(f"📋 Total documentos relevantes: {len(relevant_docs)}")
        if not relevant_docs:
            print("⚠️ Nenhum documento específico encontrado - busca híbrida")
            for processo in processos_selecionados:
                try:
                    docs_ampla = rag.vector_store.similarity_search(question, k=5)
                    for doc in docs_ampla:
                        if processo in getattr(doc, 'page_content', ''):
                            relevant_docs.append(doc)
                            print(f"   ✅ Doc híbrido encontrado")
                            break
                except:
                    continue
        if not relevant_docs:
            print("⚠️ Fallback para busca geral")
            result = rag.query(question, top_k=k)
            if 'answer' in result:
                aviso = f"\n\n⚠️ Não foram encontrados documentos específicos para: {', '.join(processos_selecionados)}"
                result['answer'] += aviso
            return result
        if len(relevant_docs) > 5:
            relevant_docs = relevant_docs[:5]
            print(f"🔄 Limitado a 5 documentos para velocidade")
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            doc_text = f"DOCUMENTO {i}:\n"
            if 'numero_processo' in metadata:
                doc_text += f"Processo: {metadata['numero_processo']}\n"
            doc_text += f"{content}"
            context_parts.append(doc_text)
        context = "\n\n" + "-"*30 + "\n\n".join(context_parts)
        prompt = f"""Responda baseado nos documentos do processo {', '.join(processos_selecionados)}:

PERGUNTA: {question}

DOCUMENTOS:
{context}

RESPOSTA:"""
        try:
            response = rag.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"❌ Erro no LLM: {e}")
            answer = f"Erro ao gerar resposta: {str(e)}"
        processing_time = time.time() - start_time
        source_documents = []
        for doc in relevant_docs:
            source_documents.append({
                "content": doc.page_content[:300] + "...",
                "filename": doc.metadata.get('numero_processo', 'Desconhecido'),
                "metadata": {"numero_processo": doc.metadata.get('numero_processo')}
            })
        print(f"⚡ Consulta concluída em {processing_time:.2f}s")
        return {
            "answer": answer,
            "question": question,
            "context_size": len(context),
            "documents_count": len(relevant_docs),
            "processing_time": processing_time,
            "search_method": "ultra_fast",
            "source_documents": source_documents,
            "context_info": {
                "documentos_selecionados": [],
                "processos_selecionados": processos_selecionados,
                "total_filtered_docs": len(relevant_docs),
                "relevant_chunks": len(relevant_docs)
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