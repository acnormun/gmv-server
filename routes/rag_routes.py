# routes/rag_routes.py - VERSÃO ULTRA-RÁPIDA SEM LOOPS

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
    
    # Função para init em background
    def init_background():
        from adaptive_rag import init_rag_system, load_data_directory
        
        print("Inicializando RAG...")
        if init_rag_system():
            print("Carregando documentos...")
            docs = load_data_directory()
            print(f"{docs} documentos carregados")
        else:
            print("Falha na inicialização")
    
    # Executa em background
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
    
    # Info da pasta
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