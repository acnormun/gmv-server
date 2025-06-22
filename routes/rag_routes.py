# routes/rag_routes.py
"""
Rotas RAG integradas com o sistema GMV
Usa WebSocket para progresso em tempo real
"""

from flask import Blueprint, jsonify, request, current_app
import logging
import os
import sys
import threading
import time
from datetime import datetime
from adaptive_rag import rag_status


logger = logging.getLogger(__name__)

# Blueprint RAG
rag_bp = Blueprint('rag', __name__, url_prefix='/api/rag')

def get_rag_system():
    try:
        from adaptive_rag import rag_system
        return rag_system, None
    except ImportError as e:
        return None, f"Módulo adaptive_rag não encontrado: {e}"
    except Exception as e:
        return None, f"Erro ao importar RAG: {e}"

@rag_bp.route('/status', methods=['GET'])
def rag_status():
    """Status do sistema RAG"""
    try:
        rag_system, error = get_rag_system()
        
        if error:
            return jsonify({
                "status": "offline",
                "message": error,
                "isReady": False,
                "can_initialize": False
            }), 200
        
        if rag_status:
            return jsonify({
                "status": "online",
                "message": "Sistema RAG inicializado",
                "isReady": True,
                "can_initialize": True,
                "details": {
                    "initialized": True,
                    "documents_loaded": 0,
                    "has_vector_store": True
                }
            }), 200
    except Exception as e:
        logger.error(f"Erro no status RAG: {e}")
        return jsonify({
            "status": "offline",
            "message": f"Erro interno: {str(e)}",
            "isReady": False
        }), 500


@rag_bp.route('/init', methods=['POST'])
def init_rag():
    """Inicializa o sistema RAG com progresso via WebSocket"""
    try:
        # Gera operation_id único
        operation_id = f"rag_init_{int(time.time())}"
        
        rag_system, error = get_rag_system()
        if error:
            return jsonify({
                "success": False,
                "message": error,
                "operation_id": operation_id
            }), 500
        
        # Se já inicializado
        if rag_system.is_initialized:
            return jsonify({
                "success": True,
                "message": "Sistema RAG já estava inicializado",
                "operation_id": operation_id,
                "already_initialized": True
            }), 200
        
        # Inicialização em background com progresso
        def init_with_progress():
            try:
                from utils import progress_step
                from adaptive_rag import init_rag_system, load_data_directory
                
                # Passo 1
                progress_step.emit_progress(operation_id, 1, "Conectando com Ollama...")
                
                success = init_rag_system()
                
                if not success:
                    progress_step.emit_progress(operation_id, 0, " Falha ao conectar com Ollama", error=True)
                    return
                
                # Passo 2
                progress_step.emit_progress(operation_id, 2, "Sistema RAG inicializado! Carregando documentos...")
                
                docs_loaded = load_data_directory()
                
                # Passo 3
                progress_step.emit_progress(operation_id, 3, f" Concluído! {docs_loaded} documentos carregados")
                
                logger.info(f" RAG inicializado via API: {docs_loaded} documentos")
                
            except Exception as e:
                progress_step.emit_progress(operation_id, 0, f" Erro: {str(e)}", error=True)
                logger.error(f" Erro na inicialização RAG: {e}")
        
        # Executa em thread separada
        threading.Thread(target=init_with_progress, daemon=True).start()
        
        return jsonify({
            "success": True,
            "message": "Inicialização RAG iniciada",
            "operation_id": operation_id,
            "websocket_progress": True
        }), 202
        
    except Exception as e:
        logger.error(f"Erro na inicialização RAG: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno: {str(e)}"
        }), 500


@rag_bp.route('/query', methods=['POST'])
def rag_query():
    """Executa consulta RAG"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "message": "Campo 'question' é obrigatório"
            }), 400
        
        question = data['question']
        context = data.get('context', '')
        
        rag_system, error = get_rag_system()
        if error:
            return jsonify({
                "success": False,
                "message": error
            }), 500
        
        if not rag_system.is_initialized:
            return jsonify({
                "success": False,
                "message": "RAG não inicializado. Execute /api/rag/init primeiro."
            }), 400
        
        # Executa consulta
        result = rag_system.query(question, context)
        
        if 'error' in result:
            return jsonify({
                "success": False,
                "message": result['error']
            }), 500
        
        return jsonify({
            "success": True,
            "data": result
        }), 200
        
    except Exception as e:
        logger.error(f"Erro na consulta RAG: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno: {str(e)}"
        }), 500


@rag_bp.route('/stats', methods=['GET'])
def rag_stats():
    """Estatísticas do sistema RAG"""
    try:
        rag_system, error = get_rag_system()
        if error:
            return jsonify({
                "success": False,
                "message": error
            }), 500
        
        stats = rag_system.get_stats()
        return jsonify({
            "success": True,
            "data": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao obter stats: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno: {str(e)}"
        }), 500


@rag_bp.route('/reload-documents', methods=['POST'])
def reload_documents():
    """Recarrega documentos com progresso"""
    try:
        operation_id = f"rag_reload_{int(time.time())}"
        
        rag_system, error = get_rag_system()
        if error:
            return jsonify({
                "success": False,
                "message": error,
                "operation_id": operation_id
            }), 500
        
        if not rag_system.is_initialized:
            return jsonify({
                "success": False,
                "message": "RAG não inicializado",
                "operation_id": operation_id
            }), 400
        
        # Recarregamento em background
        def reload_with_progress():
            try:
                from utils import progress_step
                from adaptive_rag import load_data_directory
                
                progress_step.emit_progress(operation_id, 1, "Recarregando documentos...")
                
                docs_loaded = load_data_directory()
                
                progress_step.emit_progress(operation_id, 2, f" {docs_loaded} documentos recarregados")
                
                logger.info(f" Documentos recarregados: {docs_loaded}")
                
            except Exception as e:
                progress_step.emit_progress(operation_id, 0, f" Erro: {str(e)}", error=True)
                logger.error(f" Erro ao recarregar documentos: {e}")
        
        threading.Thread(target=reload_with_progress, daemon=True).start()
        
        return jsonify({
            "success": True,
            "message": "Recarregamento iniciado",
            "operation_id": operation_id,
            "websocket_progress": True
        }), 202
        
    except Exception as e:
        logger.error(f"Erro ao recarregar documentos: {e}")
        return jsonify({
            "success": False,
            "message": f"Erro interno: {str(e)}"
        }), 500


@rag_bp.route('/health', methods=['GET'])
def rag_health():
    """Health check do RAG"""
    try:
        rag_system, error = get_rag_system()
        
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "module_available": error is None,
            "initialized": False,
            "working": False
        }
        
        if error:
            health_data["error"] = error
            return jsonify(health_data), 503
        
        health_data["initialized"] = rag_system.is_initialized
        
        if rag_system.is_initialized:
            try:
                test_response = rag_system.llm("health check")
                health_data["working"] = bool(test_response)
                health_data["test_response"] = test_response[:50] + "..."
            except Exception as e:
                health_data["working"] = False
                health_data["llm_error"] = str(e)
        
        status_code = 200 if health_data["working"] else 503
        return jsonify(health_data), status_code
        
    except Exception as e:
        logger.error(f"Erro no health check RAG: {e}")
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }), 500


# Rotas de diagnóstico (somente em desenvolvimento)
if os.getenv('FLASK_ENV') == 'development':
    
    @rag_bp.route('/debug', methods=['GET'])
    def rag_debug():
        """Debug do sistema RAG"""
        return jsonify({
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "python_path": sys.path[:3],
            "environment": dict(os.environ),
            "rag_module_path": None  # Será preenchido se import funcionar
        }), 200