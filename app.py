# app.py - Versão corrigida com sistema de progresso adaptativo

from flask import Flask, jsonify, request
import os, sys, logging, signal, threading, time
from flask_socketio import SocketIO
from utils.auto_setup import setup_environment
from utils import progress_step
from routes import register_routes

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua_chave_secreta'

# WebSocket
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5173", "http://127.0.0.1:5173"], logger=True)
operation_sockets = {}

# NOVO: Conecta socketio e operation_sockets ao app para acesso 
app.socketio = socketio
app.operation_sockets = operation_sockets

# Inicializa o sistema de progresso
progress_step.init_progress(socketio, operation_sockets)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('server.log'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Setup de ambiente
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()
app.config['PATH_TRIAGEM'] = PATH_TRIAGEM
app.config['PASTA_DESTINO'] = PASTA_DESTINO
app.config['PASTA_DAT'] = PASTA_DAT

# CORS
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
except ImportError:
    logger.warning("flask-cors não instalado")

# NOVO: Passa operation_sockets para as rotas antes de registrá-las
with app.app_context():
    register_routes(app, operation_sockets)

# === SISTEMA DE PROGRESSO ADAPTATIVO ===
def emit_progress_safe(operation_id, step, message, error=False):
    """Emite progresso de forma segura, adaptando ao sistema existente"""
    try:
        # Verifica se progress_step tem emit_progress
        if hasattr(progress_step, 'emit_progress'):
            progress_step.emit_progress(operation_id, step, message, error=error)
        else:
            # Fallback: emite diretamente via socketio
            percentage = (step / 5) * 100 if step > 0 else 0
            
            socketio.emit('progress_update', {
                'step': step,
                'message': message,
                'percentage': percentage,
                'operation_id': operation_id,
                'error': error
            })
            
            logger.info(f" Progresso {operation_id}: {message}")
            
    except Exception as e:
        # Se falhar, pelo menos loga
        logger.info(f" {operation_id}: {message}")
        logger.debug(f"Erro ao emitir progresso: {e}")

# === INICIALIZAÇÃO AUTOMÁTICA DO RAG ===
def init_rag_on_startup():
    def rag_startup_task():
        time.sleep(3)
        
        operation_id = "rag_startup"
        
        try:
            logger.info(" Iniciando RAG automaticamente...")
            
            emit_progress_safe(operation_id, 1, "Verificando sistema RAG...")
            try:
                from adaptive_rag import rag_system, init_rag_system, load_data_directory
                emit_progress_safe(operation_id, 2, " Módulo RAG encontrado")
            except ImportError as e:
                emit_progress_safe(operation_id, 0, f" Módulo RAG não encontrado: {e}", error=True)
                logger.warning(f" RAG não disponível: {e}")
                return
            
            if rag_system.is_initialized:
                emit_progress_safe(operation_id, 5, " RAG já estava inicializado")
                logger.info(" RAG já estava inicializado")
                
                socketio.emit('rag_ready', {
                    'status': 'online',
                    'documents_loaded': len(rag_system.documents) if rag_system.documents else 0,
                    'message': 'Sistema RAG já estava pronto!'
                })
                return
            
            emit_progress_safe(operation_id, 3, "Conectando com Ollama...")
            
            success = init_rag_system()
            
            if success:
                emit_progress_safe(operation_id, 4, " RAG inicializado! Carregando documentos...")
                
                docs_loaded = load_data_directory()
                
                emit_progress_safe(operation_id, 5, f" RAG pronto! {docs_loaded} documentos carregados")
                
                logger.info(f" RAG inicializado automaticamente: {docs_loaded} documentos")
                
                socketio.emit('rag_ready', {
                    'status': 'online',
                    'documents_loaded': docs_loaded,
                    'message': 'Sistema RAG pronto para uso!'
                })
                
            else:
                emit_progress_safe(operation_id, 0, " Falha ao inicializar RAG - Verifique se Ollama está rodando", error=True)
                logger.warning(" RAG não inicializou - Ollama pode estar offline")
                logger.warning(" Execute: ollama serve")
                
                socketio.emit('rag_error', {
                    'status': 'offline',
                    'message': 'RAG não inicializou - Ollama offline?',
                    'suggestion': 'Execute: ollama serve'
                })
                
        except Exception as e:
            emit_progress_safe(operation_id, 0, f" Erro na inicialização: {str(e)}", error=True)
            logger.error(f" Erro na inicialização automática do RAG: {e}")
            
            socketio.emit('rag_error', {
                'status': 'error',
                'message': str(e)
            })

    threading.Thread(target=rag_startup_task, daemon=True).start()

@socketio.on('connect')
def handle_connect():
    logger.info(f' Cliente WebSocket conectado: {request.sid}')
    
@socketio.on('start_listening')
def handle_start_listening(data):  
    operation_id = data.get('operation_id')
    
    if operation_id:
        operation_sockets[operation_id] = request.sid
        logger.info(f" Registrado progresso: {operation_id} -> {request.sid}")
        logger.info(f"Sockets ativos: {len(operation_sockets)}")
        
        # Confirma registro
        socketio.emit('progress_update', {
            'step': 1,
            'message': 'Canal de progresso ativo! Aguardando processamento...',
            'percentage': 5,
            'operation_id': operation_id,
            'error': False
        }, to=request.sid)
        
    else:
        socketio.emit('error', {'message': 'operation_id é obrigatório'}, to=request.sid)

# NOVO: Handler específico para RAG status
@socketio.on('check_rag_status')
def handle_check_rag_status():
    """Verifica status do RAG e envia para o cliente"""
    try:
        from adaptive_rag import rag_system
        
        status = {
            'status': 'online' if rag_system.is_initialized else 'offline',
            'initialized': rag_system.is_initialized,
            'documents_count': len(rag_system.documents) if rag_system.documents else 0
        }
        
        socketio.emit('rag_status_update', status, to=request.sid)
        
    except ImportError:
        socketio.emit('rag_status_update', {
            'status': 'unavailable',
            'message': 'Módulo RAG não encontrado'
        }, to=request.sid)
    except Exception as e:
        socketio.emit('rag_status_update', {
            'status': 'error',
            'message': str(e)
        }, to=request.sid)

@socketio.on('subscribe_progress')
def handle_subscribe_progress(data):
    operation_id = data.get('operation_id')
    
    if operation_id:
        operation_sockets[operation_id] = request.sid
        logger.info(f" Cliente {request.sid} subscrito ao progresso da operação {operation_id}")
        socketio.emit('progress_subscribed', {'operation_id': operation_id}, to=request.sid)
    else:
        socketio.emit('error', {'message': 'operation_id é obrigatório'}, to=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    # Remove connections órfãs
    to_remove = []
    for op_id, sid in operation_sockets.items():
        if sid == request.sid:
            to_remove.append(op_id)
    
    for op_id in to_remove:
        del operation_sockets[op_id]
        
    logger.info(f' Cliente desconectado: {request.sid} (removidas {len(to_remove)} operações)')

# Health check atualizado com info do RAG
@app.route('/health', methods=['GET'])
def health_check():
    rag_status = "unknown"
    
    try:
        from adaptive_rag import rag_system
        rag_status = "online" if rag_system.is_initialized else "offline"
    except ImportError:
        rag_status = "unavailable"
    except Exception:
        rag_status = "error"
    
    return jsonify({
        "status": "ok",
        "websocket": "enabled",
        "active_operations": len(operation_sockets),
        "rag_status": rag_status
    })

# NOVO: Endpoint para verificar se RAG está pronto
@app.route('/api/system/status', methods=['GET'])
def system_status():
    """Status completo do sistema"""
    
    # Status básico
    status = {
        "backend": "online",
        "websocket": "enabled",
        "active_operations": len(operation_sockets)
    }
    
    # Status do RAG
    try:
        from adaptive_rag import rag_system
        
        if rag_system.is_initialized:
            # Testa se está funcionando
            try:
                rag_system.llm("test")
                status["rag"] = {
                    "status": "online",
                    "initialized": True,
                    "working": True,
                    "documents": len(rag_system.documents) if rag_system.documents else 0
                }
            except Exception:
                status["rag"] = {
                    "status": "offline",
                    "initialized": True,
                    "working": False,
                    "error": "Communication error"
                }
        else:
            status["rag"] = {
                "status": "offline",
                "initialized": False,
                "working": False
            }
            
    except ImportError:
        status["rag"] = {
            "status": "unavailable",
            "error": "Module not found"
        }
    except Exception as e:
        status["rag"] = {
            "status": "error",
            "error": str(e)
        }
    
    return jsonify(status)

# NOVO: Função auxiliar para emitir progresso no contexto das rotas
app.emit_progress_safe = emit_progress_safe

# Finalização graciosa
def graceful_shutdown(sig, frame):
    logger.info(f"Sinal {sig} recebido. Finalizando graciosamente...")
    logger.info(f"Servidor com PID {os.getpid()} finalizado")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

if __name__ == '__main__':
    logger.info(f" Servidor iniciado com PID {os.getpid()}")
    logger.info(f" URL: http://0.0.0.0:5000")
    logger.info(f" WebSocket: ws://0.0.0.0:5000/socket.io/")
    logger.info(f" Health: http://0.0.0.0:5000/health")
    logger.info(f" System Status: http://0.0.0.0:5000/api/system/status")
    logger.info(f" Operações ativas: {len(operation_sockets)}")
    
    # === INICIA RAG AUTOMATICAMENTE ===
    init_rag_on_startup()
    
    # Inicia o servidor
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)