from flask import Flask, jsonify, request
import os, sys, logging, signal, threading, time
from flask_socketio import SocketIO
from utils.auto_setup import setup_environment
from utils import progress_step
from routes import register_routes

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua_chave_secreta'
app.config['THREADED'] = True
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

socketio = SocketIO(
    app,
    cors_allowed_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    logger=False,
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10**8
)

operation_sockets = {}
socket_lock = threading.RLock()

app.socketio = socketio
app.operation_sockets = operation_sockets

progress_step.init_progress(socketio, operation_sockets)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)-12s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()
app.config['PATH_TRIAGEM'] = PATH_TRIAGEM
app.config['PASTA_DESTINO'] = PASTA_DESTINO
app.config['PASTA_DAT'] = PASTA_DAT

try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
    logger.info("CORS configurado")
except ImportError:
    logger.warning("flask-cors não instalado")

with app.app_context():
    register_routes(app, operation_sockets)
    logger.info("Rotas registradas")

def get_system_stats():
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'active_threads': threading.active_count(),
            'active_operations': len(operation_sockets)
        }
    except Exception:
        return {
            'active_threads': threading.active_count(),
            'active_operations': len(operation_sockets)
        }

@app.route('/api/system/stats', methods=['GET'])
def system_stats():
    stats = get_system_stats()
    return jsonify({'success': True, 'data': stats})

@app.route('/api/system/health', methods=['GET'])
def health_check():
    stats = get_system_stats()
    is_healthy = True
    issues = []
    if stats.get('memory_percent', 0) > 90:
        is_healthy = False
        issues.append('Memória alta')
    if stats.get('active_threads', 0) > 50:
        is_healthy = False
        issues.append('Muitas threads ativas')
    return jsonify({
        'healthy': is_healthy,
        'issues': issues,
        'timestamp': time.time(),
        'stats': stats
    })

@socketio.on('connect')
def handle_connect():
    logger.info(f"Cliente conectado: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Cliente desconectado: {request.sid}")
    with socket_lock:
        to_remove = [op_id for op_id, sid in operation_sockets.items() if sid == request.sid]
        for op_id in to_remove:
            del operation_sockets[op_id]
            logger.info(f"Removida operação órfã: {op_id[:8]}")

@socketio.on('start_listening')
def handle_start_listening(data):
    operation_id = data.get('operation_id')
    if operation_id:
        with socket_lock:
            operation_sockets[operation_id] = request.sid
        logger.info(f"Registrado: {operation_id[:8]} -> {request.sid}")
        try:
            from routes.triagem import triagem_bp
            if hasattr(triagem_bp, 'register_websocket'):
                triagem_bp.register_websocket(operation_id, request.sid)
        except Exception as e:
            logger.warning(f"Aviso: Não foi possível registrar no triagem: {e}")
        socketio.emit('progress_update', {
            'step': 1,
            'message': 'Canal ativo - processamento iniciado!',
            'percentage': 0,
            'operation_id': operation_id,
            'success': True
        }, to=request.sid)

@socketio.on('get_system_stats')
def handle_get_system_stats():
    stats = get_system_stats()
    socketio.emit('system_stats', stats, to=request.sid)

def background_cleanup():
    while True:
        try:
            stats = get_system_stats()
            socketio.emit('system_stats', stats)
            with socket_lock:
                if len(operation_sockets) > 50:
                    old_ops = list(operation_sockets.keys())[:10]
                    for op_id in old_ops:
                        del operation_sockets[op_id]
                        logger.info(f"Limpeza automática: {op_id[:8]}")
            time.sleep(30)
        except Exception as e:
            logger.error(f"Erro em background cleanup: {e}")
            time.sleep(60)

def init_rag_background():
    try:
        from adaptive_rag import rag_system, init_rag_system
        logger.info("Tentando inicializar RAG...")
        if init_rag_system():
            logger.info("RAG inicializado com sucesso")
            socketio.emit('rag_status', {
                'status': 'ready',
                'message': 'Sistema RAG online'
            })
        else:
            logger.warning("RAG falhou - Ollama pode estar offline")
            socketio.emit('rag_status', {
                'status': 'offline',
                'message': 'RAG offline - Execute: ollama serve'
            })
    except ImportError:
        logger.info("RAG não disponível")
    except Exception as e:
        logger.error(f"Erro RAG: {e}")
        socketio.emit('rag_status', {
            'status': 'error',
            'message': str(e)
        })

def signal_handler(signum, frame):
    logger.info(f"Sinal {signum} recebido, encerrando...")
    try:
        from routes.triagem import get_thread_pool
        pool = get_thread_pool()
        if pool:
            logger.info("Aguardando threads finalizarem...")
            pool.shutdown(wait=True, timeout=10)
            logger.info("Threads finalizadas")
    except Exception as e:
        logger.warning(f"Aviso no shutdown: {e}")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    logger.info("Iniciando servidor com processamento paralelo...")
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    logger.info("Background cleanup iniciado")
    rag_thread = threading.Thread(target=init_rag_background, daemon=True)
    rag_thread.start()
    logger.info("Inicialização RAG em background")
    logger.info("Servidor pronto!")
    logger.info("Processamento paralelo: HABILITADO")
    logger.info("WebSocket: Modo threading otimizado")
    logger.info("Monitoramento: /api/system/health")
    socketio.run(
        app,
        debug=False,
        host='0.0.0.0',
        port=5000,
        use_reloader=False,
        log_output=False
    )