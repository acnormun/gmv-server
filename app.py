from flask import Flask, jsonify, request
import os, sys, logging, signal
from flask_socketio import SocketIO
from utils.auto_setup import setup_environment
from utils import progress_step
from routes import register_routes

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua_chave_secreta'

# WebSocket
socketio = SocketIO(app, cors_allowed_origins=["http://localhost:5173", "http://127.0.0.1:5173"], logger=True)
operation_sockets = {}

# NOVO: Conecta socketio e operation_sockets ao app para acesso global
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

# RAG
try:
    from utils.adaptive_rag import initialize_rag, query_rag, get_rag_statistics
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False

rag_inicializado = False
if RAG_AVAILABLE:
    try:
        rag_inicializado = initialize_rag(
            triagem_path=PATH_TRIAGEM,
            pasta_destino=PASTA_DESTINO,
            pasta_dat=PASTA_DAT
        )
    except Exception as e:
        logger.error(f"Erro ao inicializar RAG: {str(e)}")

app.config['RAG_AVAILABLE'] = RAG_AVAILABLE
app.config['RAG_INICIALIZADO'] = rag_inicializado

# CORS
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
except ImportError:
    logger.warning("flask-cors não instalado")

# NOVO: Passa operation_sockets para as rotas antes de registrá-las
with app.app_context():
    register_routes(app, operation_sockets)

# Middleware de logs para RAG
@app.before_request
def log_rag_requests():
    if request.path.startswith('/rag/'):
        logger.info(f"RAG Request: {request.method} {request.path}")
        if request.method == 'POST' and request.is_json:
            data = request.get_json()
            if data and 'query' in data:
                logger.info(f"Query: {data['query'][:50]}")

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    logger.info(f'✅ Cliente WebSocket conectado: {request.sid}')
    
@socketio.on('start_listening')
def handle_start_listening(data):  
    operation_id = data.get('operation_id')
    
    if operation_id:
        operation_sockets[operation_id] = request.sid
        logger.info(f"📡 Registrado progresso: {operation_id} -> {request.sid}")
        logger.info(f"📊 Sockets ativos: {len(operation_sockets)}")
        
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

# NOVO: Adiciona também o evento subscribe_progress para compatibilidade
@socketio.on('subscribe_progress')
def handle_subscribe_progress(data):
    operation_id = data.get('operation_id')
    
    if operation_id:
        operation_sockets[operation_id] = request.sid
        logger.info(f"📡 Cliente {request.sid} subscrito ao progresso da operação {operation_id}")
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
        
    logger.info(f'❌ Cliente desconectado: {request.sid} (removidas {len(to_remove)} operações)')

# NOVO: Health check com info do WebSocket
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "websocket": "enabled",
        "active_operations": len(operation_sockets),
        "rag_available": app.config.get('RAG_AVAILABLE', False),
        "rag_initialized": app.config.get('RAG_INICIALIZADO', False)
    })

# Finalização graciosa
def graceful_shutdown(sig, frame):
    logger.info(f"🛑 Sinal {sig} recebido. Finalizando graciosamente...")
    logger.info(f"🏁 Servidor com PID {os.getpid()} finalizado")
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

if __name__ == '__main__':
    logger.info(f"🚀 Servidor iniciado com PID {os.getpid()}")
    logger.info(f"🔗 URL: http://0.0.0.0:5000")
    logger.info(f"🔌 WebSocket: ws://0.0.0.0:5000/socket.io/")
    logger.info(f"🩺 Health: http://0.0.0.0:5000/health")
    logger.info(f"📊 Operações ativas: {len(operation_sockets)}")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)