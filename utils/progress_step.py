
import logging

logger = logging.getLogger(__name__)

_socketio = None
_operation_sockets = {}

def init_progress(socketio, operation_sockets):
    global _socketio, _operation_sockets
    _socketio = socketio
    _operation_sockets = operation_sockets
    logger.info(" Sistema de progresso inicializado")

def send_progress_ws(operation_id, step, message, percentage):
    global _socketio, _operation_sockets
    try:
        if not _socketio:
            logger.warning(f" SocketIO não inicializado - Progresso apenas no log: {percentage}% - {message}")
            return
        if operation_id in _operation_sockets:
            client_sid = _operation_sockets[operation_id]
            progress_data = {
                'operation_id': operation_id,
                'step': step,
                'message': message,
                'percentage': percentage,
                'success': step > 0,
                'error': step == 0
            }
            _socketio.emit('progress_update', progress_data, to=client_sid)
            logger.info(f" Progresso enviado para {operation_id}: {percentage}% - {message}")
            if percentage >= 100 or step == 0:
                if operation_id in _operation_sockets:
                    del _operation_sockets[operation_id]
                    logger.info(f" Operação {operation_id} removida da lista")
        else:
            logger.warning(f" Cliente não encontrado para operação {operation_id}")
            logger.warning(f"   Sockets ativos: {list(_operation_sockets.keys())}")
    except Exception as e:
        logger.error(f"Erro ao enviar progresso: {e}")
        pass

def get_active_operations():
    return len(_operation_sockets)

def get_operation_sockets():
    return _operation_sockets.copy()

def force_cleanup_operation(operation_id):
    global _operation_sockets
    if operation_id in _operation_sockets:
        del _operation_sockets[operation_id]
        logger.info(f"Operação {operation_id} removida manualmente")
        return True
    return False