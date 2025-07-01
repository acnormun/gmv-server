# utils/progress_step.py

import logging

logger = logging.getLogger(__name__)

# Variáveis globais para SocketIO e operation_sockets
_socketio = None
_operation_sockets = {}

def init_progress(socketio, operation_sockets):
    """
    Inicializa o sistema de progresso com SocketIO e operation_sockets
    
    Args:
        socketio: Instância do Flask-SocketIO
        operation_sockets: Dicionário que mapeia operation_id para session_id
    """
    global _socketio, _operation_sockets
    _socketio = socketio
    _operation_sockets = operation_sockets
    logger.info(" Sistema de progresso inicializado")

def send_progress_ws(operation_id, step, message, percentage):
    """
    Envia progresso via WebSocket para cliente específico
    
    Args:
        operation_id (str): ID único da operação
        step (int): Número do passo atual (0 = erro, >0 = sucesso)
        message (str): Mensagem de progresso
        percentage (int): Percentual de conclusão (0-100)
    """
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
                'success': step > 0,  # step 0 = erro, >0 = sucesso
                'error': step == 0   # Para compatibilidade
            }
            
            # Emite para cliente específico
            _socketio.emit('progress_update', progress_data, to=client_sid)
            
            logger.info(f" Progresso enviado para {operation_id}: {percentage}% - {message}")
            
            # Remove da lista se completou ou falhou
            if percentage >= 100 or step == 0:
                if operation_id in _operation_sockets:
                    del _operation_sockets[operation_id]
                    logger.info(f" Operação {operation_id} removida da lista")
        else:
            logger.warning(f" Cliente não encontrado para operação {operation_id}")
            logger.warning(f"   Sockets ativos: {list(_operation_sockets.keys())}")
            
    except Exception as e:
        logger.error(f"Erro ao enviar progresso: {e}")
        # Não falha o processamento se WebSocket der erro
        pass

def get_active_operations():
    """Retorna número de operações ativas"""
    return len(_operation_sockets)

def get_operation_sockets():
    """Retorna dicionário de operation_sockets para debug"""
    return _operation_sockets.copy()

def force_cleanup_operation(operation_id):
    """Remove operação específica da lista"""
    global _operation_sockets
    if operation_id in _operation_sockets:
        del _operation_sockets[operation_id]
        logger.info(f"Operação {operation_id} removida manualmente")
        return True
    return False