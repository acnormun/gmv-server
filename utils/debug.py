from flask import jsonify
from utils.progress_step import send_progress_ws

def debug_sockets(operation_sockets):
    return jsonify({
        'active_sockets': operation_sockets,
        'socket_count': len(operation_sockets),
        'keys': list(operation_sockets.keys())
    })

def test_progress(operation_id, operation_sockets):
    send_progress_ws(operation_id, 2, 'Teste manual de progresso', 50)
    return jsonify({
        'message': f'Progresso teste enviado para {operation_id}',
        'socket_registered': operation_id in operation_sockets,
        'socket_id': operation_sockets.get(operation_id, 'Não encontrado')
    })
