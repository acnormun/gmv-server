from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua_chave_secreta'
socketio = SocketIO(app, cors_allowed_origins="*")

operation_sockets = {}

def send_progress_ws(operation_id, step, message, percentage):
    if operation_id in operation_sockets:
        socket_id = operation_sockets[operation_id]
        socketio.emit('progress_update', {
            'step': step,
            'message': message,
            'percentage': percentage,
            'operation_id': operation_id
        }, to=socket_id)