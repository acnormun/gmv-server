socketio_instance = None
operation_sockets = {}

def init_progress(socketio, sockets_dict):
    """Inicializa as variáveis globais do progresso"""
    global socketio_instance, operation_sockets
    socketio_instance = socketio
    operation_sockets = sockets_dict
    print("📡 Sistema de progresso WebSocket inicializado")

def send_progress_ws(operation_id, step, message, percentage):
    """Envia progresso via WebSocket"""
    global socketio_instance, operation_sockets
    
    try:
        if socketio_instance and operation_id in operation_sockets:
            socket_id = operation_sockets[operation_id]
            
            data = {
                'step': step,
                'message': message,
                'percentage': percentage,
                'operation_id': operation_id,
                'error': step == 0 and percentage == 0
            }
            
            socketio_instance.emit('progress_update', data, to=socket_id)
            print(f"📡 Progresso enviado para {socket_id}: Etapa {step} - {message} ({percentage}%)")
            
        else:
            print(f"⚠️ Socket não encontrado para operation_id: {operation_id}")
            if not socketio_instance:
                print("❌ SocketIO não inicializado")
            if operation_id not in operation_sockets:
                print(f"❌ Operation ID {operation_id} não registrado")
                print(f"🔍 Sockets ativos: {list(operation_sockets.keys())}")
                
    except Exception as e:
        print(f"❌ Erro ao enviar progresso: {e}")
        import traceback
        traceback.print_exc()
