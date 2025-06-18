# routes/__init__.py

from .health import health_bp
from .triagem import triagem_bp
from .debug import debug_bp
from .rag import rag_bp

def register_routes(app, operation_sockets=None):
    """
    Registra todas as rotas da aplicação
    
    Args:
        app: Instância do Flask
        operation_sockets: Dicionário de WebSocket connections (opcional)
    """
    
    # Se operation_sockets foi fornecido, passa para o blueprint de triagem
    if operation_sockets is not None:
        triagem_bp.operation_sockets = operation_sockets
        print(f"✅ operation_sockets passado para triagem_bp: {type(operation_sockets)}")
    
    # Registra todos os blueprints (mantendo sua ordem original)
    app.register_blueprint(health_bp)
    app.register_blueprint(triagem_bp)
    app.register_blueprint(debug_bp)
    app.register_blueprint(rag_bp)
    
    print(f"✅ Todos os blueprints registrados: health, triagem, debug, rag")