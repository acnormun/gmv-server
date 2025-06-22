from .health import health_bp
from .triagem import triagem_bp
from .rag_routes import rag_bp
from .debug import debug_bp

def register_routes(app, operation_sockets=None):
    if operation_sockets is not None:
        triagem_bp.operation_sockets = operation_sockets

    app.register_blueprint(health_bp)
    app.register_blueprint(triagem_bp)
    app.register_blueprint(debug_bp)
    app.register_blueprint(rag_bp)
    
    print(f" Todos os blueprints registrados: health, triagem, debug")