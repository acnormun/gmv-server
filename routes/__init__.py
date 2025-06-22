# routes/__init__.py

from .health import health_bp
from .triagem import triagem_bp
from .debug import debug_bp
from .rag import rag_bp


def register_routes(app, operation_sockets=None):
    """Registra todos os blueprints da aplicação."""

    if operation_sockets is not None:
        triagem_bp.operation_sockets = operation_sockets
        print(f" operation_sockets passado para triagem_bp: {type(operation_sockets)}")

    app.register_blueprint(health_bp)
    app.register_blueprint(triagem_bp)
    app.register_blueprint(debug_bp)
    app.register_blueprint(rag_bp)

    print(" Todos os blueprints registrados: health, triagem, debug, rag")
