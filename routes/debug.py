from flask import Blueprint, current_app
from utils.debug import debug_sockets, test_progress

debug_bp = Blueprint('debug', __name__)

@debug_bp.route('/debug/sockets')
def sockets():
    operation_sockets = current_app.config.get('OPERATION_SOCKETS', {})
    return debug_sockets(operation_sockets)

@debug_bp.route('/debug/test-progress/<operation_id>')
def test_progress_route(operation_id):
    operation_sockets = current_app.config.get('OPERATION_SOCKETS', {})
    return test_progress(operation_id, operation_sockets)
