# routes/rag.py

from flask import Blueprint, request
from utils.rag import (
    handle_rag_status,
    handle_rag_query,
    handle_rag_reinitialize,
    handle_rag_search,
    handle_rag_statistics,
    handle_rag_suggestions,
    handle_rag_health
)

rag_bp = Blueprint('rag', __name__)

@rag_bp.route('/rag/status', methods=['GET'])
def rag_status():
    return handle_rag_status()

@rag_bp.route('/rag/query', methods=['POST'])
def rag_query():
    data = request.get_json()
    return handle_rag_query(data)

@rag_bp.route('/rag/reinitialize', methods=['POST'])
def rag_reinitialize():
    return handle_rag_reinitialize()

@rag_bp.route('/rag/search', methods=['POST'])
def rag_search():
    data = request.get_json()
    return handle_rag_search(data)

@rag_bp.route('/rag/statistics', methods=['GET'])
def rag_statistics():
    return handle_rag_statistics()

@rag_bp.route('/rag/suggestions', methods=['GET'])
def rag_suggestions():
    return handle_rag_suggestions()

@rag_bp.route('/rag/health', methods=['GET'])
def rag_health():
    return handle_rag_health()
