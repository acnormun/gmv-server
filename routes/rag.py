from flask import Blueprint, jsonify, request, current_app
import os
from utils.rag import AdaptiveRAG

rag_bp = Blueprint('rag', __name__)
_rag_instance: AdaptiveRAG | None = None

def _get_rag() -> AdaptiveRAG:
    global _rag_instance
    if _rag_instance is None:
        db_path = current_app.config.get('RAG_DB_PATH', './data/rag_base')
        model = current_app.config.get('OLLAMA_MODEL', 'llama3')
        _rag_instance = AdaptiveRAG(db_path, model)
    return _rag_instance

@rag_bp.route('/rag/status', methods=['GET'])
def rag_status():
    rag = _get_rag()
    return jsonify({'status': 'ok', 'documents': len(rag.docs)})

@rag_bp.route('/rag/suggestions', methods=['GET'])
def rag_suggestions():
    rag = _get_rag()
    names = [os.path.basename(d['path']) for d in rag.docs]
    return jsonify({'documents': names})

@rag_bp.route('/rag/query', methods=['POST'])
def rag_query():
    data = request.get_json() or {}
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'question required'}), 400
    rag = _get_rag()
    result = rag.chat(question)
    return jsonify(result)
