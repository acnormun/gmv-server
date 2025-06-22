# routes/health.py
from flask import Blueprint, jsonify, current_app
from datetime import datetime
import os

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "pid": os.getpid(),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "env_vars": {
            "PATH_TRIAGEM": current_app.config.get('PATH_TRIAGEM'),
            "PASTA_DESTINO": current_app.config.get('PASTA_DESTINO'),
            "PASTA_DAT": current_app.config.get('PASTA_DAT'),
            "RAG_DB_PATH": current_app.config.get('RAG_DB_PATH')
        }
    }), 200


@health_bp.route('/process-info', methods=['GET'])
def process_info():
    return jsonify({
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, 'getppid') else None,
        "cwd": os.getcwd(),
        "env_vars": {
            "PATH_TRIAGEM": current_app.config.get('PATH_TRIAGEM'),
            "PASTA_DESTINO": current_app.config.get('PASTA_DESTINO'),
            "PASTA_DAT": current_app.config.get('PASTA_DAT'),
            "RAG_DB_PATH": current_app.config.get('RAG_DB_PATH')
        }
    }), 200