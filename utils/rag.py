# utils/rag.py
from flask import jsonify, current_app
from datetime import datetime
from core.rag_core import (
    RAG_AVAILABLE, rag_inicializado,
    query_rag, initialize_rag,
    get_rag_statistics
)

def handle_rag_status():
    if not RAG_AVAILABLE:
        return jsonify({'available': False, 'initialized': False, 'error': 'Sistema RAG não foi carregado'}), 503
    if not rag_inicializado:
        return jsonify({'available': True, 'initialized': False, 'error': 'Sistema RAG não foi inicializado'}), 503
    return jsonify({'available': True, 'initialized': True, 'statistics': get_rag_statistics()}), 200

def handle_rag_query(data):
    if not RAG_AVAILABLE or not rag_inicializado:
        return jsonify({'error': 'Sistema RAG não disponível ou não inicializado'}), 503
    if not data or 'query' not in data:
        return jsonify({'error': 'Parâmetro query é obrigatório'}), 400

    query_text = data.get('query', '').strip()
    k = data.get('k', 5)
    if not query_text:
        return jsonify({'error': 'Query não pode estar vazia'}), 400
    if not isinstance(k, int) or not (1 <= k <= 20):
        k = 5

    resultado = query_rag(query_text, k=k)
    return jsonify(resultado), 200

def handle_rag_reinitialize():
    global rag_inicializado
    if not RAG_AVAILABLE:
        return jsonify({'error': 'Sistema RAG não disponível'}), 503

    sucesso = initialize_rag(
        triagem_path=current_app.config['PATH_TRIAGEM'],
        pasta_destino=current_app.config['PASTA_DESTINO'],
        pasta_dat=current_app.config['PASTA_DAT']
    )
    rag_inicializado = sucesso

    if sucesso:
        return jsonify({
            'success': True,
            'message': 'Sistema RAG reinicializado com sucesso',
            'statistics': get_rag_statistics()
        }), 200
    else:
        return jsonify({'success': False, 'error': 'Falha na reinicialização do sistema RAG'}), 500

def handle_rag_search(data):
    if not RAG_AVAILABLE or not rag_inicializado:
        return jsonify({'error': 'Sistema RAG não disponível ou não inicializado'}), 503
    if not data:
        return jsonify({'error': 'Dados não fornecidos'}), 400

    query_text = data.get('query', '').strip()
    filtros = data.get('filters', {})
    k = data.get('k', 5)
    if not query_text:
        return jsonify({'error': 'Query é obrigatória'}), 400

    if filtros:
        partes = []
        if filtros.get('tema'):
            partes.append(f"tema: {filtros['tema']}")
        if filtros.get('status'):
            partes.append(f"status: {filtros['status']}")
        if filtros.get('responsavel'):
            partes.append(f"responsável: {filtros['responsavel']}")
        if filtros.get('suspeitos'):
            partes.append(f"suspeitos: {filtros['suspeitos']}")
        if partes:
            query_text = f"{query_text} considerando {', '.join(partes)}"

    resultado = query_rag(query_text, k=k)
    resultado['filters_applied'] = filtros
    resultado['original_query'] = data.get('query', '')
    return jsonify(resultado), 200

def handle_rag_statistics():
    if not RAG_AVAILABLE or not rag_inicializado:
        return jsonify({'error': 'Sistema RAG não disponível ou não inicializado'}), 503
    stats = get_rag_statistics()
    stats['system_info'] = {
        'rag_available': RAG_AVAILABLE,
        'rag_initialized': rag_inicializado,
        'paths': {
            'triagem': current_app.config['PATH_TRIAGEM'],
            'processos': current_app.config['PASTA_DESTINO'],
            'dat': current_app.config['PASTA_DAT']
        }
    }
    return jsonify(stats), 200

def handle_rag_suggestions():
    if not RAG_AVAILABLE or not rag_inicializado:
        return jsonify({'error': 'Sistema RAG não disponível ou não inicializado'}), 503
    stats = get_rag_statistics()
    suggestions = []

    if 'tema_distribution' in stats:
        for tema in list(stats['tema_distribution'].keys())[:3]:
            suggestions.append({
                'type': 'factual',
                'query': f"Quais processos estão relacionados ao tema {tema}?",
                'category': 'Consulta por Tema'
            })

    if 'status_distribution' in stats:
        for status in list(stats['status_distribution'].keys())[:3]:
            suggestions.append({
                'type': 'analytical',
                'query': f"Analise os processos com status {status}",
                'category': 'Análise por Status'
            })

    if 'top_suspeitos' in stats:
        for s in list(stats['top_suspeitos'].keys())[:2]:
            suggestions.append({
                'type': 'contextual',
                'query': f"Quais processos envolvem {s}?",
                'category': 'Consulta por Suspeito'
            })

    suggestions += [
        {'type': 'analytical', 'query': 'Compare a distribuição de processos por tema', 'category': 'Análise Geral'},
        {'type': 'opinion', 'query': 'Qual a tendência dos processos investigados?', 'category': 'Análise de Tendências'},
        {'type': 'contextual', 'query': 'Identifique padrões nos processos suspeitos', 'category': 'Identificação de Padrões'},
        {'type': 'factual', 'query': 'Quantos processos estão em investigação?', 'category': 'Consulta Quantitativa'}
    ]

    return jsonify({
        'suggestions': suggestions[:10],
        'total_suggestions': len(suggestions),
        'based_on_data': {
            'total_documents': stats.get('total_documents', 0),
            'unique_themes': len(stats.get('tema_distribution', {})),
            'unique_status': len(stats.get('status_distribution', {}))
        }
    }), 200

def handle_rag_health():
    status = 'healthy' if RAG_AVAILABLE and rag_inicializado else 'unhealthy'
    result = {
        'status': status,
        'rag_available': RAG_AVAILABLE,
        'rag_initialized': rag_inicializado,
        'timestamp': datetime.now().isoformat()
    }
    if status == 'healthy':
        stats = get_rag_statistics()
        result.update({
            'documents_loaded': stats.get('total_documents', 0),
            'chunks_processed': stats.get('total_chunks', 0),
            'cache_size': stats.get('cache_size', 0)
        })
    return jsonify(result), 200 if status == 'healthy' else 503


def handle_rag_analyze_processo(numero):
    if not RAG_AVAILABLE or not rag_inicializado:
        return jsonify({
            'error': 'Sistema RAG não disponível para análise'
        }), 503

    try:
        query_text = f"Analise detalhadamente o processo {numero} incluindo status, tema, suspeitos e evidências"
        resultado = query_rag(query_text, k=3)

        return jsonify({
            'processo': numero,
            'analise': resultado,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': f'Erro ao consultar RAG: {str(e)}'}), 500
