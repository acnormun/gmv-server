
"""
Pacote utils para o sistema RAG modular

Contém:
- conversational_layer: Camada para respostas conversacionais
- smart_tfidf_embedder: Embeddings TF-IDF otimizados
- optimized_vector_store: Vector store com cache inteligente  
- smart_rag_handler: Handler inteligente para RAG conversacional
- ultrafast_rag: Sistema RAG principal
"""

__version__ = "1.0.0"
__all__ = []

def safe_import(module_name, class_name):
    try:
        module = __import__(f'utils.{module_name}', fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        print(f"⚠️ {class_name} não disponível: {e}")
        return None
    except Exception as e:
        print(f"❌ Erro ao importar {class_name}: {e}")
        return None

ConversationalLayer = safe_import('conversational_layer', 'ConversationalLayer')
if ConversationalLayer:
    __all__.append('ConversationalLayer')

SmartTFIDFEmbedder = safe_import('smart_tfidf_embedder', 'SmartTFIDFEmbedder')
if SmartTFIDFEmbedder:
    __all__.append('SmartTFIDFEmbedder')

OptimizedVectorStore = safe_import('optimized_vector_store', 'OptimizedVectorStore')
if OptimizedVectorStore:
    __all__.append('OptimizedVectorStore')

SmartRAGHandler = safe_import('smart_rag_handler', 'SmartRAGHandler')
if SmartRAGHandler:
    __all__.append('SmartRAGHandler')

try:
    UltraFastRAG = safe_import('ultrafast_rag', 'UltraFastRAG')
    UltraFastRAGConfig = safe_import('ultrafast_rag', 'UltraFastRAGConfig')
    
    if UltraFastRAG:
        __all__.append('UltraFastRAG')
    if UltraFastRAGConfig:
        __all__.append('UltraFastRAGConfig')
        
except Exception as e:
    print(f"⚠️ UltraFastRAG componentes não disponíveis: {e}")

print(f"📦 Utils carregado com {len(__all__)} componentes: {__all__}")