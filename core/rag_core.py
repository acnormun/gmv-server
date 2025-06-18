from utils.adaptive_rag import initialize_rag, query_rag, get_rag_statistics

# Flags globais
RAG_AVAILABLE = True
rag_inicializado = False

__all__ = [
    "RAG_AVAILABLE",
    "rag_inicializado",
    "initialize_rag",
    "query_rag",
    "get_rag_statistics"
]
