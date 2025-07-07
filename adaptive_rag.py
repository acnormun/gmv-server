import os
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv

from utils.ultrafast_rag import UltraFastRAGConfig
load_dotenv()

try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        try:
            from langchain.llms import Ollama as OllamaLLM
            from langchain.embeddings import OllamaEmbeddings
        except ImportError:
            class OllamaLLM:
                def __init__(self, *args, **kwargs):
                    pass
                def invoke(self, prompt): return "Resposta mock - Ollama não disponível"
                def __call__(self, prompt): return "Resposta mock - Ollama não disponível"
            class OllamaEmbeddings:
                def __init__(self, *args, **kwargs):
                    pass
                def embed_query(self, text): return np.random.random(384).tolist()
                def embed_documents(self, docs): return [np.random.random(384).tolist() for _ in docs]

try:
    from utils.conversational_layer import ConversationalLayer
except ImportError:
    ConversationalLayer = None

try:
    from utils.smart_rag_handler import SmartRAGHandler
except ImportError:
    SmartRAGHandler = None

try:
    from utils.ultrafast_rag import UltraFastRAG
except ImportError:
    UltraFastRAG = None

def enhance_rag_with_conversation(rag_system):
    if not SmartRAGHandler:
        return rag_system
    try:
        original_query_method = rag_system.query
        smart_handler = SmartRAGHandler(rag_system)
        def base_rag_query(question: str, top_k: int = 4):
            try:
                return original_query_method(question, top_k)
            except Exception as e:
                return {"error": f"Erro no processamento básico: {str(e)}"}
        smart_handler.rag_system.base_query = base_rag_query
        def enhanced_query(question: str, top_k: int = 4, user_id: str = "default"):
            try:
                return smart_handler.process_query(question, user_id)
            except Exception:
                return base_rag_query(question, top_k)
        rag_system.query = enhanced_query
        rag_system.conversational_handler = smart_handler
    except Exception:
        pass
    return rag_system

if UltraFastRAG:
    rag_system = UltraFastRAG()
else:
    class MockRAG:
        def __init__(self):
            self.is_initialized = False
            self.documents = []
            self.vector_store = None
            self.data_path = os.getenv("PASTA_DESTINO", "data")
            self.config = UltraFastRAGConfig()
        def initialize(self):
            return False
        def load_documents_from_directory(self):
            return 0
        def query(self, question, top_k=4):
            return {"error": "Sistema RAG não disponível"}
    rag_system = MockRAG()

def init_rag_system():
    return rag_system.initialize()

def load_data_directory():
    return rag_system.load_documents_from_directory()

def get_rag_status():
    if not rag_system.is_initialized:
        return {
            "status": "offline",
            "message": "Sistema não inicializado",
            "isReady": False,
            "data_path": getattr(rag_system, 'data_path')
        }
    if not rag_system.vector_store or len(rag_system.documents) == 0:
        return {
            "status": "offline",
            "message": "Nenhum documento carregado",
            "isReady": False,
            "data_path": getattr(rag_system, 'data_path')
        }
    embedding_method = "TF-IDF"
    conversational_status = "Inativo"
    model_name = "N/A"
    if hasattr(rag_system, 'config'):
        if rag_system.config.use_matryoshka_embeddings:
            embedding_method = "Matryoshka"
        else:
            embedding_method = "Híbrido (Ollama + TF-IDF)" if rag_system.config.use_ollama_embeddings else "TF-IDF"
        conversational_status = "Ativo" if rag_system.config.enable_conversational and hasattr(rag_system, 'conversational_handler') else "Inativo"
        model_name = rag_system.config.model_name
    return {
        "status": "online",
        "message": f"{len(rag_system.documents)} documentos carregados",
        "isReady": True,
        "documents_loaded": len(rag_system.documents),
        "embedding_method": embedding_method,
        "conversational": conversational_status,
        "model": model_name,
        "data_path": getattr(rag_system, 'data_path', 'data')
    }

def query_rag(question: str, top_k: int = 4):
    return rag_system.query(question, top_k)

def debug_rag_search(question: str):
    if hasattr(rag_system, 'debug_search'):
        return rag_system.debug_search(question)
    return {"error": "Debug não disponível"}

def list_available_processes(limit: int = 20):
    if hasattr(rag_system, 'list_available_processes'):
        return rag_system.list_available_processes(limit)
    return []

def get_sample_documents(limit: int = 5):
    try:
        if not rag_system.documents:
            return []
        samples = []
        for i, doc in enumerate(rag_system.documents[:limit]):
            samples.append({
                "index": i,
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "content_length": len(doc.page_content)
            })
        return samples
    except Exception as e:
        return [{"error": str(e)}]