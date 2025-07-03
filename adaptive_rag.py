import os
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
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

@dataclass
class UltraFastRAGConfig:
    model_name: str = "qwen3:1.7b"
    temperature: float = 0.0
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 6
    max_chunks: int = 60
    data_dir: str = "data"
    use_ollama_embeddings: bool = True
    enable_conversational: bool = True

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

def test_search_detailed(question: str):
    if hasattr(rag_system, 'test_search_detailed'):
        return rag_system.test_search_detailed(question)
    return {"error": "Teste detalhado não disponível"}

def quick_test():
    try:
        test_queries = [
            "triagem",
            "processo",
            "agravo"
        ]
        for query in test_queries:
            try:
                result = rag_system.query(query)
                docs_found = result.get('documents_found', 0)
                has_answer = len(result.get('answer', '')) > 0
            except Exception:
                pass
        processes = list_available_processes(5)
        return True
    except Exception:
        return False

def test_conversational_responses():
    if not ConversationalLayer:
        return
    conv_layer = ConversationalLayer()
    test_cases = [
        "Oi!",
        "Bom dia!",
        "Como você está?",
        "Obrigado pela ajuda",
        "Você pode me ajudar?",
        "Tchau!",
        "Qual seu nome?",
        "Oi, bom dia! Preciso saber sobre um processo específico",
        "Olá! Gostaria de entender sobre TEA e terapia ABA",
        "Valeu pelas informações! Qual o valor da causa do processo 1005888?"
    ]
    for test in test_cases:
        conv_type = conv_layer.detect_conversation_type(test)
        should_conv = conv_layer.should_use_conversational_response(test)
        if conv_type and should_conv:
            response = conv_layer.generate_conversational_response(test, conv_type)
        else:
            pass

if __name__ == "__main__":
    test_conversational_responses()
    if UltraFastRAG:
        test_dir = "data"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            test_content = """### METADADOS DO PROCESSO
numero_processo: "1005888-76.2025.8.11.0000"
data_distribuicao: "27/02/2025"
valor_causa: "R$ 1.518,00"
assuntos: "Liminar, Multas e demais Sanções, Tratamento médico-hospitalar"

### PARTES ENVOLVIDAS
agravante: "Y. F. O."
agravado: "MUNICIPIO DE SINOP, ESTADO DE MATO GROSSO"
terceiro_interessado: "MINISTERIO PUBLICO DO ESTADO DE MATO GROSSO"

### CONTEÚDO PRINCIPAL

Trata-se de Recurso de Agravo de Instrumento interposto por Y.F.O., representado por sua genitora, em face da decisão da Vara Especializada da Infância e Juventude.

O agravante foi diagnosticado com Transtorno do Espectro Autista (TEA) CID 10 - F84.0 e necessita de:
- Fonoaudiologia especializada em TEA (3x por semana)
- Terapia ocupacional especializada (3x por semana) 
- Psicoterapia comportamental tipo ABA (3x por semana)
- Atendimento com neuropediatra
- Professor de apoio especializado
- Psicopedagoga (3x por semana)

PARECER NAT-JUS:
A terapia ABA não tem embasamento científico robusto e os procedimentos são de caráter eletivo. Estudos indicam que não há evidência de superioridade da ABA sobre alternativas terapêuticas, além do alto custo individual.

O SUS disponibiliza fisioterapia, psicoterapia, fonoaudiologia e terapia ocupacional. A psicopedagogia é responsabilidade da Secretaria Municipal de Educação.

A responsabilidade dos entes federativos na saúde é solidária, podendo qualquer um ser demandado para fornecer os serviços.
"""
            with open(os.path.join(test_dir, "processo_teste.md"), "w", encoding="utf-8") as f:
                f.write(test_content)
        config = UltraFastRAGConfig(
            model_name="qwen3:1.7b",
            temperature=0.1,
            data_dir=test_dir,
            use_ollama_embeddings=True,
            enable_conversational=True
        )
        test_rag = UltraFastRAG(config)
        if test_rag.initialize():
            docs_loaded = test_rag.load_documents_from_directory()
            if docs_loaded > 0:
                queries = [
                    "Oi! Bom dia!",
                    "Qual é o número do processo e quem são as partes envolvidas?",
                    "Obrigado! A terapia ABA é disponibilizada pelo SUS?",
                    "Quais tratamentos o agravante necessita?",
                    "Valeu pelas informações! Tchau!"
                ]
                for query in queries:
                    result = test_rag.query(query)
        else:
            pass
    else:
        pass