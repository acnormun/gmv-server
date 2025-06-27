import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# =================================================================================
# IMPORTS CONDICIONAIS PARA LANGCHAIN/OLLAMA
# =================================================================================

try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
    print("✅ Usando langchain_ollama")
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        from langchain_community.embeddings import OllamaEmbeddings
        print("✅ Usando langchain_community")
    except ImportError:
        try:
            from langchain.llms import Ollama as OllamaLLM
            from langchain.embeddings import OllamaEmbeddings
            print("✅ Usando langchain legacy")
        except ImportError as e:
            print(f"❌ Erro crítico: Não foi possível importar Ollama: {e}")
            class OllamaLLM:
                def __init__(self, *args, **kwargs):
                    print("⚠️ Usando OllamaLLM mock")
                def invoke(self, prompt): return "Resposta mock - Ollama não disponível"
                def __call__(self, prompt): return "Resposta mock - Ollama não disponível"
            
            class OllamaEmbeddings:
                def __init__(self, *args, **kwargs):
                    print("⚠️ Usando OllamaEmbeddings mock")
                def embed_query(self, text): return np.random.random(384).tolist()
                def embed_documents(self, docs): return [np.random.random(384).tolist() for _ in docs]

# =================================================================================
# IMPORTS DOS UTILS COM TRATAMENTO DE ERRO
# =================================================================================

try:
    from utils.conversational_layer import ConversationalLayer
    print("✅ ConversationalLayer importado")
except ImportError as e:
    print(f"⚠️ ConversationalLayer não encontrado: {e}")
    ConversationalLayer = None

try:
    from utils.smart_rag_handler import SmartRAGHandler
    print("✅ SmartRAGHandler importado")
except ImportError as e:
    print(f"⚠️ SmartRAGHandler não encontrado: {e}")
    SmartRAGHandler = None

try:
    from utils.ultrafast_rag import UltraFastRAG
    print("✅ UltraFastRAG importado")
except ImportError as e:
    print(f"⚠️ UltraFastRAG não encontrado: {e}")
    UltraFastRAG = None

# =================================================================================
# CONFIGURAÇÃO
# =================================================================================

@dataclass
class UltraFastRAGConfig:
    model_name: str = "gemma:2b"
    temperature: float = 0.0           # Lowered for deterministic, factual output
    chunk_size: int = 800              # Smaller chunks improve retrieval accuracy
    chunk_overlap: int = 200           # Maintains context across sections
    top_k: int = 6                     # More candidates = better recall for precision
    max_chunks: int = 60               # Limits context to most relevant chunks
    data_dir: str = "data"
    use_ollama_embeddings: bool = True
    enable_conversational: bool = True  # Keep enabled for multi-turn interactions

# =================================================================================
# FUNÇÃO DE INTEGRAÇÃO CONVERSACIONAL
# =================================================================================

def enhance_rag_with_conversation(rag_system):
    """Melhora o sistema RAG com capacidades conversacionais"""
    if not SmartRAGHandler:
        print("⚠️ SmartRAGHandler não disponível - pulando integração conversacional")
        return rag_system
    
    try:
        # Salva referência para o método original
        original_query_method = rag_system.query
        
        # Cria handler inteligente
        smart_handler = SmartRAGHandler(rag_system)
        
        # Método que usa o original diretamente (evita recursão)
        def base_rag_query(question: str, top_k: int = 4) -> Dict[str, Any]:
            """Chama o método RAG original diretamente"""
            try:
                return original_query_method(question, top_k)
            except Exception as e:
                print(f"❌ Erro no base_rag_query: {e}")
                return {"error": f"Erro no processamento básico: {str(e)}"}
        
        # Atualiza o handler para usar o método base
        smart_handler.rag_system.base_query = base_rag_query
        
        def enhanced_query(question: str, top_k: int = 4, user_id: str = "default") -> Dict[str, Any]:
            try:
                return smart_handler.process_query(question, user_id)
            except Exception as e:
                print(f"❌ Erro no processamento conversacional: {e}")
                # Usa o método base em caso de erro
                return base_rag_query(question, top_k)
        
        rag_system.query = enhanced_query
        rag_system.conversational_handler = smart_handler
        print("✅ Sistema RAG melhorado com capacidades conversacionais!")
        
    except Exception as e:
        print(f"❌ Erro ao integrar sistema conversacional: {e}")
        print("🔄 Continuando com sistema RAG básico")
    
    return rag_system

# =================================================================================
# INSTÂNCIA GLOBAL E FUNÇÕES DE INTERFACE
# =================================================================================

# Cria instância apenas se UltraFastRAG foi importado com sucesso
if UltraFastRAG:
    rag_system = UltraFastRAG()
else:
    print("❌ UltraFastRAG não disponível - criando mock")
    class MockRAG:
        def __init__(self):
            self.is_initialized = False
            self.documents = []
            self.vector_store = None
            self.data_path = "data"
            self.config = UltraFastRAGConfig()
        
        def initialize(self):
            print("⚠️ Mock RAG - inicialização simulada")
            return False
        
        def load_documents_from_directory(self):
            print("⚠️ Mock RAG - carregamento simulado")
            return 0
        
        def query(self, question, top_k=4):
            return {"error": "Sistema RAG não disponível"}
    
    rag_system = MockRAG()

def init_rag_system():
    """Inicializa o sistema RAG"""
    print("🚀 Inicializando sistema RAG...")
    return rag_system.initialize()

def load_data_directory():
    """Carrega documentos do diretório"""
    print("📂 Carregando diretório de dados...")
    return rag_system.load_documents_from_directory()

def get_rag_status():
    """Retorna status do sistema RAG"""
    if not rag_system.is_initialized:
        return {
            "status": "offline", 
            "message": "Sistema não inicializado", 
            "isReady": False,
            "data_path": getattr(rag_system, 'data_path', 'data')
        }
    
    if not rag_system.vector_store or len(rag_system.documents) == 0:
        return {
            "status": "offline", 
            "message": "Nenhum documento carregado", 
            "isReady": False,
            "data_path": getattr(rag_system, 'data_path', 'data')
        }
    
    # Verifica se tem os atributos necessários
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
    """Interface para consultas RAG"""
    return rag_system.query(question, top_k)

def debug_rag_search(question: str):
    """Interface para debug de busca"""
    if hasattr(rag_system, 'debug_search'):
        return rag_system.debug_search(question)
    return {"error": "Debug não disponível"}

def list_available_processes(limit: int = 20):
    """Lista processos disponíveis"""
    if hasattr(rag_system, 'list_available_processes'):
        return rag_system.list_available_processes(limit)
    return []

def get_sample_documents(limit: int = 5):
    """Retorna amostra de documentos para debug"""
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
    """Teste detalhado de busca para debug"""
    if hasattr(rag_system, 'test_search_detailed'):
        return rag_system.test_search_detailed(question)
    return {"error": "Teste detalhado não disponível"}

def quick_test():
    """Teste rápido do sistema RAG"""
    try:
        print("\n🚀 TESTE RÁPIDO DO RAG")
        print("=" * 50)
        
        # Status básico
        print(f"Sistema inicializado: {rag_system.is_initialized}")
        print(f"Documentos carregados: {len(rag_system.documents) if rag_system.documents else 0}")
        print(f"Vector store disponível: {rag_system.vector_store is not None}")
        
        # Teste de queries simples
        test_queries = [
            "triagem",
            "processo",
            "agravo"
        ]
        
        print("\n🔍 TESTANDO QUERIES SIMPLES:")
        for query in test_queries:
            try:
                result = rag_system.query(query)
                docs_found = result.get('documents_found', 0)
                has_answer = len(result.get('answer', '')) > 0
                print(f"   '{query}': {docs_found} docs, resposta: {'✅' if has_answer and 'error' not in result else '❌'}")
            except Exception as e:
                print(f"   '{query}': ❌ Erro - {e}")
        
        # Lista alguns processos
        print(f"\n📋 PROCESSOS DISPONÍVEIS:")
        processes = list_available_processes(5)
        for proc in processes:
            print(f"   - {proc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste rápido: {e}")
        return False

# =================================================================================
# FUNÇÃO DE TESTE
# =================================================================================

def test_conversational_responses():
    """Teste específico para respostas conversacionais"""
    if not ConversationalLayer:
        print("⚠️ ConversationalLayer não disponível para teste")
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
    
    print("🧪 TESTE DE RESPOSTAS CONVERSACIONAIS")
    print("=" * 60)
    
    for test in test_cases:
        conv_type = conv_layer.detect_conversation_type(test)
        should_conv = conv_layer.should_use_conversational_response(test)
        print(f"\n📝 Input: '{test}'")
        print(f"🎯 Tipo: {conv_type}")
        print(f"🗣️ Conversacional: {should_conv}")
        
        if conv_type and should_conv:
            response = conv_layer.generate_conversational_response(test, conv_type)
            print(f"💬 Resposta: {response}")
        else:
            print(f"🔍 Resposta: [Processamento técnico RAG]")

# =================================================================================
# TESTE STANDALONE
# =================================================================================

if __name__ == "__main__":
    print("🧪 TESTE DO SISTEMA RAG ADAPTATIVO MODULAR")
    print("=" * 60)
    
    # Primeiro testa as respostas conversacionais
    test_conversational_responses()
    
    print("\n" + "="*60)
    
    # Testa apenas se as classes estão disponíveis
    if UltraFastRAG:
        # Cria diretório de teste se não existir
        test_dir = "data"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
            
            # Cria documento de teste
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
            print(f"✅ Documento de teste criado em {test_dir}")
        
        # Configura sistema com conversacional habilitado
        config = UltraFastRAGConfig(
            model_name="gemma:2b",
            temperature=0.1,
            data_dir=test_dir,
            use_ollama_embeddings=True,
            enable_conversational=True
        )
        
        test_rag = UltraFastRAG(config)
        
        print("\n🔄 Inicializando sistema...")
        if test_rag.initialize():
            print("✅ Sistema inicializado")
            
            print("\n📂 Carregando documentos...")
            docs_loaded = test_rag.load_documents_from_directory()
            print(f"✅ {docs_loaded} documentos carregados")
            
            if docs_loaded > 0:
                print("\n🧪 Executando testes mistos (conversacional + técnico)...")
                
                queries = [
                    "Oi! Bom dia!",  # Conversacional
                    "Qual é o número do processo e quem são as partes envolvidas?",  # Técnica
                    "Obrigado! A terapia ABA é disponibilizada pelo SUS?",  # Híbrida
                    "Quais tratamentos o agravante necessita?",  # Técnica
                    "Valeu pelas informações! Tchau!"  # Conversacional
                ]
                
                for i, query in enumerate(queries, 1):
                    print(f"\n--- TESTE {i} ---")
                    print(f"Pergunta: {query}")
                    
                    result = test_rag.query(query)
                    
                    if "error" in result:
                        print(f"❌ Erro: {result['error']}")
                    else:
                        print(f"Resposta: {result['answer']}")
                        print(f"Tipo: {result.get('type', 'técnico')}")
                        if result.get('sources'):
                            print(f"Fontes: {result['sources']}")
                        if result.get('documents_found'):
                            print(f"Documentos: {result['documents_found']}")
            else:
                print("⚠️ Nenhum documento carregado para teste")
        else:
            print("❌ Falha na inicialização")
            print("💡 Certifique-se que o Ollama está rodando:")
            print("   ollama serve")
            print("   ollama pull gemma:2b")
            print("   ollama pull nomic-embed-text")
    else:
        print("❌ UltraFastRAG não disponível - pulando testes completos")
        print("💡 Verifique se os arquivos utils estão corretos")