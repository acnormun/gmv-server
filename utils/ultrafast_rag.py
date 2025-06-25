# utils/ultrafast_rag_optimized.py

import os
import re
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache
import concurrent.futures
import threading

import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import local
from utils.optimized_vector_store import OptimizedVectorStore

# Import condicional para Ollama
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
    except ImportError:
        try:
            from langchain.llms import Ollama as OllamaLLM
        except ImportError:
            class OllamaLLM:
                def __init__(self, *args, **kwargs):
                    print("⚠️ Usando OllamaLLM mock")
                def invoke(self, prompt): return "Resposta mock - Ollama não disponível"
                def __call__(self, prompt): return "Resposta mock - Ollama não disponível"


@dataclass
class UltraFastRAGConfig:
    model_name: str = "mistral:7b-instruct"
    temperature: float = 0.0
    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 7
    max_chunks: int = 50 
    data_dir: str = "data"
    use_ollama_embeddings: bool = True
    enable_conversational: bool = True
    max_context_length: int = 8192 
    num_predict: int = 1024
    enable_cache: bool = True
    cache_ttl: int = 3600 
    min_similarity_score: float = 0.15
    enable_parallel_search: bool = True
    enable_preprocessing: bool = True


class UltraFastRAG:
    def __init__(self, config: Optional[UltraFastRAGConfig] = None):
        self.config = config or UltraFastRAGConfig()
        self.llm = None
        self.vector_store = None
        self.documents = []
        self.is_initialized = False
        self.conversational_handler = None
        
        # Cache em memória para respostas
        self.response_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Lock para thread safety
        self.cache_lock = threading.Lock()
        
        # Caminhos configuráveis
        self.data_path = os.getenv("DADOS_ANONIMOS", 
                                  os.getenv("PASTA_DESTINO", 
                                           self.config.data_dir))
        self.cache_path = os.path.join(os.path.dirname(self.data_path), ".rag_cache")
        os.makedirs(self.cache_path, exist_ok=True)
        
        # Splitter otimizado para chunks menores
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        # Padrões de preprocessamento compilados
        self._compile_preprocessing_patterns()
    
    def _compile_preprocessing_patterns(self):
        """Compila padrões regex para preprocessamento rápido"""
        self.cleanup_patterns = [
            (re.compile(r'\s+'), ' '),  # Múltiplos espaços
            (re.compile(r'\n+'), '\n'),  # Múltiplas quebras
            (re.compile(r'[^\w\s\.\?\!\,\:\;\-\(\)]', re.UNICODE), ''),  # Caracteres especiais
        ]
        
        self.query_patterns = [
            (re.compile(r'\b(qual|quais|como|quando|onde|por que|porque)\b', re.IGNORECASE), ''),
            (re.compile(r'\b(me|nos|lhe|te)\s+(ajude|ajudar|diga|dizer|fale|falar)\b', re.IGNORECASE), ''),
            (re.compile(r'\b(por favor|obrigad[oa]|valeu)\b', re.IGNORECASE), ''),
        ]
    
    @lru_cache(maxsize=1000)
    def _preprocess_query(self, query: str) -> str:
        """Preprocessa query para melhor busca (com cache)"""
        if not self.config.enable_preprocessing:
            return query
        
        # Remove palavras de cortesia e melhora a query
        processed = query.strip().lower()
        
        for pattern, replacement in self.query_patterns:
            processed = pattern.sub(replacement, processed)
        
        # Remove espaços extras
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        # Mantém palavras-chave importantes
        keywords = [word for word in processed.split() 
                   if len(word) > 2 and word not in ['para', 'com', 'por', 'sobre']]
        
        return ' '.join(keywords) if keywords else query
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Gera chave única para cache"""
        content = f"{query}_{top_k}_{self.config.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Busca resposta no cache"""
        if not self.config.enable_cache:
            return None
        
        with self.cache_lock:
            if cache_key in self.response_cache:
                cached_data, timestamp = self.response_cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl:
                    self.cache_stats["hits"] += 1
                    return cached_data
                else:
                    # Remove cache expirado
                    del self.response_cache[cache_key]
        
        self.cache_stats["misses"] += 1
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Salva resposta no cache"""
        if not self.config.enable_cache:
            return
        
        with self.cache_lock:
            # Limita tamanho do cache
            if len(self.response_cache) > 500:
                # Remove entradas mais antigas
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k][1])
                del self.response_cache[oldest_key]
            
            self.response_cache[cache_key] = (response, time.time())
    
    def initialize(self):
        """Inicializa o sistema RAG otimizado"""
        try:
            print("🚀 Inicializando UltraFast RAG Otimizado...")
            
            # Inicializa LLM com configurações otimizadas
            try:
                self.llm = OllamaLLM(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.num_predict,  # Reduzido para velocidade
                    # Parâmetros otimizados para velocidade
                    repeat_penalty=1.1,
                    top_k=10,
                    top_p=0.9,
                )
                
                # Teste rápido de conexão
                test_response = self.llm.invoke("OK")
                print(f"✅ LLM {self.config.model_name} otimizado conectado")
                
            except Exception as e:
                print(f"❌ Erro na conexão LLM: {e}")
                return False
            
            self.is_initialized = True
            self._load_cache()
            
            # Integra sistema conversacional se habilitado
            if self.config.enable_conversational:
                try:
                    self._integrate_conversational()
                except Exception as e:
                    print(f"⚠️ Erro na integração conversacional: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro crítico na inicialização: {e}")
            self.is_initialized = False
            return False
    
    def _integrate_conversational(self):
        """Integra sistema conversacional de forma otimizada"""
        try:
            import sys
            if 'adaptive_rag' in sys.modules:
                module = sys.modules['adaptive_rag']
                if hasattr(module, 'enhance_rag_with_conversation'):
                    enhance_rag_with_conversation = module.enhance_rag_with_conversation
                    enhance_rag_with_conversation(self)
                    print("✅ Sistema conversacional integrado!")
        except Exception as e:
            print(f"⚠️ Erro ao integrar conversacional: {e}")
    
    def _load_cache(self):
        """Carrega cache de forma otimizada"""
        try:
            self.vector_store = OptimizedVectorStore(
                os.path.join(self.cache_path, "optimized"),
                use_ollama=self.config.use_ollama_embeddings
            )
            
            if self.vector_store.load():
                try:
                    self.documents = []
                    for i, (doc_content, metadata) in enumerate(zip(self.vector_store.documents, self.vector_store.metadata)):
                        self.documents.append(Document(page_content=doc_content, metadata=metadata))
                    print(f"📦 Cache carregado: {len(self.documents)} documentos")
                except Exception as e:
                    print(f"⚠️ Erro ao processar cache: {e}")
                    self.documents = []
            else:
                self.documents = []
                
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}")
            self.vector_store = None
    
    def load_documents_from_directory(self):
        """Carrega documentos de forma otimizada"""
        if not os.path.exists(self.data_path):
            print(f"⚠️ Diretório não encontrado: {self.data_path}")
            return 0
        
        print(f"📁 Carregando documentos otimizado de: {self.data_path}")
        
        documents = []
        try:
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.lower().endswith(('.txt', '.md')):
                        try:
                            filepath = os.path.join(root, file)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            if len(content) < 100:
                                continue
                            
                            # Parse de metadados otimizado
                            metadata = self._parse_legal_metadata_fast(content)
                            metadata['filename'] = file
                            metadata['source'] = filepath
                            
                            # Limpeza otimizada do conteúdo
                            clean_content = self._clean_content_fast(content)
                            
                            # Chunking otimizado
                            chunks = self.text_splitter.split_text(clean_content)
                            for chunk in chunks:
                                if len(chunk.strip()) > 100:
                                    documents.append(Document(
                                        page_content=chunk.strip(),
                                        metadata=metadata.copy()
                                    ))
                            
                        except Exception as e:
                            continue
            
            if documents:
                print(f"📄 {len(documents)} chunks carregados (otimizado)")
                self.documents = documents
                self._create_vector_store()
            
            return len(documents)
            
        except Exception as e:
            print(f"❌ Erro ao carregar documentos: {e}")
            return 0
    
    @lru_cache(maxsize=100)
    def _parse_legal_metadata_fast(self, content: str) -> Dict[str, str]:
        """Parse otimizado de metadados com cache"""
        metadata = {}
        
        # Parse mais direto com regex
        patterns = {
            'numero_processo': r'numero_processo[:\s]+["\']?([^"\'\n]+)["\']?',
            'agravante': r'agravante[:\s]+["\']?([^"\'\n]+)["\']?',
            'agravado': r'agravado[:\s]+["\']?([^"\'\n]+)["\']?',
            'valor_causa': r'valor_causa[:\s]+["\']?([^"\'\n]+)["\']?',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
        
        return metadata
    
    def _clean_content_fast(self, content: str) -> str:
        """Limpeza otimizada de conteúdo"""
        # Remove seções de metadados
        content = re.sub(r"### METADADOS DO PROCESSO.*?(?=\n###|\Z)", "", content, flags=re.DOTALL)
        content = re.sub(r"### PARTES ENVOLVIDAS.*?(?=\n###|\Z)", "", content, flags=re.DOTALL)
        
        # Limpeza básica
        for pattern, replacement in self.cleanup_patterns:
            content = pattern.sub(replacement, content)
        
        return content.strip()
    
    def _create_vector_store(self):
        """Cria vector store otimizado"""
        if not self.documents:
            return
        
        try:
            print("🗃️ Criando vector store otimizado...")
            
            self.vector_store = OptimizedVectorStore(
                os.path.join(self.cache_path, "optimized"),
                use_ollama=self.config.use_ollama_embeddings
            )
            
            # Limita documentos para performance
            max_docs = min(len(self.documents), self.config.max_chunks)
            self.vector_store.add_documents(self.documents[:max_docs], max_docs=max_docs)
            print("✅ Vector store otimizado criado!")
            
        except Exception as e:
            print(f"❌ Erro ao criar vector store: {e}")
            self.vector_store = None
    
    def _search_documents_optimized(self, query: str, top_k: int) -> List[Document]:
        """Busca otimizada de documentos"""
        if not self.vector_store:
            return []
        
        # Preprocessa query
        processed_query = self._preprocess_query(query)
        
        if self.config.enable_parallel_search:
            # Busca paralela com diferentes estratégias
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Busca principal
                future_main = executor.submit(
                    self.vector_store.similarity_search,
                    processed_query, top_k, self.config.min_similarity_score
                )
                
                # Busca de fallback
                future_fallback = executor.submit(
                    self.vector_store.similarity_search,
                    query, max(1, top_k//2), 0.01
                )
                
                main_results = future_main.result()
                
                if main_results:
                    return main_results
                else:
                    return future_fallback.result()
        else:
            # Busca sequencial
            results = self.vector_store.similarity_search(
                processed_query, top_k, self.config.min_similarity_score
            )
            
            if not results:
                results = self.vector_store.similarity_search(query, top_k, 0.01)
            
            return results
    
    def _create_optimized_context(self, docs: List[Document]) -> str:
        """Cria contexto otimizado e compacto"""
        if not docs:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            
            # Limita tamanho do contexto individual
            if len(content) > 400:
                content = content[:400] + "..."
            
            # Adiciona informações essenciais
            doc_info = f"DOC{i}: {content}"
            
            if total_length + len(doc_info) > self.config.max_context_length:
                break
                
            context_parts.append(doc_info)
            total_length += len(doc_info)
        
        return "\n\n".join(context_parts)
    
    def _create_optimized_prompt(self, question: str, context: str) -> str:
        """Cria prompt otimizado e mais direto"""
        # Template muito mais conciso
        return f"""Baseado nos documentos, responda objetivamente:

DOCUMENTOS:
{context}

PERGUNTA: {question}

RESPOSTA (seja direto e preciso):"""
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Query otimizada com cache e paralelização"""
        if not self.is_initialized or not self.vector_store:
            return {"error": "Sistema não inicializado"}
        
        k = top_k or self.config.top_k
        
        # Verifica cache primeiro
        cache_key = self._get_cache_key(question, k)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            cached_result["from_cache"] = True
            return cached_result
        
        try:
            start_time = time.time()
            
            # Busca otimizada de documentos
            relevant_docs = self._search_documents_optimized(question, k)
            
            if not relevant_docs:
                result = {
                    "error": "Nenhum documento relevante encontrado",
                    "suggestion": "Tente reformular a pergunta",
                    "processing_time": time.time() - start_time
                }
                self._save_to_cache(cache_key, result)
                return result
            
            # Cria contexto otimizado
            context = self._create_optimized_context(relevant_docs)
            
            # Cria prompt otimizado
            prompt = self._create_optimized_prompt(question, context)
            
            # Chama LLM de forma otimizada
            if hasattr(self.llm, 'invoke'):
                answer = self.llm.invoke(prompt)
            else:
                answer = self.llm(prompt)
            
            result = {
                "answer": answer.strip(),
                "documents_found": len(relevant_docs),
                "processing_time": time.time() - start_time,
                "from_cache": False,
                "cache_stats": self.cache_stats.copy()
            }
            
            # Salva no cache
            self._save_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            error_result = {"error": str(e), "processing_time": time.time() - start_time}
            return error_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""
        return {
            "cache_stats": self.cache_stats.copy(),
            "cache_size": len(self.response_cache),
            "documents_loaded": len(self.documents) if self.documents else 0,
            "config": {
                "num_predict": self.config.num_predict,
                "max_context_length": self.config.max_context_length,
                "top_k": self.config.top_k,
                "chunk_size": self.config.chunk_size,
                "enable_cache": self.config.enable_cache,
                "enable_parallel_search": self.config.enable_parallel_search,
                "enable_preprocessing": self.config.enable_preprocessing
            }
        }
    
    def clear_cache(self):
        """Limpa cache de respostas"""
        with self.cache_lock:
            self.response_cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
        print("🗑️ Cache limpo")


# =================================================================================
# INSTÂNCIA GLOBAL E FUNÇÕES DE INTERFACE OTIMIZADAS
# =================================================================================

# Instância global otimizada
optimized_rag_system = None

def init_optimized_rag():
    """Inicializa sistema RAG otimizado"""
    global optimized_rag_system
    
    config = UltraFastRAGConfig(
        model_name="mistral:7b-instruct",
        enable_cache=True,
        enable_parallel_search=True,
        enable_preprocessing=True,
        num_predict=200,  # Reduzido para velocidade
        max_context_length=2000,  # Limitado para velocidade
        top_k=3  # Reduzido para velocidade
    )
    
    optimized_rag_system = OptimizedUltraFastRAG(config)
    return optimized_rag_system.initialize()

def load_optimized_data():
    """Carrega dados de forma otimizada"""
    if optimized_rag_system:
        return optimized_rag_system.load_documents_from_directory()
    return 0

def query_optimized_rag(question: str, top_k: int = 3):
    """Interface otimizada para consultas"""
    if optimized_rag_system:
        return optimized_rag_system.query(question, top_k)
    return {"error": "Sistema não inicializado"}

def get_optimized_performance_stats():
    """Estatísticas de performance"""
    if optimized_rag_system:
        return optimized_rag_system.get_performance_stats()
    return {"error": "Sistema não inicializado"}

def clear_optimized_cache():
    """Limpa cache do sistema otimizado"""
    if optimized_rag_system:
        optimized_rag_system.clear_cache()


# =================================================================================
# TESTE DE PERFORMANCE
# =================================================================================

def performance_comparison_test():
    """Teste de comparação de performance"""
    print("🏃‍♂️ TESTE DE PERFORMANCE - RAG OTIMIZADO")
    print("=" * 60)
    
    if not optimized_rag_system or not optimized_rag_system.is_initialized:
        print("❌ Sistema otimizado não inicializado")
        return
    
    # Queries de teste
    test_queries = [
        "processo 1005888",
        "terapia ABA",
        "valor da causa",
        "agravante agravado",
        "SUS tratamento"
    ]
    
    print("🧪 Testando velocidade das consultas...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- TESTE {i} ---")
        print(f"Query: {query}")
        
        start_time = time.time()
        result = optimized_rag_system.query(query)
        end_time = time.time()
        
        if "error" not in result:
            print(f"✅ Resposta obtida")
            print(f"⏱️ Tempo: {end_time - start_time:.2f}s")
            print(f"📊 Docs encontrados: {result.get('documents_found', 0)}")
            print(f"💾 Do cache: {result.get('from_cache', False)}")
        else:
            print(f"❌ Erro: {result['error']}")
    
    # Estatísticas finais
    print(f"\n📈 ESTATÍSTICAS FINAIS:")
    stats = optimized_rag_system.get_performance_stats()
    print(f"Cache hits: {stats['cache_stats']['hits']}")
    print(f"Cache misses: {stats['cache_stats']['misses']}")
    if stats['cache_stats']['hits'] + stats['cache_stats']['misses'] > 0:
        hit_rate = stats['cache_stats']['hits'] / (stats['cache_stats']['hits'] + stats['cache_stats']['misses'])
        print(f"Taxa de acerto do cache: {hit_rate:.2%}")

if __name__ == "__main__":
    print("🚀 SISTEMA RAG ULTRA OTIMIZADO")
    print("=" * 50)
    
    # Inicializa sistema otimizado
    if init_optimized_rag():
        print("✅ Sistema otimizado inicializado")
        
        # Carrega dados
        docs_loaded = load_optimized_data()
        if docs_loaded > 0:
            print(f"✅ {docs_loaded} documentos carregados")
            
            # Executa teste de performance
            performance_comparison_test()
        else:
            print("⚠️ Nenhum documento carregado")
    else:
        print("❌ Falha na inicialização do sistema otimizado")