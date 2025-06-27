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
    model_name: str = "gemma:2b"  # ⚡ SEU MODELO
    temperature: float = 0.0
    chunk_size: int = 2000  # ⚡ AUMENTADO de 800
    chunk_overlap: int = 400  # ⚡ AUMENTADO de 200
    top_k: int = 8  # ⚡ AUMENTADO de 7
    max_chunks: int = 1000  # ⚡ AUMENTADO de 50 (MUITO IMPORTANTE!)
    data_dir: str = "data"
    use_ollama_embeddings: bool = True
    enable_conversational: bool = True
    max_context_length: int = 20000  # ⚡ AUMENTADO de 8192
    num_predict: int = 1500  # ⚡ AUMENTADO de 1024
    enable_cache: bool = True
    cache_ttl: int = 3600 
    min_similarity_score: float = 0.001  # ⚡ REDUZIDO de 0.15
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
            
            # Inicializa LLM com configurações para contexto completo
            try:
                self.llm = OllamaLLM(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.num_predict,  # ⚡ Agora 1500
                    num_ctx=8192,  # ⚡ Contexto máximo
                    repeat_penalty=1.1,
                    top_k=40,  # ⚡ Aumentado
                    top_p=0.9,
                    stop=[],  # ⚡ Sem limitações
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
        """Busca MUITO mais permissiva e abrangente"""
        if not self.vector_store:
            return []
        
        print(f"🔍 Busca híbrida para: '{query}'")
        
        # === BUSCA 1: KEYWORDS ESPECÍFICAS ===
        keyword_results = []
        
        # Keywords jurídicas específicas
        legal_keywords = [
            'argumento', 'argumenta', 'sustenta', 'alega', 'defesa', 
            'motivação', 'fundamento', 'razão', 'motivo',
            'pad', 'processo administrativo', 'disciplinar', 'demissão',
            'cerceamento', 'contraditório', 'ampla defesa',
            'tutela', 'liminar', 'urgência', 'recurso', 'agravo'
        ]
        
        # Identifica keywords relevantes
        query_lower = query.lower()
        relevant_keywords = []
        
        for keyword in legal_keywords:
            if keyword in query_lower:
                relevant_keywords.append(keyword)
        
        # Se não achou keywords específicas, usa palavras da query
        if not relevant_keywords:
            relevant_keywords = [word for word in query_lower.split() if len(word) > 3]
        
        print(f"📝 Buscando por: {relevant_keywords}")
        
        # Busca keywords nos documentos
        for doc in self.documents or []:
            content_lower = doc.page_content.lower()
            score = 0
            
            for keyword in relevant_keywords:
                count = content_lower.count(keyword.lower())
                score += count
            
            if score > 0:
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "similarity_score": min(score / 5, 1.0),
                        "search_type": "keyword"
                    }
                )
                keyword_results.append((score, doc_copy))
        
        keyword_results.sort(key=lambda x: x[0], reverse=True)
        keyword_docs = [doc for _, doc in keyword_results[:top_k]]
        
        print(f"🎯 Keywords encontraram: {len(keyword_docs)} docs")
        
        # === BUSCA 2: NÚMERO DE PROCESSO ===
        process_results = []
        process_pattern = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
        process_match = re.search(process_pattern, query)
        
        if process_match:
            process_number = process_match.group()
            print(f"🔢 Processo específico: {process_number}")
            
            for doc in self.documents or []:
                if process_number in doc.page_content:
                    doc_copy = Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "similarity_score": 0.95,
                            "search_type": "process"
                        }
                    )
                    process_results.append(doc_copy)
            
            print(f"🔢 Processo encontrou: {len(process_results)} docs")
        
        # === BUSCA 3: SEMÂNTICA PERMISSIVA ===
        semantic_results = []
        try:
            semantic_results = self.vector_store.similarity_search(
                query, k=top_k, min_score=0.001  # MUITO permissivo
            )
            print(f"🧠 Semântica encontrou: {len(semantic_results)} docs")
        except Exception as e:
            print(f"⚠️ Busca semântica falhou: {e}")
        
        # === COMBINA RESULTADOS ===
        all_results = []
        seen = set()
        
        # Prioriza: processo > keywords > semântica
        for result_set in [process_results, keyword_docs, semantic_results]:
            for doc in result_set:
                doc_hash = hash(doc.page_content[:100])
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    all_results.append(doc)
        
        final_results = all_results[:top_k]
        print(f"✅ Total selecionado: {len(final_results)} documentos")
        
        return final_results
    
    def _create_optimized_context(self, docs: List[Document]) -> str:
        """Cria contexto SEM limitações"""
        if not docs:
            return ""
        
        context_parts = []
        total_length = 0
        max_length = self.config.max_context_length  # Agora 20000
        
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            
            # ⚡ REMOVIDO: Limitação de 400 chars por documento
            # Agora usa documento COMPLETO
            
            # Metadados importantes
            metadata_info = []
            for key in ['numero_processo', 'agravante', 'agravado', 'assuntos', 'valor_causa']:
                if key in doc.metadata and doc.metadata[key]:
                    metadata_info.append(f"{key}: {doc.metadata[key]}")
            
            # Monta documento completo
            doc_text = f"DOCUMENTO {i}"
            if metadata_info:
                doc_text += f" (Metadados: {', '.join(metadata_info)})"
            doc_text += f":\n{content}"  # ⚡ DOCUMENTO COMPLETO
            
            # Só limita se não couber no total
            if total_length + len(doc_text) > max_length:
                remaining = max_length - total_length - 500
                if remaining > 1000:
                    doc_text = f"DOCUMENTO {i} (PARCIAL):\n{content[:remaining]}..."
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)
    
    def _create_optimized_prompt(self, question: str, context: str) -> str:
        """Prompt otimizado para encontrar conteúdo específico"""
        return f"""Você é um assistente jurídico especializado. Analise TODOS os documentos fornecidos e responda de forma DETALHADA.

    INSTRUÇÕES CRÍTICAS:
    1. Use TODAS as informações relevantes dos documentos
    2. Para argumentos de defesa: extraia alegações, sustentações específicas
    3. Para motivações: identifique tipo de ação, fundamentos, razões
    4. Cite trechos específicos quando relevante
    5. NÃO se limite aos metadados - use o CONTEÚDO COMPLETO
    6. Se houver múltiplas informações, inclua todas

    DOCUMENTOS COMPLETOS:
    {context}

    PERGUNTA: {question}

    RESPOSTA DETALHADA E FUNDAMENTADA:"""
    
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
    """Inicializa sistema RAG com configurações corretas"""
    global optimized_rag_system
    
    config = UltraFastRAGConfig(
        model_name="gemma:2b",  # ⚡ SEU MODELO
        enable_cache=True,
        enable_parallel_search=True,
        enable_preprocessing=True,
        num_predict=1500,  # ⚡ Respostas completas
        max_context_length=20000,  # ⚡ Contexto expandido
        top_k=8,  # ⚡ Mais documentos
        min_similarity_score=0.001,  # ⚡ Muito permissivo
        max_chunks=1000  # ⚡ CRÍTICO: documentos suficientes
    )
    
    optimized_rag_system = UltraFastRAG(config)  # ⚡ CLASSE CORRETA
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
        
def apply_advanced_search_fix(rag_system):
    """
    Aplica correção avançada de busca para encontrar conteúdo real dos documentos
    """
    print("🔧 Aplicando correção avançada de busca...")
    
    try:
        # 1. RECONFIGURAÇÃO COMPLETA PARA ENCONTRAR CONTEÚDO
        if hasattr(rag_system, 'config'):
            rag_system.config.min_similarity_score = 0.001  # ⚡ MUITO mais permissivo
            rag_system.config.top_k = 8  # ⚡ Mais documentos
            
        # 2. SISTEMA DE BUSCA HÍBRIDA AVANÇADA
        def advanced_hybrid_search(query: str, top_k: int = 8) -> List:
            """Sistema de busca híbrida que garante encontrar conteúdo relevante"""
            if not rag_system.vector_store:
                return []
            
            print(f"🔍 Busca híbrida para: '{query}'")
            
            # === ESTRATÉGIA 1: BUSCA POR KEYWORDS ESPECÍFICAS ===
            keyword_results = []
            
            # Keywords específicas para argumentos jurídicos
            legal_keywords = {
                'argumento': ['argumento', 'argumenta', 'sustenta', 'defesa', 'alega'],
                'defesa': ['defesa', 'razões de defesa', 'contraditório', 'ampla defesa', 'cerceamento'],
                'motivacao': ['motivação', 'fundamento', 'razão', 'motivo', 'objetivo'],
                'pad': ['PAD', 'processo administrativo', 'disciplinar', 'demissão'],
                'lei': ['lei municipal', 'lei complementar', 'art.', 'artigo'],
                'processo': ['processo', 'agravo', 'recurso', 'ação'],
                'tutela': ['tutela', 'liminar', 'urgência', 'antecipação']
            }
            
            # Identifica tipo de busca e keywords relevantes
            query_lower = query.lower()
            relevant_keywords = []
            
            for categoria, keywords in legal_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        relevant_keywords.extend(keywords)
            
            # Se não encontrou keywords específicas, usa palavras da query
            if not relevant_keywords:
                relevant_keywords = [word for word in query_lower.split() if len(word) > 3]
            
            print(f"📝 Keywords relevantes: {relevant_keywords}")
            
            # Busca por keywords nos documentos
            for i, doc in enumerate(rag_system.documents or []):
                content_lower = doc.page_content.lower()
                score = 0
                matches = []
                
                for keyword in relevant_keywords:
                    count = content_lower.count(keyword.lower())
                    if count > 0:
                        score += count * (len(keyword) / 10)  # Score ponderado por tamanho
                        matches.append(f"{keyword}({count})")
                
                if score > 0:
                    # Cria documento com score de keyword
                    doc_copy = type(doc)(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "similarity_score": min(score / 10, 1.0),  # Normaliza score
                            "search_type": "keyword",
                            "matches": matches
                        }
                    )
                    keyword_results.append((score, doc_copy))
            
            # Ordena por score e pega os melhores
            keyword_results.sort(key=lambda x: x[0], reverse=True)
            keyword_docs = [doc for _, doc in keyword_results[:top_k]]
            
            print(f"🎯 Busca por keywords encontrou: {len(keyword_docs)} documentos")
            
            # === ESTRATÉGIA 2: BUSCA SEMÂNTICA MUITO PERMISSIVA ===
            semantic_results = []
            try:
                # Busca semântica com threshold mínimo
                semantic_results = rag_system.vector_store.similarity_search(
                    query, k=top_k, min_score=0.001  # Muito permissivo
                )
                print(f"🧠 Busca semântica encontrou: {len(semantic_results)} documentos")
            except Exception as e:
                print(f"⚠️ Busca semântica falhou: {e}")
            
            # === ESTRATÉGIA 3: BUSCA POR NÚMERO DE PROCESSO ===
            process_results = []
            process_pattern = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            process_match = re.search(process_pattern, query)
            
            if process_match:
                process_number = process_match.group()
                print(f"🔢 Buscando processo específico: {process_number}")
                
                for doc in rag_system.documents or []:
                    if process_number in doc.page_content or process_number in str(doc.metadata):
                        doc_copy = type(doc)(
                            page_content=doc.page_content,
                            metadata={
                                **doc.metadata,
                                "similarity_score": 0.9,
                                "search_type": "process_number"
                            }
                        )
                        process_results.append(doc_copy)
                
                print(f"🔢 Busca por processo encontrou: {len(process_results)} documentos")
            
            # === ESTRATÉGIA 4: BUSCA POR CONTEÚDO SUBSTANTIVO ===
            substantial_results = []
            
            # Identifica documentos com conteúdo substantivo (não apenas metadados)
            for doc in rag_system.documents or []:
                content = doc.page_content
                
                # Score baseado em indicadores de conteúdo substantivo
                substantial_score = 0
                
                # Frases que indicam conteúdo jurídico substantivo
                substantial_indicators = [
                    'sustenta', 'argumenta', 'alega', 'defende', 'contesta',
                    'fundamento', 'razão', 'motivo', 'ementa', 'decisão',
                    'voto', 'acórdão', 'sentença', 'despacho', 'parecer',
                    'lei municipal', 'código de processo', 'constituição',
                    'jurisprudência', 'precedente', 'súmula'
                ]
                
                for indicator in substantial_indicators:
                    if indicator in content.lower():
                        substantial_score += 1
                
                # Penaliza documentos que são só metadados
                if len(content) < 200 or content.count(':') > content.count('.'):
                    substantial_score *= 0.3
                
                if substantial_score > 2:  # Threshold para conteúdo substantivo
                    doc_copy = type(doc)(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "similarity_score": min(substantial_score / 10, 1.0),
                            "search_type": "substantial_content"
                        }
                    )
                    substantial_results.append(doc_copy)
            
            print(f"📚 Busca substantiva encontrou: {len(substantial_results)} documentos")
            
            # === COMBINAÇÃO E RANKING FINAL ===
            all_results = []
            seen_content = set()
            
            # Prioriza por tipo de busca: processo > keyword > substantial > semantic
            for result_set, priority in [
                (process_results, 100),
                (keyword_docs, 80),
                (substantial_results, 60),
                (semantic_results, 40)
            ]:
                for doc in result_set:
                    # Evita duplicatas por conteúdo
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        
                        # Ajusta score com prioridade
                        current_score = doc.metadata.get("similarity_score", 0)
                        final_score = (current_score * priority) / 100
                        
                        doc.metadata["final_score"] = final_score
                        doc.metadata["priority"] = priority
                        all_results.append(doc)
            
            # Ordena por score final e retorna top_k
            all_results.sort(key=lambda x: x.metadata.get("final_score", 0), reverse=True)
            final_results = all_results[:top_k]
            
            print(f"✅ Busca híbrida final: {len(final_results)} documentos selecionados")
            
            # Debug: mostra o que foi encontrado
            for i, doc in enumerate(final_results[:3], 1):
                search_type = doc.metadata.get("search_type", "unknown")
                score = doc.metadata.get("final_score", 0)
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   {i}. {search_type} (score: {score:.3f}): {preview}...")
            
            return final_results
        
        # 3. SUBSTITUI O MÉTODO DE BUSCA
        rag_system._search_documents_optimized = lambda query, top_k: advanced_hybrid_search(query, top_k)
        
        # 4. MÉTODO DE CONTEXTO INTELIGENTE QUE PRIORIZA CONTEÚDO SUBSTANTIVO
        def create_intelligent_context(docs):
            """Cria contexto priorizando conteúdo substantivo"""
            if not docs:
                return ""
            
            context_parts = []
            total_length = 0
            max_length = 20000  # Aumentado ainda mais
            
            # Ordena documentos por qualidade de conteúdo
            docs_sorted = sorted(docs, key=lambda d: len(d.page_content), reverse=True)
            
            for i, doc in enumerate(docs_sorted, 1):
                content = doc.page_content.strip()
                
                # Metadados importantes
                metadata_info = []
                search_type = doc.metadata.get("search_type", "embedding")
                score = doc.metadata.get("final_score", doc.metadata.get("similarity_score", 0))
                
                for key in ['numero_processo', 'agravante', 'agravado', 'assuntos']:
                    if key in doc.metadata and doc.metadata[key]:
                        metadata_info.append(f"{key}: {doc.metadata[key]}")
                
                # Informações de debug da busca
                matches = doc.metadata.get("matches", [])
                debug_info = f"Busca: {search_type}, Score: {score:.3f}"
                if matches:
                    debug_info += f", Matches: {matches}"
                
                # Monta documento completo
                doc_text = f"DOCUMENTO {i} ({debug_info})"
                if metadata_info:
                    doc_text += f"\nMetadados: {', '.join(metadata_info)}"
                doc_text += f"\nConteúdo:\n{content}"
                
                # Controle de tamanho total
                if total_length + len(doc_text) > max_length:
                    remaining = max_length - total_length - 500
                    if remaining > 1000:
                        doc_text = f"DOCUMENTO {i} (PARCIAL - {debug_info}):\n{content[:remaining]}..."
                        context_parts.append(doc_text)
                    break
                
                context_parts.append(doc_text)
                total_length += len(doc_text)
            
            return "\n\n" + "="*80 + "\n\n".join(context_parts)
        
        # Substitui método de contexto
        rag_system._create_optimized_context = lambda docs: create_intelligent_context(docs)
        
        # 5. PROMPT OTIMIZADO PARA EXTRAIR INFORMAÇÕES ESPECÍFICAS
        def create_extraction_prompt(question, context):
            return f"""Você é um assistente jurídico especializado. Analise os documentos fornecidos e responda de forma DETALHADA e COMPLETA.

INSTRUÇÕES IMPORTANTES:
1. Use TODAS as informações relevantes dos documentos
2. Para argumentos de defesa: extraia alegações, sustentações, argumentações específicas
3. Para motivações: identifique o tipo de ação, objeto, fundamentos jurídicos
4. Cite trechos específicos dos documentos quando relevante
5. Se houver informações contraditórias, mencione todas as versões
6. NÃO se limite apenas aos metadados - use o CONTEÚDO COMPLETO

DOCUMENTOS ANALISADOS:
{context}

PERGUNTA: {question}

RESPOSTA DETALHADA E FUNDAMENTADA:"""
        
        # Substitui método de prompt
        rag_system._create_optimized_prompt = create_extraction_prompt
        
        print("✅ Correção avançada de busca aplicada!")
        print("🎯 Melhorias implementadas:")
        print("  - Busca híbrida: keywords + semântica + processo + conteúdo")
        print("  - Threshold reduzido: 0.05 → 0.001 (muito mais permissivo)")
        print("  - Priorização de conteúdo substantivo vs metadados")
        print("  - Sistema de ranking inteligente")
        print("  - Contexto expandido: 15000 → 20000 chars")
        print("  - Prompt otimizado para extração específica")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na correção avançada: {e}")
        return False

def test_advanced_search():
    """Testa o sistema de busca avançada"""
    print("\n🧪 TESTANDO SISTEMA DE BUSCA AVANÇADA...")
    
    if not optimized_rag_system:
        print("❌ Sistema não inicializado")
        return
    
    # Testes específicos que estavam falhando
    test_cases = [
        "Qual foi o argumento da defesa no processo 1002436-58.2025.8.11.0000?",
        "Qual a motivação do processo 1002436-58.2025.8.11.0000?",
        "O que a agravante alega sobre cerceamento de defesa?",
        "Quais são os fundamentos da demissão da servidora?",
        "Qual foi a decisão do tribunal sobre o PAD?"
    ]
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n--- TESTE {i} ---")
        print(f"Pergunta: {question}")
        
        start_time = time.time()
        result = optimized_rag_system.query(question)
        elapsed = time.time() - start_time
        
        if "error" not in result:
            answer = result.get("answer", "")
            docs_found = result.get("documents_found", 0)
            
            print(f"✅ Resposta obtida em {elapsed:.2f}s")
            print(f"📊 Documentos encontrados: {docs_found}")
            print(f"📏 Tamanho da resposta: {len(answer)} caracteres")
            
            # Verifica se encontrou conteúdo substantivo
            if len(answer) > 200 and not answer.startswith("Não é possível"):
                print("🎉 SUCESSO: Conteúdo substantivo encontrado!")
            else:
                print("⚠️ Ainda limitado aos metadados")
            
            print(f"📝 Prévia: {answer[:200]}...")
            
        else:
            print(f"❌ Erro: {result['error']}")

# =================================================================================
# INTERFACE SIMPLIFICADA
# =================================================================================

def fix_search_and_test():
    """Aplica correção e testa imediatamente"""
    print("🚀 APLICANDO CORREÇÃO AVANÇADA DE BUSCA...")
    
    # Aplica a correção
    if apply_advanced_search_fix(optimized_rag_system):
        print("✅ Correção aplicada com sucesso!")
        
        # Testa imediatamente
        test_advanced_search()
        
        print("\n🎯 SISTEMA PRONTO PARA USO!")
        print("Use: result = optimized_rag_system.query('sua pergunta')")
    else:
        print("❌ Falha na aplicação da correção")

print("\n" + "="*80)
print("🔍 CORREÇÃO AVANÇADA DE BUSCA DISPONÍVEL!")
print("="*80)
print("Para aplicar e testar:")
print("fix_search_and_test()")
print()
print("Para aplicar apenas a correção:")
print("apply_advanced_search_fix(optimized_rag_system)")
print("="*80)

def test_specific_questions():
    """Testa perguntas específicas que estavam falhando"""
    if not optimized_rag_system:
        print("❌ Sistema não inicializado")
        return
    
    questions = [
        "Qual foi o argumento da defesa no processo 1002436-58.2025.8.11.0000?",
        "Qual a motivação do processo 1002436-58.2025.8.11.0000?",
        "O que a agravante alega sobre cerceamento de defesa?",
        "Resuma o que trata o processo 1002436-58.2025.8.11.0000"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"TESTE {i}: {question}")
        print('='*60)
        
        result = optimized_rag_system.query(question)
        
        if "error" not in result:
            answer = result.get("answer", "")
            docs = result.get("documents_found", 0)
            
            print(f"📊 Docs encontrados: {docs}")
            print(f"📏 Tamanho resposta: {len(answer)} chars")
            print(f"📝 Resposta:\n{answer}")
            
            if len(answer) > 300 and "não é possível" not in answer.lower():
                print("🎉 SUCESSO - Conteúdo substantivo encontrado!")
            else:
                print("⚠️ Resposta ainda limitada")
        else:
            print(f"❌ Erro: {result['error']}")

print("\n" + "="*80)
print("🔧 CORREÇÕES DIRETAS PARA SEU ARQUIVO")
print("="*80)
print("1. Substitua as seções numeradas acima no seu arquivo")
print("2. Execute: init_optimized_rag() para inicializar")
print("3. Carregue dados: optimized_rag_system.load_documents_from_directory()")
print("4. Teste: test_specific_questions()")
print("="*80)