
import os
import re
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from functools import lru_cache
import threading
from dotenv import load_dotenv

load_dotenv()

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.optimized_vector_store import OptimizedVectorStore

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
                    print("Usando OllamaLLM mock")
                def invoke(self, prompt): return "Resposta mock - Ollama não disponível"
                def __call__(self, prompt): return "Resposta mock - Ollama não disponível"

@dataclass
class UltraFastRAGConfig:
    model_name: str = "gemma3:4b"
    temperature: float = 0.0
    chunk_size: int = 2000
    chunk_overlap: int = 400
    top_k: int = 8
    max_chunks: int = 1000
    data_dir: str = "data"
    use_ollama_embeddings: bool = True
    enable_conversational: bool = True
    max_context_length: int = 20000
    num_predict: int = 1500
    enable_cache: bool = True
    cache_ttl: int = 3600 
    min_similarity_score: float = 0.001
    enable_parallel_search: bool = True
    enable_preprocessing: bool = True

class UltraFastRAG:
    def __init__(self, config: Optional[UltraFastRAGConfig] = None):
        self.config = config or UltraFastRAGConfig()
        self.llm = None
        self.vector_store = None
        self.documents = []
        self.is_initialized = False
        self.response_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.cache_lock = threading.Lock()
        self.data_path = os.getenv("PASTA_DESTINO", self.config.data_dir)
        self.cache_path = os.path.join(os.path.dirname(self.data_path), ".rag_cache")
        os.makedirs(self.cache_path, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        self._compile_preprocessing_patterns()
    
    def _compile_preprocessing_patterns(self):
        self.cleanup_patterns = [
            (re.compile(r'\s+'), ' '),
            (re.compile(r'\n+'), '\n'),
            (re.compile(r'[^\w\s\.\?\!\,\:\;\-\(\)]', re.UNICODE), ''),
        ]
        self.query_patterns = [
            (re.compile(r'\b(qual|quais|como|quando|onde|por que|porque)\b', re.IGNORECASE), ''),
            (re.compile(r'\b(me|nos|lhe|te)\s+(ajude|ajudar|diga|dizer|fale|falar)\b', re.IGNORECASE), ''),
            (re.compile(r'\b(por favor|obrigad[oa]|valeu)\b', re.IGNORECASE), ''),
        ]
    
    @lru_cache(maxsize=1000)
    def _preprocess_query(self, query: str) -> str:
        if not self.config.enable_preprocessing:
            return query
        processed = query.strip().lower()
        for pattern, replacement in self.query_patterns:
            processed = pattern.sub(replacement, processed)
        processed = re.sub(r'\s+', ' ', processed).strip()
        keywords = [word for word in processed.split() if len(word) > 2 and word not in ['para', 'com', 'por', 'sobre']]
        return ' '.join(keywords) if keywords else query
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        content = f"{query}_{top_k}_{self.config.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        if not self.config.enable_cache:
            return None
        with self.cache_lock:
            if cache_key in self.response_cache:
                cached_data, timestamp = self.response_cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl:
                    self.cache_stats["hits"] += 1
                    return cached_data
                else:
                    del self.response_cache[cache_key]
        self.cache_stats["misses"] += 1
        return None
    
    def _save_to_cache(self, cache_key: str, response: Dict[str, Any]) -> None:
        if not self.config.enable_cache:
            return
        with self.cache_lock:
            if len(self.response_cache) > 500:
                oldest_key = min(self.response_cache.keys(), key=lambda k: self.response_cache[k][1])
                del self.response_cache[oldest_key]
            self.response_cache[cache_key] = (response, time.time())
    
    def initialize(self):
        try:
            print("Inicializando UltraFast RAG Otimizado...")
            try:
                self.llm = OllamaLLM(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=self.config.num_predict,
                    num_ctx=8192,
                    repeat_penalty=1.1,
                    top_k=40,
                    top_p=0.9,
                    stop=[],
                )
                self.llm.invoke("OK")
                print(f"LLM {self.config.model_name} otimizado conectado")
            except Exception as e:
                print(f"Erro na conexão LLM: {e}")
                return False
            self.is_initialized = True
            self._load_cache()
            if self.config.enable_conversational:
                try:
                    self._integrate_conversational()
                except Exception as e:
                    print(f"Erro na integração conversacional: {e}")
            return True
        except Exception as e:
            print(f"Erro crítico na inicialização: {e}")
            self.is_initialized = False
            return False

    def _integrate_conversational(self):
        try:
            import sys
            if 'adaptive_rag' in sys.modules:
                module = sys.modules['adaptive_rag']
                if hasattr(module, 'enhance_rag_with_conversation'):
                    enhance_rag_with_conversation = module.enhance_rag_with_conversation
                    enhance_rag_with_conversation(self)
                    print("Sistema conversacional integrado!")
        except Exception as e:
            print(f"Erro ao integrar conversacional: {e}")
    
    def _load_cache(self):
        try:
            self.vector_store = OptimizedVectorStore(
                os.path.join(self.cache_path, "optimized"),
                use_ollama=self.config.use_ollama_embeddings
            )
            if self.vector_store.load():
                try:
                    self.documents = []
                    for doc_content, metadata in zip(self.vector_store.documents, self.vector_store.metadata):
                        self.documents.append(Document(page_content=doc_content, metadata=metadata))
                    print(f"Cache carregado: {len(self.documents)} documentos")
                except Exception as e:
                    print(f"Erro ao processar cache: {e}")
                    self.documents = []
            else:
                self.documents = []
        except Exception as e:
            print(f"Erro ao carregar cache: {e}")
            self.vector_store = None
    
    def load_documents_from_directory(self):
        if not os.path.exists(self.data_path):
            print(f"❌ Diretório não encontrado: {self.data_path}")
            return 0
        print(f"🔍 Carregando documentos de: {self.data_path}")
        documents = []
        arquivos_processados = 0
        arquivos_ignorados = 0
        pastas_ignoradas = {'.rag_cache', 'anonimizados', 'dat', 'mapas', '__pycache__', '.git'}
        try:
            for root, dirs, files in os.walk(self.data_path):
                dirs[:] = [d for d in dirs if d not in pastas_ignoradas and not d.startswith('.')]
                rel_path = os.path.relpath(root, self.data_path)
                if rel_path != ".":
                    print(f"📂 Explorando: {rel_path}")
                for file in files:
                    if file.lower().endswith(('.txt', '.md')):
                        try:
                            filepath = os.path.join(root, file)
                            file_size = os.path.getsize(filepath)
                            if file_size < 100:
                                print(f"   ⏭️ Pulando {file} (muito pequeno: {file_size} bytes)")
                                arquivos_ignorados += 1
                                continue
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            if len(content) < 100:
                                print(f"   ⏭️ Pulando {file} (conteúdo insuficiente)")
                                arquivos_ignorados += 1
                                continue
                            print(f"   📄 Processando: {file} ({len(content)} chars)")
                            metadata = self._parse_enhanced_metadata(content, filepath)
                            metadata['filename'] = file
                            metadata['source'] = filepath
                            metadata['relative_path'] = os.path.relpath(filepath, self.data_path)
                            clean_content = self._clean_content_fast(content)
                            if len(clean_content.strip()) < 50:
                                print(f"   ⏭️ Pulando {file} (sem conteúdo substantivo após limpeza)")
                                arquivos_ignorados += 1
                                continue
                            chunks = self.text_splitter.split_text(clean_content)
                            chunks_adicionados = 0
                            for i, chunk in enumerate(chunks):
                                chunk_clean = chunk.strip()
                                if len(chunk_clean) > 100:
                                    chunk_metadata = metadata.copy()
                                    chunk_metadata['chunk_index'] = i
                                    chunk_metadata['total_chunks'] = len(chunks)
                                    documents.append(Document(
                                        page_content=chunk_clean,
                                        metadata=chunk_metadata
                                    ))
                                    chunks_adicionados += 1
                            print(f"      ✅ {chunks_adicionados} chunks criados")
                            arquivos_processados += 1
                        except Exception as e:
                            print(f"   ❌ Erro ao processar {file}: {e}")
                            arquivos_ignorados += 1
                            continue
            print(f"\n📊 RELATÓRIO DE CARREGAMENTO:")
            print(f"   📄 Arquivos processados: {arquivos_processados}")
            print(f"   ⏭️ Arquivos ignorados: {arquivos_ignorados}")
            print(f"   📚 Total de chunks: {len(documents)}")
            if documents:
                print(f"✅ {len(documents)} chunks carregados com sucesso!")
                self.documents = documents
                self._create_vector_store()
                print(f"\n🔍 EXEMPLOS DE DOCUMENTOS CARREGADOS:")
                for i, doc in enumerate(documents[:3]):
                    meta = doc.metadata
                    print(f"   {i+1}. {meta.get('filename', 'N/A')} (chunk {meta.get('chunk_index', 0)})")
                    print(f"      Processo: {meta.get('numero_processo', 'N/A')}")
                    print(f"      Tamanho: {len(doc.page_content)} chars")
                    print(f"      Preview: {doc.page_content[:100]}...")
            else:
                print("⚠️ Nenhum documento foi carregado!")
            return len(documents)
        except Exception as e:
            print(f"❌ Erro crítico ao carregar documentos: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def _parse_enhanced_metadata(self, content: str, filepath: str) -> Dict[str, str]:
        metadata = {}
        if content.startswith('---'):
            try:
                end_pos = content.find('---', 3)
                if end_pos != -1:
                    yaml_content = content[3:end_pos]
                    for line in yaml_content.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            if value and value != 'N/A':
                                metadata[key] = value
            except Exception as e:
                print(f"      ⚠️ Erro ao processar front matter: {e}")
        if not metadata.get('numero_processo'):
            regex_metadata = self._parse_legal_metadata_fast(content)
            metadata.update(regex_metadata)
        if not metadata.get('numero_processo'):
            processo_match = re.search(r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})', filepath)
            if processo_match:
                metadata['numero_processo'] = processo_match.group(1)
        return metadata

    @lru_cache(maxsize=100)
    def _parse_legal_metadata_fast(self, content: str) -> Dict[str, str]:
        metadata = {}
        patterns = {
            'numero_processo': [
                r'numero_processo[:\s]+["\']?([^"\'\n]+)["\']?',
                r'Número[:\s]+(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})',
                r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})'
            ],
            'agravante': [
                r'agravante[:\s]+["\']?([^"\'\n]+)["\']?',
                r'AGRAVANTE[:\s]+([^:\n]+)',
                r'Agravante[:\s]+([^:\n]+)'
            ],
            'agravado': [
                r'agravado[:\s]+["\']?([^"\'\n]+)["\']?',
                r'AGRAVADO[:\s]+([^:\n]+)',
                r'Agravado[:\s]+([^:\n]+)'
            ],
            'valor_causa': [
                r'valor_causa[:\s]+["\']?([^"\'\n]+)["\']?',
                r'Valor da causa[:\s]+(R\$[^:\n]+)',
                r'VALOR DA CAUSA[:\s]+(R\$[^:\n]+)'
            ],
            'assuntos': [
                r'assuntos[:\s]+["\']?([^"\'\n]+)["\']?',
                r'Assuntos[:\s]+([^:\n]+)'
            ]
        }
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if value and len(value) > 1:
                        metadata[key] = value
                        break
        return metadata
    
    def _clean_content_fast(self, content: str) -> str:
        content = re.sub(r"### METADADOS DO PROCESSO.*?(?=\n###|\Z)", "", content, flags=re.DOTALL)
        content = re.sub(r"### PARTES ENVOLVIDAS.*?(?=\n###|\Z)", "", content, flags=re.DOTALL)
        for pattern, replacement in self.cleanup_patterns:
            content = pattern.sub(replacement, content)
        return content.strip()
    
    def _create_vector_store(self):
        if not self.documents:
            return
        try:
            print("Criando vector store otimizado...")
            self.vector_store = OptimizedVectorStore(
                os.path.join(self.cache_path, "optimized"),
                use_ollama=self.config.use_ollama_embeddings
            )
            max_docs = min(len(self.documents), self.config.max_chunks)
            self.vector_store.add_documents(self.documents[:max_docs], max_docs=max_docs)
            print("Vector store otimizado criado!")
        except Exception as e:
            print(f"Erro ao criar vector store: {e}")
            self.vector_store = None
    
    def _search_documents_optimized(self, query: str, top_k: int) -> List[Document]:
        if not self.vector_store:
            return []
        print(f"Busca híbrida para: '{query}'")
        keyword_results = []
        legal_keywords = [
            'argumento', 'argumenta', 'sustenta', 'alega', 'defesa', 
            'motivação', 'fundamento', 'razão', 'motivo',
            'pad', 'processo administrativo', 'disciplinar', 'demissão',
            'cerceamento', 'contraditório', 'ampla defesa',
            'tutela', 'liminar', 'urgência', 'recurso', 'agravo'
        ]
        query_lower = query.lower()
        relevant_keywords = []
        for keyword in legal_keywords:
            if keyword in query_lower:
                relevant_keywords.append(keyword)
        if not relevant_keywords:
            relevant_keywords = [word for word in query_lower.split() if len(word) > 3]
        print(f"Buscando por: {relevant_keywords}")
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
        print(f"Keywords encontraram: {len(keyword_docs)} docs")
        process_results = []
        process_pattern = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
        process_match = re.search(process_pattern, query)
        if process_match:
            process_number = process_match.group()
            print(f"Processo específico: {process_number}")
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
            print(f"Processo encontrou: {len(process_results)} docs")
        semantic_results = []
        try:
            semantic_results = self.vector_store.similarity_search(
                query, k=top_k, min_score=0.001
            )
            print(f"Semântica encontrou: {len(semantic_results)} docs")
        except Exception as e:
            print(f"Busca semântica falhou: {e}")
        all_results = []
        seen = set()
        for result_set in [process_results, keyword_docs, semantic_results]:
            for doc in result_set:
                doc_hash = hash(doc.page_content[:100])
                if doc_hash not in seen:
                    seen.add(doc_hash)
                    all_results.append(doc)
        final_results = all_results[:top_k]
        print(f"Total selecionado: {len(final_results)} documentos")
        return final_results
    
    def _create_optimized_context(self, docs: List[Document]) -> str:
        if not docs:
            return ""
        context_parts = []
        total_length = 0
        max_length = self.config.max_context_length
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            metadata_info = []
            for key in ['numero_processo', 'agravante', 'agravado', 'assuntos', 'valor_causa']:
                if key in doc.metadata and doc.metadata[key]:
                    metadata_info.append(f"{key}: {doc.metadata[key]}")
            doc_text = f"DOCUMENTO {i}"
            if metadata_info:
                doc_text += f" (Metadados: {', '.join(metadata_info)})"
            doc_text += f":\n{content}"
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
        if not self.is_initialized or not self.vector_store:
            return {"error": "Sistema não inicializado"}
        k = top_k or self.config.top_k
        cache_key = self._get_cache_key(question, k)
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            cached_result["from_cache"] = True
            return cached_result
        try:
            start_time = time.time()
            relevant_docs = self._search_documents_optimized(question, k)
            if not relevant_docs:
                result = {
                    "error": "Nenhum documento relevante encontrado",
                    "suggestion": "Tente reformular a pergunta",
                    "processing_time": time.time() - start_time
                }
                self._save_to_cache(cache_key, result)
                return result
            context = self._create_optimized_context(relevant_docs)
            prompt = self._create_optimized_prompt(question, context)
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
            self._save_to_cache(cache_key, result)
            return result
        except Exception as e:
            error_result = {"error": str(e), "processing_time": time.time() - start_time}
            return error_result
    
    def get_performance_stats(self) -> Dict[str, Any]:
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
        with self.cache_lock:
            self.response_cache.clear()
            self.cache_stats = {"hits": 0, "misses": 0}
        print("Cache limpo")

optimized_rag_system = None

def init_optimized_rag():
    global optimized_rag_system
    config = UltraFastRAGConfig(
        model_name="gemma3:4b",
        enable_cache=True,
        enable_parallel_search=True,
        enable_preprocessing=True,
        num_predict=1500,
        max_context_length=20000,
        top_k=8,
        min_similarity_score=0.001,
        max_chunks=1000
    )
    optimized_rag_system = UltraFastRAG(config)
    return optimized_rag_system.initialize()

def load_optimized_data():
    if optimized_rag_system:
        return optimized_rag_system.load_documents_from_directory()
    return 0

def query_optimized_rag(question: str, top_k: int = 3):
    if optimized_rag_system:
        return optimized_rag_system.query(question, top_k)
    return {"error": "Sistema não inicializado"}

def get_optimized_performance_stats():
    if optimized_rag_system:
        return optimized_rag_system.get_performance_stats()
    return {"error": "Sistema não inicializado"}

def clear_optimized_cache():
    if optimized_rag_system:
        optimized_rag_system.clear_cache()

def performance_comparison_test():
    print("TESTE DE PERFORMANCE - RAG OTIMIZADO")
    print("=" * 60)
    if not optimized_rag_system or not optimized_rag_system.is_initialized:
        print("Sistema otimizado não inicializado")
        return
    test_queries = [
        "processo 1005888",
        "terapia ABA",
        "valor da causa",
        "agravante agravado",
        "SUS tratamento"
    ]
    print("Testando velocidade das consultas...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- TESTE {i} ---")
        print(f"Query: {query}")
        start_time = time.time()
        result = optimized_rag_system.query(query)
        end_time = time.time()
        if "error" not in result:
            print(f"Resposta obtida")
            print(f"Tempo: {end_time - start_time:.2f}s")
            print(f"Docs encontrados: {result.get('documents_found', 0)}")
            print(f"Do cache: {result.get('from_cache', False)}")
        else:
            print(f"Erro: {result['error']}")
    print(f"\nESTATÍSTICAS FINAIS:")
    stats = optimized_rag_system.get_performance_stats()
    print(f"Cache hits: {stats['cache_stats']['hits']}")
    print(f"Cache misses: {stats['cache_stats']['misses']}")
    if stats['cache_stats']['hits'] + stats['cache_stats']['misses'] > 0:
        hit_rate = stats['cache_stats']['hits'] / (stats['cache_stats']['hits'] + stats['cache_stats']['misses'])
        print(f"Taxa de acerto do cache: {hit_rate:.2%}")

if __name__ == "__main__":
    print("SISTEMA RAG ULTRA OTIMIZADO")
    print("=" * 50)
    if init_optimized_rag():
        print("Sistema otimizado inicializado")
        docs_loaded = load_optimized_data()
        if docs_loaded > 0:
            print(f"{docs_loaded} documentos carregados")
            performance_comparison_test()
        else:
            print("Nenhum documento carregado")
    else:
        print("Falha na inicialização do sistema otimizado")
        
def apply_advanced_search_fix(rag_system):
    print("Aplicando correção avançada de busca...")
    try:
        if hasattr(rag_system, 'config'):
            rag_system.config.min_similarity_score = 0.001
            rag_system.config.top_k = 8
        def advanced_hybrid_search(query: str, top_k: int = 8) -> List:
            if not rag_system.vector_store:
                return []
            print(f"Busca híbrida para: '{query}'")
            keyword_results = []
            legal_keywords = {
                'argumento': ['argumento', 'argumenta', 'sustenta', 'defesa', 'alega'],
                'defesa': ['defesa', 'razões de defesa', 'contraditório', 'ampla defesa', 'cerceamento'],
                'motivacao': ['motivação', 'fundamento', 'razão', 'motivo', 'objetivo'],
                'pad': ['PAD', 'processo administrativo', 'disciplinar', 'demissão'],
                'lei': ['lei municipal', 'lei complementar', 'art.', 'artigo'],
                'processo': ['processo', 'agravo', 'recurso', 'ação'],
                'tutela': ['tutela', 'liminar', 'urgência', 'antecipação']
            }
            query_lower = query.lower()
            relevant_keywords = []
            for keywords in legal_keywords.values():
                for keyword in keywords:
                    if keyword in query_lower:
                        relevant_keywords.extend(keywords)
            if not relevant_keywords:
                relevant_keywords = [word for word in query_lower.split() if len(word) > 3]
            print(f"Keywords relevantes: {relevant_keywords}")
            for doc in rag_system.documents or []:
                content_lower = doc.page_content.lower()
                score = 0
                matches = []
                for keyword in relevant_keywords:
                    count = content_lower.count(keyword.lower())
                    if count > 0:
                        score += count * (len(keyword) / 10)
                        matches.append(f"{keyword}({count})")
                if score > 0:
                    doc_copy = type(doc)(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "similarity_score": min(score / 10, 1.0),
                            "search_type": "keyword",
                            "matches": matches
                        }
                    )
                    keyword_results.append((score, doc_copy))
            keyword_results.sort(key=lambda x: x[0], reverse=True)
            keyword_docs = [doc for _, doc in keyword_results[:top_k]]
            print(f"Busca por keywords encontrou: {len(keyword_docs)} documentos")
            semantic_results = []
            try:
                semantic_results = rag_system.vector_store.similarity_search(
                    query, k=top_k, min_score=0.001
                )
                print(f"Busca semântica encontrou: {len(semantic_results)} documentos")
            except Exception as e:
                print(f"Busca semântica falhou: {e}")
            process_results = []
            process_pattern = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            process_match = re.search(process_pattern, query)
            if process_match:
                process_number = process_match.group()
                print(f"Buscando processo específico: {process_number}")
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
                print(f"Busca por processo encontrou: {len(process_results)} documentos")
            substantial_results = []
            for doc in rag_system.documents or []:
                content = doc.page_content
                substantial_score = 0
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
                if len(content) < 200 or content.count(':') > content.count('.'):
                    substantial_score *= 0.3
                if substantial_score > 2:
                    doc_copy = type(doc)(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "similarity_score": min(substantial_score / 10, 1.0),
                            "search_type": "substantial_content"
                        }
                    )
                    substantial_results.append(doc_copy)
            print(f"Busca substantiva encontrou: {len(substantial_results)} documentos")
            all_results = []
            seen_content = set()
            for result_set, priority in [
                (process_results, 100),
                (keyword_docs, 80),
                (substantial_results, 60),
                (semantic_results, 40)
            ]:
                for doc in result_set:
                    content_hash = hash(doc.page_content[:200])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        current_score = doc.metadata.get("similarity_score", 0)
                        final_score = (current_score * priority) / 100
                        doc.metadata["final_score"] = final_score
                        doc.metadata["priority"] = priority
                        all_results.append(doc)
            all_results.sort(key=lambda x: x.metadata.get("final_score", 0), reverse=True)
            final_results = all_results[:top_k]
            print(f"Busca híbrida final: {len(final_results)} documentos selecionados")
            for i, doc in enumerate(final_results[:3], 1):
                search_type = doc.metadata.get("search_type", "unknown")
                score = doc.metadata.get("final_score", 0)
                preview = doc.page_content[:100].replace('\n', ' ')
                print(f"   {i}. {search_type} (score: {score:.3f}): {preview}...")
            return final_results
        rag_system._search_documents_optimized = lambda query, top_k: advanced_hybrid_search(query, top_k)
        def create_intelligent_context(docs):
            if not docs:
                return ""
            context_parts = []
            total_length = 0
            max_length = 20000
            docs_sorted = sorted(docs, key=lambda d: len(d.page_content), reverse=True)
            for i, doc in enumerate(docs_sorted, 1):
                content = doc.page_content.strip()
                metadata_info = []
                search_type = doc.metadata.get("search_type", "embedding")
                score = doc.metadata.get("final_score", doc.metadata.get("similarity_score", 0))
                for key in ['numero_processo', 'agravante', 'agravado', 'assuntos']:
                    if key in doc.metadata and doc.metadata[key]:
                        metadata_info.append(f"{key}: {doc.metadata[key]}")
                matches = doc.metadata.get("matches", [])
                debug_info = f"Busca: {search_type}, Score: {score:.3f}"
                if matches:
                    debug_info += f", Matches: {matches}"
                doc_text = f"DOCUMENTO {i} ({debug_info})"
                if metadata_info:
                    doc_text += f"\nMetadados: {', '.join(metadata_info)}"
                doc_text += f"\nConteúdo:\n{content}"
                if total_length + len(doc_text) > max_length:
                    remaining = max_length - total_length - 500
                    if remaining > 1000:
                        doc_text = f"DOCUMENTO {i} (PARCIAL - {debug_info}):\n{content[:remaining]}..."
                        context_parts.append(doc_text)
                    break
                context_parts.append(doc_text)
                total_length += len(doc_text)
            return "\n\n" + "="*80 + "\n\n".join(context_parts)
        rag_system._create_optimized_context = lambda docs: create_intelligent_context(docs)
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
        rag_system._create_optimized_prompt = create_extraction_prompt
        return True
    except Exception as e:
        print(f"Erro na correção avançada: {e}")
        return False

def test_advanced_search():
    print("\nTESTANDO SISTEMA DE BUSCA AVANÇADA...")
    if not optimized_rag_system:
        print("Sistema não inicializado")
        return
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
            print(f"Resposta obtida em {elapsed:.2f}s")
            print(f"Documentos encontrados: {docs_found}")
            print(f"Tamanho da resposta: {len(answer)} caracteres")
            if len(answer) > 200 and not answer.startswith("Não é possível"):
                print("SUCESSO: Conteúdo substantivo encontrado!")
            else:
                print("Ainda limitado aos metadados")
            print(f"Prévia: {answer[:200]}...")
        else:
            print(f"Erro: {result['error']}")

def fix_search_and_test():
    print("APLICANDO CORREÇÃO AVANÇADA DE BUSCA...")
    if apply_advanced_search_fix(optimized_rag_system):
        print("Correção aplicada com sucesso!")
        test_advanced_search()
        print("\nSISTEMA PRONTO PARA USO!")
        print("Use: result = optimized_rag_system.query('sua pergunta')")
    else:
        print("Falha na aplicação da correção")

def test_specific_questions():
    if not optimized_rag_system:
        print("Sistema não inicializado")
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
            print(f"Docs encontrados: {docs}")
            print(f"Tamanho resposta: {len(answer)} chars")
            print(f"Resposta:\n{answer}")
            if len(answer) > 300 and "não é possível" not in answer.lower():
                print("SUCESSO - Conteúdo substantivo encontrado!")
            else:
                print("Resposta ainda limitada")
        else:
            print(f"Erro: {result['error']}")