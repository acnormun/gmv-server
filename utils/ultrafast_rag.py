
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
    use_ollama_embeddings: bool = False
    use_matryoshka_embeddings: bool = True
    matryoshka_preset: str = "fast"
    enable_conversational: bool = True
    max_context_length: int = 20000
    num_predict: int = 1500
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
            connected = False
            for attempt in range(3):
                try:
                    self.llm = OllamaLLM(
                        model=self.config.model_name,
                        temperature=self.config.temperature,
                        num_predict=self.config.num_predict,
                        base_url=self.config.ollama_base_url,
                        num_ctx=8192,
                        repeat_penalty=1.1,
                        top_k=40,
                        top_p=0.9,
                        stop=[],
                    )
                    self.llm.invoke("OK")
                    print(
                        f"LLM {self.config.model_name} conectado em {self.config.ollama_base_url}"
                    )
                    connected = True
                    break
                except Exception as e:
                    print(f"Tentativa {attempt+1} falhou ao conectar Ollama: {e}")
                    time.sleep(2)
            if not connected:
                print("Erro na conexão LLM: todas as tentativas falharam")
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
                use_ollama=self.config.use_ollama_embeddings,
                use_matryoshka=self.config.use_matryoshka_embeddings,
                matryoshka_preset=self.config.matryoshka_preset,
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
                use_ollama=self.config.use_ollama_embeddings,
                use_matryoshka=self.config.use_matryoshka_embeddings,
                matryoshka_preset=self.config.matryoshka_preset,
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
        print(f"Busca otimizada para: '{query}'")
        keyword_docs = self.vector_store._keyword_search(query, k=top_k)
        process_results = []
        process_pattern = r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
        process_match = re.search(process_pattern, query)
        if process_match:
            process_number = process_match.group()
            idxs = self.vector_store.process_index.get(process_number)
            if idxs:
                for idx in idxs:
                    doc = Document(
                        page_content=self.vector_store.documents[idx],
                        metadata={
                             **self.vector_store.metadata[idx],
                            "similarity_score": 0.95,
                            "search_type": "process"
                        }
                    )
                    process_results.append(doc)
        semantic_results = []
        try:
            semantic_results = self.vector_store.hybrid_search(
                query,
                k=top_k,
                semantic_weight=0.7,
                lexical_weight=0.3,
                min_score=self.config.min_similarity_score,
            )
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
        return all_results[:top_k]
    
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
        return f"""Você é um assistente jurídico especializado. Utilize apenas o texto a seguir como referência e responda de forma clara e completa, sem mencionar de onde as informações vieram.
    {context}

    PERGUNTA: {question}

    RESPOSTA:"""
    
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
                snippets = []
            for doc in relevant_docs:
                text = doc.page_content.strip()
                if len(text) > 200:
                    text = text[:200] + "..."
                snippets.append({
                    "snippet": text,
                    "metadata": doc.metadata
                })
            result = {
                "answer": answer.strip(),
                "documents_found": len(relevant_docs),
                "processing_time": time.time() - start_time,
                "from_cache": False,
                "cache_stats": self.cache_stats.copy(),
                "snippets": snippets
            }
            self._save_to_cache(cache_key, result)
            return result
        except Exception as e:
            error_result = {"error": str(e), "processing_time": time.time() - start_time}
            return error_result
            
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