# utils/ultrafast_rag.py

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    model_name: str = "gemma:2b"
    temperature: float = 0.1
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4
    max_chunks: int = 500
    data_dir: str = "data"
    use_ollama_embeddings: bool = True
    enable_conversational: bool = True


class UltraFastRAG:
    def __init__(self, config: Optional[UltraFastRAGConfig] = None):
        self.config = config or UltraFastRAGConfig()
        self.llm = None
        self.vector_store = None
        self.documents = []
        self.is_initialized = False
        self.conversational_handler = None
        
        # Caminhos configuráveis
        self.data_path = os.getenv("DADOS_ANONIMOS", 
                                  os.getenv("PASTA_DESTINO", 
                                           self.config.data_dir))
        self.cache_path = os.path.join(os.path.dirname(self.data_path), ".rag_cache")
        os.makedirs(self.cache_path, exist_ok=True)
        
        # Splitter para chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
    
    def initialize(self):
        """Inicializa o sistema RAG"""
        try:
            print("🔄 Inicializando UltraFast RAG...")
            
            # Inicializa LLM com timeout
            try:
                print("🔌 Conectando com Ollama LLM...")
                self.llm = OllamaLLM(
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    num_predict=400
                )
                
                # Testa conexão com timeout implícito
                print("🧪 Testando conexão...")
                test_response = self.llm.invoke("Teste")
                print(f"✅ LLM {self.config.model_name} conectado")
                
            except Exception as e:
                print(f"❌ Erro na conexão LLM: {e}")
                print("⚠️ Continuando sem LLM - funcionalidade limitada")
                # Cria LLM mock para não quebrar o sistema
                class MockLLM:
                    def invoke(self, prompt):
                        return "Sistema RAG não disponível - LLM offline"
                    def __call__(self, prompt):
                        return "Sistema RAG não disponível - LLM offline"
                
                self.llm = MockLLM()
            
            self.is_initialized = True
            
            # Carrega cache de forma segura
            try:
                self._load_cache()
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache: {e}")
            
            # Integra sistema conversacional se habilitado
            if self.config.enable_conversational:
                try:
                    self._integrate_conversational()
                except Exception as e:
                    print(f"⚠️ Erro na integração conversacional: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro crítico na inicialização: {e}")
            print("💡 Verifique se o Ollama está rodando e os modelos instalados")
            self.is_initialized = False
            return False
    
    def _integrate_conversational(self):
        """Integra o sistema conversacional"""
        try:
            # Evita imports circulares usando import local
            import sys
            if 'adaptive_rag' in sys.modules:
                module = sys.modules['adaptive_rag']
                if hasattr(module, 'enhance_rag_with_conversation'):
                    enhance_rag_with_conversation = module.enhance_rag_with_conversation
                    enhance_rag_with_conversation(self)
                    print("✅ Sistema conversacional integrado!")
                else:
                    print("⚠️ Função enhance_rag_with_conversation não encontrada")
            else:
                print("⚠️ Módulo adaptive_rag não carregado")
        except Exception as e:
            print(f"⚠️ Erro ao integrar sistema conversacional: {e}")
            print("🔄 Continuando com sistema RAG básico")
    
    def _load_cache(self):
        """Carrega cache existente"""
        try:
            print("📦 Verificando cache existente...")
            
            self.vector_store = OptimizedVectorStore(
                os.path.join(self.cache_path, "optimized"),
                use_ollama=self.config.use_ollama_embeddings
            )
            
            if self.vector_store.load():
                try:
                    # Reconstrói lista de documentos do vector store
                    self.documents = []
                    for i, (doc_content, metadata) in enumerate(zip(self.vector_store.documents, self.vector_store.metadata)):
                        try:
                            self.documents.append(Document(page_content=doc_content, metadata=metadata))
                        except Exception as e:
                            print(f"⚠️ Erro ao carregar documento {i} do cache: {e}")
                            continue
                    
                    print(f"📦 Cache carregado: {len(self.documents)} documentos")
                except Exception as e:
                    print(f"⚠️ Erro ao processar documentos do cache: {e}")
                    self.documents = []
            else:
                print("📦 Nenhum cache encontrado - inicializando vazio")
                self.documents = []
                
        except Exception as e:
            print(f"⚠️ Erro ao carregar cache: {e}")
            # Cria vector store vazio em caso de erro
            try:
                self.vector_store = OptimizedVectorStore(
                    os.path.join(self.cache_path, "optimized"),
                    use_ollama=self.config.use_ollama_embeddings
                )
                self.documents = []
            except Exception as e2:
                print(f"❌ Erro crítico ao criar vector store: {e2}")
                # Mock vector store como último recurso
                class MockVectorStore:
                    def similarity_search(self, query, k=4, min_score=0.1):
                        return []
    
    def test_search_detailed(self, question: str):
        """Teste detalhado da busca para debug"""
        try:
            print(f"\n🔬 TESTE DETALHADO PARA: '{question}'")
            print("=" * 70)
            
            # Informações básicas
            print(f"📊 Sistema inicializado: {self.is_initialized}")
            print(f"📊 Total documentos: {len(self.documents) if self.documents else 0}")
            print(f"📊 Vector store: {self.vector_store is not None}")
            
            if not self.is_initialized or not self.vector_store:
                print("❌ Sistema não está pronto para busca")
                return
            
            # Teste de similaridade detalhado
            if hasattr(self.vector_store, 'test_similarity_calculation'):
                print(f"\n🧮 CALCULANDO SIMILARIDADES...")
                similarities = self.vector_store.test_similarity_calculation(question, max_docs=20)
            
            # Teste com diferentes thresholds
            print(f"\n🎯 TESTE COM DIFERENTES THRESHOLDS:")
            thresholds = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
            
            for threshold in thresholds:
                try:
                    results = self.vector_store.similarity_search(question, k=3, min_score=threshold)
                    print(f"   Threshold {threshold:4.2f}: {len(results)} resultados")
                    
                    if results and threshold <= 0.05:  # Mostra detalhes para thresholds baixos
                        for i, doc in enumerate(results):
                            score = doc.metadata.get("similarity_score", 0)
                            preview = doc.page_content[:50].replace('\n', ' ')
                            print(f"      {i+1}. Score {score:.3f}: {preview}...")
                            
                except Exception as e:
                    print(f"   Threshold {threshold:4.2f}: ERRO - {e}")
            
            # Busca por keywords
            print(f"\n🔍 TESTE DE BUSCA POR KEYWORDS:")
            if hasattr(self.vector_store, '_keyword_search'):
                keyword_results = self.vector_store._keyword_search(question, k=5)
                print(f"   Keywords encontraram: {len(keyword_results)} resultados")
                
                for i, doc in enumerate(keyword_results):
                    score = doc.metadata.get("similarity_score", 0)
                    search_type = doc.metadata.get("search_type", "unknown")
                    preview = doc.page_content[:50].replace('\n', ' ')
                    print(f"      {i+1}. {search_type} Score {score:.3f}: {preview}...")
            
            # Busca de fallback
            print(f"\n🆘 TESTE DE BUSCA FALLBACK:")
            fallback_results = self._simple_keyword_search(question, k=5)
            print(f"   Fallback encontrou: {len(fallback_results)} resultados")
            
            return {
                "question": question,
                "system_ready": self.is_initialized and self.vector_store is not None,
                "total_documents": len(self.documents) if self.documents else 0,
                "threshold_results": {str(t): "tested" for t in thresholds},
                "keyword_results": len(keyword_results) if 'keyword_results' in locals() else 0,
                "fallback_results": len(fallback_results)
            }
            
        except Exception as e:
            print(f"❌ Erro no teste detalhado: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def debug_search(self, question: str) -> Dict[str, Any]:
        """Método de debug para diagnosticar problemas de busca"""
        try:
            debug_info = {
                "question": question,
                "system_initialized": self.is_initialized,
                "total_documents": len(self.documents) if self.documents else 0,
                "vector_store_available": self.vector_store is not None,
                "config": {
                    "model_name": self.config.model_name,
                    "use_ollama_embeddings": self.config.use_ollama_embeddings,
                    "top_k": self.config.top_k
                }
            }
            
            if self.documents:
                # Amostra de documentos
                sample_docs = []
                for i, doc in enumerate(self.documents[:3]):
                    sample_docs.append({
                        "index": i,
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    })
                debug_info["sample_documents"] = sample_docs
            
            if self.vector_store and hasattr(self.vector_store, 'embeddings'):
                debug_info["embeddings_count"] = len(self.vector_store.embeddings)
                debug_info["embeddings_type"] = "ollama" if self.vector_store.use_ollama else "tfidf"
                
                # Testa busca com diferentes thresholds
                test_results = {}
                for threshold in [0.0, 0.01, 0.05, 0.1, 0.15]:
                    try:
                        results = self.vector_store.similarity_search(question, k=3, min_score=threshold)
                        test_results[f"threshold_{threshold}"] = len(results)
                    except Exception as e:
                        test_results[f"threshold_{threshold}"] = f"Error: {str(e)}"
                
                debug_info["threshold_tests"] = test_results
            
            return debug_info
            
        except Exception as e:
            return {"error": f"Debug failed: {str(e)}"}
    
    def list_available_processes(self, limit: int = 20) -> List[str]:
        """Lista números de processos disponíveis na base"""
        try:
            processes = set()
            pattern = r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            
            for doc in (self.documents or [])[:500]:  # Limita para performance
                # Busca no conteúdo
                found = re.findall(pattern, doc.page_content)
                processes.update(found)
                
                # Busca nos metadados
                if hasattr(doc, 'metadata'):
                    for key, value in doc.metadata.items():
                        if isinstance(value, str) and 'processo' in key.lower():
                            found = re.findall(pattern, value)
                            processes.update(found)
            
            return sorted(list(processes))[:limit]
            
        except Exception as e:
            print(f"⚠️ Erro ao listar processos: {e}")
            return []
                # Removed unreachable code
    def _parse_legal_metadata(self, file_content: str) -> Dict[str, str]:
        """Extrai metadados de documentos legais"""
        metadata = {}
        
        # Extrai metadados do processo
        metadata_match = re.search(r"### METADADOS DO PROCESSO\n(.*?)(?=\n###|\Z)", file_content, re.DOTALL)
        if metadata_match:
            block = metadata_match.group(1)
            # Parse simples key: value
            for line in block.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().strip('"')
                    value = value.strip().strip('"')
                    if key and value:
                        metadata[key] = value
        
        # Extrai partes envolvidas
        partes_match = re.search(r"### PARTES ENVOLVIDAS\n(.*?)(?=\n###|\Z)", file_content, re.DOTALL)
        if partes_match:
            block = partes_match.group(1)
            for line in block.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().strip('"')
                    value = value.strip().strip('"')
                    if key and value:
                        metadata[key] = value
        
        return metadata
    
    def load_documents_from_directory(self):
        """Carrega documentos do diretório"""
        if not os.path.exists(self.data_path):
            print(f"⚠️ Diretório não encontrado: {self.data_path}")
            return 0
        
        print(f"📁 Carregando documentos de: {self.data_path}")
        
        documents = []
        try:
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.lower().endswith(('.txt', '.md')):
                        try:
                            filepath = os.path.join(root, file)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            if len(content) < 100:  # Ignora arquivos muito pequenos
                                continue
                            
                            # Extrai metadados se for documento legal
                            metadata = self._parse_legal_metadata(content)
                            metadata['filename'] = file
                            metadata['source'] = filepath
                            
                            # Remove blocos de metadados do conteúdo principal
                            clean_content = re.sub(r"### METADADOS DO PROCESSO.*?(?=\n###|\Z)", "", content, flags=re.DOTALL)
                            clean_content = re.sub(r"### PARTES ENVOLVIDAS.*?(?=\n###|\Z)", "", clean_content, flags=re.DOTALL)
                            clean_content = clean_content.strip()
                            
                            # Cria chunks do documento de forma mais segura
                            try:
                                chunks = self.text_splitter.split_text(clean_content)
                                for chunk in chunks:
                                    if len(chunk.strip()) > 100:  # Chunks muito pequenos
                                        documents.append(Document(
                                            page_content=chunk.strip(),
                                            metadata=metadata.copy()
                                        ))
                            except Exception as e:
                                print(f"⚠️ Erro ao criar chunks para {file}: {e}")
                                # Adiciona documento inteiro se falhar o chunking
                                if len(clean_content) > 100:
                                    documents.append(Document(
                                        page_content=clean_content,
                                        metadata=metadata.copy()
                                    ))
                            
                        except Exception as e:
                            print(f"⚠️ Erro ao processar {file}: {e}")
                            continue
            
            if documents:
                print(f"📄 {len(documents)} chunks carregados")
                self.documents = documents
                self._create_vector_store()
            else:
                print("⚠️ Nenhum documento válido encontrado")
            
            return len(documents)
            
        except Exception as e:
            print(f"❌ Erro crítico ao carregar documentos: {e}")
            return 0
    
    def _create_vector_store(self):
        """Cria vector store com os documentos"""
        if not self.documents:
            print("⚠️ Nenhum documento disponível para criar vector store")
            return
        
        try:
            print("🗃️ Criando vector store...")
            
            self.vector_store = OptimizedVectorStore(
                os.path.join(self.cache_path, "optimized"),
                use_ollama=self.config.use_ollama_embeddings
            )
            
            # Adiciona documentos de forma segura
            try:
                self.vector_store.add_documents(self.documents, max_docs=self.config.max_chunks)
                print("✅ Vector store criado!")
            except Exception as e:
                print(f"❌ Erro ao adicionar documentos ao vector store: {e}")
                # Cria vector store vazio mesmo com erro
                print("🔄 Continuando com vector store vazio...")
                
        except Exception as e:
            print(f"❌ Erro ao criar vector store: {e}")
            # Cria um mock vector store para não quebrar o sistema
            class MockVectorStore:
                def similarity_search(self, query, k=4, min_score=0.1):
                    print("⚠️ Usando mock vector store - retornando lista vazia")
                    return []
                
                def load(self):
                    return False
            
            self.vector_store = MockVectorStore()
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Realiza consulta no sistema RAG (método básico, pode ser sobrescrito pelo conversacional)"""
        if not self.is_initialized or not self.vector_store:
            return {"error": "Sistema não inicializado"}
        
        try:
            k = top_k or self.config.top_k
            print(f"🔍 Processando: {question[:50]}...")
            
            # Busca documentos relevantes com threshold mais baixo
            relevant_docs = self.vector_store.similarity_search(question, k=k, min_score=0.05)  # Reduzido de 0.15 para 0.05
            
            if not relevant_docs:
                # Tenta busca mais permissiva se não encontrou nada
                print("🔄 Tentando busca mais permissiva...")
                relevant_docs = self.vector_store.similarity_search(question, k=k, min_score=0.01)
            
            if not relevant_docs:
                # Como último recurso, pega os documentos com maior score, independente do threshold
                print("🔄 Buscando documentos com qualquer score...")
                try:
                    # Força busca sem threshold mínimo
                    all_docs = self.vector_store.similarity_search(question, k=k, min_score=0.0)
                    if all_docs:
                        relevant_docs = all_docs
                    else:
                        # Busca por palavras-chave simples se embedding falhou
                        print("🔄 Tentando busca por palavras-chave...")
                        simple_search = self._simple_keyword_search(question, k)
                        if simple_search:
                            relevant_docs = simple_search
                except Exception as e:
                    print(f"⚠️ Erro na busca de fallback: {e}")
            
            if not relevant_docs:
                # Fornece informação mais útil sobre o que está disponível
                total_docs = len(self.documents) if self.documents else 0
                return {
                    "error": "Nenhum documento relevante encontrado",
                    "suggestion": f"Tente reformular a pergunta. Há {total_docs} documentos disponíveis na base.",
                    "query_analyzed": question,
                    "available_documents": total_docs
                }
            
            # Prepara contexto
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.page_content.strip()
                score = doc.metadata.get("similarity_score", 0)
                search_type = doc.metadata.get("search_type", "embedding")
                
                # Adiciona metadados relevantes se existirem
                metadata_info = []
                for key in ['numero_processo', 'agravante', 'agravado', 'assuntos', 'filename']:
                    if key in doc.metadata and doc.metadata[key]:
                        metadata_info.append(f"{key}: {doc.metadata[key]}")
                
                doc_text = f"DOCUMENTO {i} (relevância: {score:.3f}, busca: {search_type}):\n"
                if metadata_info:
                    doc_text += f"Metadados: {', '.join(metadata_info)}\n"
                doc_text += f"Conteúdo: {content}"
                
                context_parts.append(doc_text)
                
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
                elif 'filename' in doc.metadata:
                    sources.add(doc.metadata['filename'])
            
            context = "\n\n" + "="*50 + "\n\n".join(context_parts)
            
            # Template otimizado para documentos legais
            prompt_template = f"""Você é um assistente especializado em documentos legais. Responda com base EXCLUSIVAMENTE nos documentos fornecidos.

INSTRUÇÕES:
1. Use APENAS as informações dos documentos abaixo
2. Cite metadados relevantes (número do processo, partes, etc.) quando disponíveis
3. Se a informação não estiver nos documentos, responda "Informação não encontrada nos documentos disponíveis"
4. Seja preciso e objetivo
5. Mantenha linguagem jurídica adequada
6. Se encontrar informações parciais, mencione e explique o que foi encontrado

DOCUMENTOS:
{context}

PERGUNTA: {question}

RESPOSTA:"""

            print("🤖 Consultando LLM...")
            
            # Chama LLM
            if hasattr(self.llm, 'invoke'):
                answer = self.llm.invoke(prompt_template)
            else:
                answer = self.llm(prompt_template)
            
            return {
                "answer": answer.strip(),
                "context_used": context,
                "sources": list(sources),
                "documents_found": len(relevant_docs),
                "search_method": "Híbrido TF-IDF + Ollama" if self.config.use_ollama_embeddings else "TF-IDF",
                "source_documents": [{
                    "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                    "metadata": {k: v for k, v in doc.metadata.items() if k in ['filename', 'numero_processo', 'agravante', 'agravado', 'search_type']},
                    "similarity_score": doc.metadata.get("similarity_score", 0)
                } for doc in relevant_docs]
            }
            
        except Exception as e:
            print(f"❌ Erro na consulta: {e}")
            return {"error": str(e)}
    
    def _simple_keyword_search(self, question: str, k: int = 4):
        """Busca simples por palavras-chave quando embedding falha"""
        try:
            if not self.documents:
                return []
            
            question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
            if not question_words:
                return []
            
            matches = []
            for i, doc in enumerate(self.documents):
                try:
                    doc_text = doc.page_content.lower()
                    doc_words = set(re.findall(r'\b\w{3,}\b', doc_text))
                    
                    if doc_words:
                        overlap = len(question_words.intersection(doc_words))
                        score = overlap / len(question_words)
                        
                        if score > 0:
                            matches.append((i, score))
                except Exception:
                    continue
            
            # Ordena e pega os melhores
            matches.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, score in matches[:k]:
                try:
                    doc = Document(
                        page_content=self.documents[i].page_content,
                        metadata={**self.documents[i].metadata, "similarity_score": score, "search_type": "keyword_fallback"}
                    )
                    results.append(doc)
                except Exception:
                    continue
            
            print(f"🔍 Busca por palavras-chave encontrou {len(results)} resultados")
            return results
            
        except Exception as e:
            print(f"⚠️ Erro na busca por palavras-chave: {e}")
            return []