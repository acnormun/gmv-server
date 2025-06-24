# utils/optimized_vector_store.py

import hashlib
import os
import pickle
import re
from typing import List

import numpy as np
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

# Import local
from utils.smart_tfidf_embedder import SmartTFIDFEmbedder

# Import condicional para Ollama
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        try:
            from langchain.embeddings import OllamaEmbeddings
        except ImportError:
            OllamaEmbeddings = None


class OptimizedVectorStore:
    """Vector store com cache inteligente e busca otimizada"""
    
    def __init__(self, persist_dir, use_ollama=True):
        self.persist_dir = persist_dir
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.doc_hashes = []
        
        # Escolhe estratégia de embedding
        self.use_ollama = use_ollama
        if use_ollama and OllamaEmbeddings:
            try:
                self.ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
                print("✅ Usando Ollama embeddings")
            except Exception as e:
                print(f"⚠️ Ollama indisponível, usando TF-IDF: {e}")
                self.use_ollama = False
                self.tfidf_embedder = SmartTFIDFEmbedder()
        else:
            self.use_ollama = False
            self.tfidf_embedder = SmartTFIDFEmbedder()
        
        os.makedirs(persist_dir, exist_ok=True)
    
    def _get_doc_hash(self, content):
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _normalize_embedding(self, embedding):
        """Normaliza embedding para lista Python"""
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        else:
            try:
                return list(embedding)
            except:
                print(f"⚠️ Erro ao normalizar embedding tipo: {type(embedding)}")
                return []
    
    def _ensure_numpy_array(self, embedding):
        """Garante que embedding seja numpy array"""
        if isinstance(embedding, np.ndarray):
            return embedding
        elif isinstance(embedding, list):
            return np.array(embedding)
        else:
            try:
                return np.array(embedding)
            except:
                print(f"⚠️ Erro ao converter embedding para numpy: {type(embedding)}")
                return np.zeros(384)  # Fallback
    
    def add_documents(self, documents, max_docs=500):
        """Adiciona documentos com filtragem inteligente"""
        print(f"📊 Processando {len(documents)} documentos...")
        
        try:
            # Filtragem por qualidade se necessário
            if len(documents) > max_docs:
                print(f"🎯 Aplicando seleção inteligente (máximo {max_docs})...")
                # Ordena por tamanho (documentos maiores tendem a ser mais informativos)
                documents = sorted(documents, key=lambda x: len(x.page_content), reverse=True)[:max_docs]
                print(f"📝 Selecionados {len(documents)} documentos")
            
            # Carrega cache
            cache = self._load_cache()
            new_docs = []
            new_contents = []
            
            # Identifica novos documentos
            for doc in documents:
                try:
                    doc_hash = self._get_doc_hash(doc.page_content)
                    
                    if doc_hash not in cache:
                        new_docs.append(doc)
                        new_contents.append(doc.page_content)
                        self.doc_hashes.append(doc_hash)
                    else:
                        # Carrega do cache e normaliza
                        cached = cache[doc_hash]
                        embedding = self._normalize_embedding(cached["embedding"])
                        self.embeddings.append(embedding)
                        self.documents.append(doc.page_content)
                        self.metadata.append(doc.metadata)
                        self.doc_hashes.append(doc_hash)
                except Exception as e:
                    print(f"⚠️ Erro ao processar documento: {e}")
                    continue
            
            if new_contents:
                print(f"🚀 Gerando embeddings para {len(new_contents)} novos documentos...")
                
                # Tenta Ollama primeiro
                new_embeddings = None
                if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
                    try:
                        print("🔄 Tentando Ollama embeddings...")
                        # Usa Ollama em batches pequenos para evitar problemas
                        batch_size = 5  # Reduzido para evitar recursão
                        new_embeddings = []
                        for i in range(0, len(new_contents), batch_size):
                            batch = new_contents[i:i+batch_size]
                            batch_emb = self.ollama_embeddings.embed_documents(batch)
                            # Normaliza embeddings do Ollama
                            normalized_batch = [self._normalize_embedding(emb) for emb in batch_emb]
                            new_embeddings.extend(normalized_batch)
                            print(f"   📦 Batch {i//batch_size + 1}/{(len(new_contents)-1)//batch_size + 1}")
                        print("✅ Ollama embeddings gerados")
                    except Exception as e:
                        print(f"❌ Erro Ollama: {e}. Usando TF-IDF...")
                        self.use_ollama = False
                        new_embeddings = None
                        # Remove referência ao Ollama problemático
                        if hasattr(self, 'ollama_embeddings'):
                            del self.ollama_embeddings
                
                # Se Ollama falhou, usa TF-IDF
                if new_embeddings is None:
                    try:
                        print("🔄 Usando TF-IDF embeddings...")
                        self.tfidf_embedder = SmartTFIDFEmbedder()
                        
                        # Prepara documentos para treino (limita para evitar problemas)
                        all_docs = new_contents + [doc for doc in self.documents[:100]]  # Limita documentos existentes
                        print(f"📚 Treinando TF-IDF com {len(all_docs)} documentos...")
                        
                        self.tfidf_embedder.fit(all_docs)
                        
                        if self.tfidf_embedder.is_fitted:
                            embeddings_raw = self.tfidf_embedder.embed_documents(new_contents)
                            new_embeddings = [self._normalize_embedding(emb) for emb in embeddings_raw]
                            print("✅ TF-IDF embeddings gerados")
                        else:
                            print("⚠️ TF-IDF não foi treinado corretamente")
                            new_embeddings = []
                            
                    except Exception as e:
                        print(f"❌ Erro TF-IDF: {e}")
                        new_embeddings = []
                
                # Adiciona os novos embeddings (mesmo que sejam vazios)
                if new_embeddings and len(new_embeddings) == len(new_docs):
                    for doc, embedding in zip(new_docs, new_embeddings):
                        self.embeddings.append(embedding)
                        self.documents.append(doc.page_content)
                        self.metadata.append(doc.metadata)
                    print(f"✅ {len(new_contents)} embeddings adicionados!")
                else:
                    print(f"⚠️ Problema com embeddings - adicionando documentos sem embeddings")
                    # Adiciona documentos mesmo sem embeddings válidos
                    fallback_embedding = [0.0] * 100  # Embedding fallback
                    for doc in new_docs:
                        self.embeddings.append(fallback_embedding)
                        self.documents.append(doc.page_content)
                        self.metadata.append(doc.metadata)
            
            # Salva cache de forma segura
            try:
                self._save_cache()
            except Exception as e:
                print(f"⚠️ Erro ao salvar cache: {e}")
                
        except Exception as e:
            print(f"❌ Erro crítico ao adicionar documentos: {e}")
            # Continua mesmo com erro para não quebrar o sistema
    
    def similarity_search(self, query, k=4, min_score=0.05):  # Reduzido de 0.1 para 0.05
        """Busca por similaridade otimizada com busca híbrida"""
        if not self.embeddings:
            print("⚠️ Nenhum embedding disponível")
            return []
        
        print(f"🔍 Buscando para: '{query[:50]}...'")
        
        # Primeiro tenta busca por keywords específicas (números de processo)
        keyword_results = self._keyword_search(query, k)
        if keyword_results:
            print(f"🎯 {len(keyword_results)} resultados encontrados por busca de keywords")
            return keyword_results
        
        # Gera embedding da query
        query_embedding = None
        
        # Primeiro tenta Ollama
        if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
            try:
                print("🔄 Tentando Ollama embeddings...")
                query_embedding_raw = self.ollama_embeddings.embed_query(query)
                query_embedding = self._normalize_embedding(query_embedding_raw)
                print("✅ Ollama embedding gerado")
            except Exception as e:
                print(f"⚠️ Erro Ollama embedding: {e}")
                self.use_ollama = False
                if hasattr(self, 'ollama_embeddings'):
                    del self.ollama_embeddings  # Remove referência problemática
        
        # Se Ollama falhou, usa TF-IDF
        if not self.use_ollama or query_embedding is None:
            try:
                print("🔄 Usando TF-IDF embeddings...")
                
                # Inicializa TF-IDF se necessário
                if not hasattr(self, 'tfidf_embedder') or self.tfidf_embedder is None:
                    print("🔄 Inicializando TF-IDF embedder...")
                    self.tfidf_embedder = SmartTFIDFEmbedder()
                    
                    # Treina com documentos existentes se disponíveis
                    if self.documents:
                        print(f"📚 Treinando TF-IDF com {len(self.documents)} documentos...")
                        # Limita documentos para evitar problemas de memória
                        train_docs = self.documents[:500] if len(self.documents) > 500 else self.documents
                        self.tfidf_embedder.fit(train_docs)
                
                # Verifica se o embedder foi treinado
                if not self.tfidf_embedder.is_fitted:
                    print("⚠️ TF-IDF embedder não foi treinado corretamente")
                    return []
                
                query_embedding_raw = self.tfidf_embedder.embed_query(query)
                query_embedding = self._normalize_embedding(query_embedding_raw)
                print("✅ TF-IDF embedding gerado")
                
            except Exception as e:
                print(f"❌ Erro TF-IDF embedding: {e}")
                return []
        
        if query_embedding is None or len(query_embedding) == 0:
            print("❌ Falha ao gerar embedding da query")
            return []
        
        # Converte para numpy para cálculos
        try:
            query_vec = self._ensure_numpy_array(query_embedding)
            
            if len(query_vec) == 0:
                print("❌ Vetor de query vazio")
                return []
            
            # Calcula similaridades
            similarities = []
            for i, doc_embedding in enumerate(self.embeddings):
                try:
                    doc_vec = self._ensure_numpy_array(doc_embedding)
                    
                    # Verifica se os vetores têm o mesmo tamanho
                    if len(query_vec) != len(doc_vec) or len(doc_vec) == 0:
                        continue
                    
                    # Similaridade coseno segura
                    norm_query = np.linalg.norm(query_vec)
                    norm_doc = np.linalg.norm(doc_vec)
                    
                    if norm_query == 0 or norm_doc == 0:
                        sim = 0.0
                    else:
                        sim = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                        # Garante que sim está entre -1 e 1
                        sim = max(-1.0, min(1.0, float(sim)))
                    
                    # Boost robusto para matches de palavras-chave
                    try:
                        keyword_boost = self._calculate_keyword_boost(query, i)
                        sim += keyword_boost
                    except Exception:
                        pass  # Ignora erros no boost
                    
                    similarities.append((i, float(sim)))
                    
                except Exception as e:
                    print(f"⚠️ Erro ao calcular similaridade doc {i}: {e}")
                    similarities.append((i, 0.0))
            
            # Debug: mostra os melhores scores antes de filtrar
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_10 = similarities[:10]
            print(f"🔍 Top 10 scores antes do filtro: {[f'{s:.3f}' for _, s in top_10]}")
            
            # Se nenhum resultado passou no threshold, reduz temporariamente
            if not any(score >= min_score for _, score in similarities):
                min_score = 0.01  # Threshold muito baixo para garantir alguns resultados
                print(f"⚠️ Nenhum resultado com score >= {min_score}, reduzindo threshold para 0.01")
            
            # Filtra por score mínimo
            similarities = [(i, score) for i, score in similarities if score >= min_score]
            
            # Retorna top-k
            selected = similarities[:k]
            print(f"📊 Scores finais: {[f'{s:.3f}' for _, s in selected]}")
            
            results = []
            for i, score in selected:
                try:
                    if i < len(self.documents) and i < len(self.metadata):
                        doc = Document(
                            page_content=self.documents[i], 
                            metadata={**self.metadata[i], "similarity_score": score}
                        )
                        results.append(doc)
                except Exception as e:
                    print(f"⚠️ Erro ao criar documento resultado {i}: {e}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no cálculo de similaridade: {e}")
            return []
    
    def _load_cache(self):
        cache_file = os.path.join(self.persist_dir, "smart_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"⚠️ Erro ao carregar cache: {e}")
        return {}
    
    def _save_cache(self):
        try:
            # Salva cache
            cache_data = {}
            for i, doc_hash in enumerate(self.doc_hashes):
                if i < len(self.embeddings):
                    # Normaliza antes de salvar
                    embedding = self._normalize_embedding(self.embeddings[i])
                    cache_data[doc_hash] = {
                        "embedding": embedding,
                        "content": self.documents[i],
                        "metadata": self.metadata[i]
                    }
            
            cache_file = os.path.join(self.persist_dir, "smart_cache.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Salva embedder TF-IDF se usado
            if not self.use_ollama and hasattr(self, 'tfidf_embedder'):
                embedder_file = os.path.join(self.persist_dir, "tfidf_embedder.pkl")
                with open(embedder_file, 'wb') as f:
                    pickle.dump(self.tfidf_embedder, f)
                    
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")
    
    def load(self):
        """Carrega dados salvos"""
        try:
            cache = self._load_cache()
            
            for doc_hash, data in cache.items():
                embedding = self._normalize_embedding(data["embedding"])
                self.embeddings.append(embedding)
                self.documents.append(data["content"])
                self.metadata.append(data["metadata"])
                self.doc_hashes.append(doc_hash)
            
            # Carrega embedder TF-IDF se existe
            embedder_file = os.path.join(self.persist_dir, "tfidf_embedder.pkl")
            if os.path.exists(embedder_file):
                try:
                    with open(embedder_file, 'rb') as f:
                        self.tfidf_embedder = pickle.load(f)
                    self.use_ollama = False
                except Exception as e:
                    print(f"⚠️ Erro ao carregar TF-IDF embedder: {e}")
            
            return len(self.embeddings) > 0
            
        except Exception as e:
            print(f"⚠️ Erro ao carregar dados: {e}")
            return False
    
    def _keyword_search(self, query, k=4):
        """Busca específica por keywords importantes (números de processo, etc.)"""
        try:
            query_lower = query.lower()
            
            # Detecta número de processo
            process_pattern = r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            process_matches = re.findall(process_pattern, query)
            
            # Detecta outras keywords importantes
            keywords = []
            
            if process_matches:
                keywords.extend(process_matches)
                print(f"🔍 Detectado número de processo: {process_matches}")
            
            # Busca por outras palavras-chave relevantes
            important_words = re.findall(r'\b(triagem|agravo|processo|acórdão|decisão|sentença|liminar|TEA|autist[ao]|ABA|terapia)\b', query_lower)
            keywords.extend(important_words)
            
            if not keywords:
                return []
            
            # Busca nos documentos e metadados
            matches = []
            for i, (doc_content, metadata) in enumerate(zip(self.documents, self.metadata)):
                score = 0.0
                
                # Busca no conteúdo
                content_lower = doc_content.lower()
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        score += 0.3
                
                # Busca nos metadados (peso maior)
                for key, value in metadata.items():
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        for keyword in keywords:
                            if keyword.lower() in value_lower:
                                score += 0.5  # Peso maior para metadados
                
                if score > 0:
                    matches.append((i, score))
            
            # Ordena por score e retorna top-k
            matches.sort(key=lambda x: x[1], reverse=True)
            selected = matches[:k]
            
            if selected:
                print(f"🎯 Busca por keywords encontrou {len(selected)} resultados")
                
                results = []
                for i, score in selected:
                    try:
                        doc = Document(
                            page_content=self.documents[i],
                            metadata={**self.metadata[i], "similarity_score": score, "search_type": "keyword"}
                        )
                        results.append(doc)
                    except Exception as e:
                        print(f"⚠️ Erro ao criar resultado keyword {i}: {e}")
                
                return results
            
            return []
            
        except Exception as e:
            print(f"⚠️ Erro na busca por keywords: {e}")
            return []
    
    def _calculate_keyword_boost(self, query, doc_index):
        """Calcula boost baseado em matches de palavras-chave"""
        try:
            if doc_index >= len(self.documents):
                return 0.0
            
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            doc_text = self.documents[doc_index].lower()
            
            # Palavras do documento (primeiros 1000 chars para performance)
            doc_words = set(re.findall(r'\b\w{3,}\b', doc_text[:1000]))
            
            if not query_words or not doc_words:
                return 0.0
            
            # Calcula overlap básico
            overlap = len(query_words.intersection(doc_words))
            basic_boost = (overlap / len(query_words)) * 0.2
            
            # Boost extra para números de processo
            process_pattern = r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            query_processes = set(re.findall(process_pattern, query))
            doc_processes = set(re.findall(process_pattern, doc_text))
            
            if query_processes and doc_processes:
                process_overlap = len(query_processes.intersection(doc_processes))
                if process_overlap > 0:
                    basic_boost += 0.5  # Boost alto para número de processo
            
            # Boost para metadados se disponível
            if doc_index < len(self.metadata):
                metadata = self.metadata[doc_index]
                for key, value in metadata.items():
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        for word in query_words:
                            if word in value_lower:
                                basic_boost += 0.1  # Boost para match em metadados
            
            return min(basic_boost, 0.8)  # Limita boost máximo
            
        except Exception as e:
            print(f"⚠️ Erro ao calcular keyword boost: {e}")
            return 0.0
    
    def test_similarity_calculation(self, query, max_docs=10):
        """Testa cálculo de similaridade mostrando scores brutos"""
        try:
            print(f"\n🧪 TESTE DE SIMILARIDADE PARA: '{query}'")
            print("=" * 60)
            
            # Gera embedding da query
            query_embedding = None
            
            if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
                try:
                    query_embedding_raw = self.ollama_embeddings.embed_query(query)
                    query_embedding = self._normalize_embedding(query_embedding_raw)
                    print("✅ Ollama embedding gerado para query")
                except Exception as e:
                    print(f"❌ Erro Ollama: {e}")
                    self.use_ollama = False
            
            if not self.use_ollama or query_embedding is None:
                if not hasattr(self, 'tfidf_embedder') or self.tfidf_embedder is None:
                    self.tfidf_embedder = SmartTFIDFEmbedder()
                    if self.documents:
                        self.tfidf_embedder.fit(self.documents[:100])
                
                if self.tfidf_embedder.is_fitted:
                    query_embedding_raw = self.tfidf_embedder.embed_query(query)
                    query_embedding = self._normalize_embedding(query_embedding_raw)
                    print("✅ TF-IDF embedding gerado para query")
                else:
                    print("❌ Nenhum embedding disponível")
                    return
            
            if not query_embedding:
                print("❌ Falha ao gerar embedding da query")
                return
            
            query_vec = self._ensure_numpy_array(query_embedding)
            print(f"📊 Tamanho do vetor da query: {len(query_vec)}")
            
            # Calcula similaridades para uma amostra
            similarities = []
            max_check = min(max_docs, len(self.embeddings))
            
            for i in range(max_check):
                try:
                    doc_vec = self._ensure_numpy_array(self.embeddings[i])
                    
                    if len(query_vec) != len(doc_vec):
                        continue
                    
                    # Similaridade coseno
                    norm_query = np.linalg.norm(query_vec)
                    norm_doc = np.linalg.norm(doc_vec)
                    
                    if norm_query > 0 and norm_doc > 0:
                        sim = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                        sim = max(-1.0, min(1.0, float(sim)))
                        
                        # Boost de keywords
                        boost = self._calculate_keyword_boost(query, i)
                        final_sim = sim + boost
                        
                        similarities.append((i, sim, boost, final_sim))
                    
                except Exception as e:
                    print(f"⚠️ Erro no doc {i}: {e}")
            
            # Ordena por similaridade final
            similarities.sort(key=lambda x: x[3], reverse=True)
            
            print(f"\n📊 TOP {min(10, len(similarities))} RESULTADOS:")
            print("Idx | Base  | Boost | Final | Preview")
            print("-" * 60)
            
            for i, (doc_idx, base_sim, boost, final_sim) in enumerate(similarities[:10]):
                doc_preview = ""
                if doc_idx < len(self.documents):
                    content = self.documents[doc_idx][:100].replace('\n', ' ')
                    doc_preview = content + "..." if len(content) == 100 else content
                
                print(f"{doc_idx:3d} | {base_sim:5.3f} | {boost:5.3f} | {final_sim:5.3f} | {doc_preview}")
            
            print(f"\n📈 ESTATÍSTICAS:")
            if similarities:
                scores = [s[3] for s in similarities]
                print(f"   - Maior score: {max(scores):.3f}")
                print(f"   - Menor score: {min(scores):.3f}")
                print(f"   - Score médio: {sum(scores)/len(scores):.3f}")
                print(f"   - Scores > 0.1: {len([s for s in scores if s > 0.1])}")
                print(f"   - Scores > 0.05: {len([s for s in scores if s > 0.05])}")
                print(f"   - Scores > 0.01: {len([s for s in scores if s > 0.01])}")
            
            return similarities
            
        except Exception as e:
            print(f"❌ Erro no teste de similaridade: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def as_retriever(self, search_kwargs=None):
        k = search_kwargs.get("k", 4) if search_kwargs else 4
        min_score = search_kwargs.get("min_score", 0.05) if search_kwargs else 0.05  # Threshold corrigido
        
        class SmartRetriever(BaseRetriever):
            def __init__(self, vector_store, k, min_score):
                super().__init__()
                self.vector_store = vector_store
                self.k = k
                self.min_score = min_score
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.vector_store.similarity_search(query, k=self.k, min_score=self.min_score)
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)
        
        return SmartRetriever(self, k, min_score)