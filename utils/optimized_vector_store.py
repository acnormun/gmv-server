import hashlib
import os
import pickle
import re
from typing import List

import numpy as np
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from utils.smart_tfidf_embedder import SmartTFIDFEmbedder
from utils.inverted_index import InvertedIndex

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
    def __init__(self, persist_dir, use_ollama=True):
        self.persist_dir = persist_dir
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.doc_hashes = []
        self.use_ollama = use_ollama
        self.inverted_index = InvertedIndex()
        if use_ollama and OllamaEmbeddings:
            try:
                self.ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
            except Exception:
                self.use_ollama = False
                self.tfidf_embedder = SmartTFIDFEmbedder()
        else:
            self.use_ollama = False
            self.tfidf_embedder = SmartTFIDFEmbedder()
        os.makedirs(persist_dir, exist_ok=True)

    def _get_doc_hash(self, content):
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _normalize_embedding(self, embedding):
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        elif isinstance(embedding, list):
            return embedding
        else:
            try:
                return list(embedding)
            except:
                return []

    def _ensure_numpy_array(self, embedding):
        if isinstance(embedding, np.ndarray):
            return embedding
        elif isinstance(embedding, list):
            return np.array(embedding)
        else:
            try:
                return np.array(embedding)
            except:
                return np.zeros(384)

    def add_documents(self, documents, max_docs=500):
        try:
            if len(documents) > max_docs:
                documents = sorted(documents, key=lambda x: len(x.page_content), reverse=True)[:max_docs]
            cache = self._load_cache()
            new_docs = []
            new_contents = []
            for doc in documents:
                try:
                    doc_hash = self._get_doc_hash(doc.page_content)
                    if doc_hash not in cache:
                        new_docs.append(doc)
                        new_contents.append(doc.page_content)
                        self.doc_hashes.append(doc_hash)
                        self.inverted_index.add_document(
                            doc.page_content,
                            len(self.documents) - 1,
                            doc.metadata.values(),
                        )
                    else:
                        cached = cache[doc_hash]
                        embedding = self._normalize_embedding(cached["embedding"])
                        self.embeddings.append(embedding)
                        self.documents.append(doc.page_content)
                        self.metadata.append(doc.metadata)
                        self.doc_hashes.append(doc_hash)
                        self.inverted_index.add_document(
                            doc.page_content,
                            len(self.documents) - 1,
                            doc.metadata.values(),
                        )
                except Exception:
                    continue
            if new_contents:
                new_embeddings = None
                if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
                    try:
                        batch_size = 5
                        new_embeddings = []
                        for i in range(0, len(new_contents), batch_size):
                            batch = new_contents[i:i+batch_size]
                            batch_emb = self.ollama_embeddings.embed_documents(batch)
                            normalized_batch = [self._normalize_embedding(emb) for emb in batch_emb]
                            new_embeddings.extend(normalized_batch)
                    except Exception:
                        self.use_ollama = False
                        new_embeddings = None
                        if hasattr(self, 'ollama_embeddings'):
                            del self.ollama_embeddings
                if new_embeddings is None:
                    try:
                        self.tfidf_embedder = SmartTFIDFEmbedder()
                        all_docs = new_contents + [doc for doc in self.documents[:100]]
                        self.tfidf_embedder.fit(all_docs)
                        if self.tfidf_embedder.is_fitted:
                            embeddings_raw = self.tfidf_embedder.embed_documents(new_contents)
                            new_embeddings = [self._normalize_embedding(emb) for emb in embeddings_raw]
                        else:
                            new_embeddings = []
                    except Exception:
                        new_embeddings = []
                if new_embeddings and len(new_embeddings) == len(new_docs):
                    for doc, embedding in zip(new_docs, new_embeddings):
                        self.embeddings.append(embedding)
                        self.documents.append(doc.page_content)
                        self.metadata.append(doc.metadata)
                        self.inverted_index.add_document(
                            doc.page_content,
                            len(self.documents) - 1,
                            doc.metadata.values(),
                        )
                else:
                    fallback_embedding = [0.0] * 100
                    for doc in new_docs:
                        self.embeddings.append(fallback_embedding)
                        self.documents.append(doc.page_content)
                        self.metadata.append(doc.metadata)
                        self.inverted_index.add_document(
                            doc.page_content,
                            len(self.documents) - 1,
                            doc.metadata.values(),
                        )
            try:
                self._save_cache()
            except Exception:
                pass
        except Exception:
            pass

    def similarity_search(self, query, k=4, min_score=0.05):
        if not self.embeddings:
            return []
        keyword_results = self._keyword_search(query, k)
        if keyword_results:
            return keyword_results
        query_embedding = None
        if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
            try:
                query_embedding_raw = self.ollama_embeddings.embed_query(query)
                query_embedding = self._normalize_embedding(query_embedding_raw)
            except Exception:
                self.use_ollama = False
                if hasattr(self, 'ollama_embeddings'):
                    del self.ollama_embeddings
        if not self.use_ollama or query_embedding is None:
            try:
                if not hasattr(self, 'tfidf_embedder') or self.tfidf_embedder is None:
                    self.tfidf_embedder = SmartTFIDFEmbedder()
                    if self.documents:
                        train_docs = self.documents[:500] if len(self.documents) > 500 else self.documents
                        self.tfidf_embedder.fit(train_docs)
                if not self.tfidf_embedder.is_fitted:
                    return []
                query_embedding_raw = self.tfidf_embedder.embed_query(query)
                query_embedding = self._normalize_embedding(query_embedding_raw)
            except Exception:
                return []
        if query_embedding is None or len(query_embedding) == 0:
            return []
        try:
            query_vec = self._ensure_numpy_array(query_embedding)
            if len(query_vec) == 0:
                return []
            similarities = []
            candidate_indices = self.inverted_index.query(query)
            indices_to_check = candidate_indices if candidate_indices else range(len(self.embeddings))
            for i in indices_to_check:
                doc_embedding = self.embeddings[i]
                try:
                    doc_vec = self._ensure_numpy_array(doc_embedding)
                    if len(query_vec) != len(doc_vec) or len(doc_vec) == 0:
                        continue
                    norm_query = np.linalg.norm(query_vec)
                    norm_doc = np.linalg.norm(doc_vec)
                    if norm_query == 0 or norm_doc == 0:
                        sim = 0.0
                    else:
                        sim = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                        sim = max(-1.0, min(1.0, float(sim)))
                    try:
                        keyword_boost = self._calculate_keyword_boost(query, i)
                        sim += keyword_boost
                    except Exception:
                        pass
                    similarities.append((i, float(sim)))
                except Exception:
                    similarities.append((i, 0.0))
            similarities.sort(key=lambda x: x[1], reverse=True)
            if not any(score >= min_score for _, score in similarities):
                min_score = 0.001
            similarities = [(i, score) for i, score in similarities if score >= min_score]
            selected = similarities[:k]
            results = []
            for i, score in selected:
                try:
                    if i < len(self.documents) and i < len(self.metadata):
                        doc = Document(
                            page_content=self.documents[i], 
                            metadata={**self.metadata[i], "similarity_score": score}
                        )
                        results.append(doc)
                except Exception:
                    pass
            return results
        except Exception:
            return []
        
    def _lexical_similarity(self, query_tokens, doc_index):
        try:
            doc_tokens = InvertedIndex._tokenize(self.documents[doc_index])
            if doc_index < len(self.metadata):
                for value in self.metadata[doc_index].values():
                    if value and isinstance(value, str):
                        doc_tokens.extend(InvertedIndex._tokenize(value))
            if not query_tokens or not doc_tokens:
                return 0.0
            q_set = set(query_tokens)
            d_set = set(doc_tokens)
            return len(q_set.intersection(d_set)) / len(q_set)
        except Exception:
            return 0.0

    def hybrid_search(
        self,
        query: str,
        k: int = 4,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        min_score: float = 0.05,
    ):
        if not self.embeddings:
            return []
        query_tokens = InvertedIndex._tokenize(query)
        query_embedding = None
        if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
            try:
                query_embedding_raw = self.ollama_embeddings.embed_query(query)
                query_embedding = self._normalize_embedding(query_embedding_raw)
            except Exception:
                self.use_ollama = False
                if hasattr(self, 'ollama_embeddings'):
                    del self.ollama_embeddings
        if not self.use_ollama or query_embedding is None:
            try:
                if not hasattr(self, 'tfidf_embedder') or self.tfidf_embedder is None:
                    self.tfidf_embedder = SmartTFIDFEmbedder()
                    if self.documents:
                        train_docs = self.documents[:500] if len(self.documents) > 500 else self.documents
                        self.tfidf_embedder.fit(train_docs)
                if not self.tfidf_embedder.is_fitted:
                    return []
                query_embedding_raw = self.tfidf_embedder.embed_query(query)
                query_embedding = self._normalize_embedding(query_embedding_raw)
            except Exception:
                return []
        if query_embedding is None or len(query_embedding) == 0:
            return []
        try:
            query_vec = self._ensure_numpy_array(query_embedding)
            if len(query_vec) == 0:
                return []
            similarities = []
            candidate_indices = self.inverted_index.query(query)
            indices_to_check = candidate_indices if candidate_indices else range(len(self.embeddings))
            for i in indices_to_check:
                doc_embedding = self.embeddings[i]
                try:
                    doc_vec = self._ensure_numpy_array(doc_embedding)
                    if len(query_vec) != len(doc_vec) or len(doc_vec) == 0:
                        continue
                    norm_query = np.linalg.norm(query_vec)
                    norm_doc = np.linalg.norm(doc_vec)
                    if norm_query == 0 or norm_doc == 0:
                        sem_sim = 0.0
                    else:
                        sem_sim = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                        sem_sim = max(-1.0, min(1.0, float(sem_sim)))
                    lex_sim = self._lexical_similarity(query_tokens, i)
                    final_score = semantic_weight * sem_sim + lexical_weight * lex_sim
                    similarities.append((i, float(final_score)))
                except Exception:
                    similarities.append((i, 0.0))
            similarities.sort(key=lambda x: x[1], reverse=True)
            if not any(score >= min_score for _, score in similarities):
                min_score = 0.001
            similarities = [(i, score) for i, score in similarities if score >= min_score]
            selected = similarities[:k]
            results = []
            for i, score in selected:
                try:
                    if i < len(self.documents) and i < len(self.metadata):
                        doc = Document(
                            page_content=self.documents[i],
                            metadata={**self.metadata[i], "similarity_score": score, "search_type": "hybrid"},
                        )
                        results.append(doc)
                except Exception:
                    pass
            return results
        except Exception:
            return []

    def _load_cache(self):
        cache_file = os.path.join(self.persist_dir, "smart_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            cache_data = {}
            for i, doc_hash in enumerate(self.doc_hashes):
                if i < len(self.embeddings):
                    embedding = self._normalize_embedding(self.embeddings[i])
                    cache_data[doc_hash] = {
                        "embedding": embedding,
                        "content": self.documents[i],
                        "metadata": self.metadata[i]
                    }
            cache_file = os.path.join(self.persist_dir, "smart_cache.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            if not self.use_ollama and hasattr(self, 'tfidf_embedder'):
                embedder_file = os.path.join(self.persist_dir, "tfidf_embedder.pkl")
                with open(embedder_file, 'wb') as f:
                    pickle.dump(self.tfidf_embedder, f)
        except Exception:
            pass

    def load(self):
        try:
            cache = self._load_cache()
            for doc_hash, data in cache.items():
                embedding = self._normalize_embedding(data["embedding"])
                self.embeddings.append(embedding)
                self.documents.append(data["content"])
                self.metadata.append(data["metadata"])
                self.doc_hashes.append(doc_hash)
                self.inverted_index.add_document(
                    data["content"],
                    len(self.documents) - 1,
                    data["metadata"].values() if isinstance(data["metadata"], dict) else None,
                )
            embedder_file = os.path.join(self.persist_dir, "tfidf_embedder.pkl")
            if os.path.exists(embedder_file):
                try:
                    with open(embedder_file, 'rb') as f:
                        self.tfidf_embedder = pickle.load(f)
                    self.use_ollama = False
                except Exception:
                    pass
            return len(self.embeddings) > 0
        except Exception:
            return False

    def _keyword_search(self, query, k=4):
        try:
            query_lower = query.lower()
            process_pattern = r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            process_matches = re.findall(process_pattern, query)
            keywords = []
            if process_matches:
                keywords.extend(process_matches)
            important_words = re.findall(r'\b(triagem|agravo|processo|acórdão|decisão|sentença|liminar|TEA|autist[ao]|ABA|terapia)\b', query_lower)
            keywords.extend(important_words)
            if not keywords:
                return []
            matches = []
            for i, (doc_content, metadata) in enumerate(zip(self.documents, self.metadata)):
                score = 0.0
                content_lower = doc_content.lower()
                for keyword in keywords:
                    if keyword.lower() in content_lower:
                        score += 0.3
                for key, value in metadata.items():
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        for keyword in keywords:
                            if keyword.lower() in value_lower:
                                score += 0.5
                if score > 0:
                    matches.append((i, score))
            matches.sort(key=lambda x: x[1], reverse=True)
            selected = matches[:k]
            if selected:
                results = []
                for i, score in selected:
                    try:
                        doc = Document(
                            page_content=self.documents[i],
                            metadata={**self.metadata[i], "similarity_score": score, "search_type": "keyword"}
                        )
                        results.append(doc)
                    except Exception:
                        pass
                return results
            return []
        except Exception:
            return []

    def _calculate_keyword_boost(self, query, doc_index):
        try:
            if doc_index >= len(self.documents):
                return 0.0
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            doc_text = self.documents[doc_index].lower()
            doc_words = set(re.findall(r'\b\w{3,}\b', doc_text[:1000]))
            if not query_words or not doc_words:
                return 0.0
            overlap = len(query_words.intersection(doc_words))
            basic_boost = (overlap / len(query_words)) * 0.2
            process_pattern = r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}'
            query_processes = set(re.findall(process_pattern, query))
            doc_processes = set(re.findall(process_pattern, doc_text))
            if query_processes and doc_processes:
                process_overlap = len(query_processes.intersection(doc_processes))
                if process_overlap > 0:
                    basic_boost += 0.5
            if doc_index < len(self.metadata):
                metadata = self.metadata[doc_index]
                for key, value in metadata.items():
                    if value and isinstance(value, str):
                        value_lower = value.lower()
                        for word in query_words:
                            if word in value_lower:
                                basic_boost += 0.1
            return min(basic_boost, 0.8)
        except Exception:
            return 0.0

    def test_similarity_calculation(self, query, max_docs=10):
        try:
            query_embedding = None
            if self.use_ollama and hasattr(self, 'ollama_embeddings') and self.ollama_embeddings:
                try:
                    query_embedding_raw = self.ollama_embeddings.embed_query(query)
                    query_embedding = self._normalize_embedding(query_embedding_raw)
                except Exception:
                    self.use_ollama = False
            if not self.use_ollama or query_embedding is None:
                if not hasattr(self, 'tfidf_embedder') or self.tfidf_embedder is None:
                    self.tfidf_embedder = SmartTFIDFEmbedder()
                    if self.documents:
                        self.tfidf_embedder.fit(self.documents[:100])
                if self.tfidf_embedder.is_fitted:
                    query_embedding_raw = self.tfidf_embedder.embed_query(query)
                    query_embedding = self._normalize_embedding(query_embedding_raw)
                else:
                    return
            if not query_embedding:
                return
            query_vec = self._ensure_numpy_array(query_embedding)
            similarities = []
            max_check = min(max_docs, len(self.embeddings))
            for i in range(max_check):
                try:
                    doc_vec = self._ensure_numpy_array(self.embeddings[i])
                    if len(query_vec) != len(doc_vec):
                        continue
                    norm_query = np.linalg.norm(query_vec)
                    norm_doc = np.linalg.norm(doc_vec)
                    if norm_query > 0 and norm_doc > 0:
                        sim = np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
                        sim = max(-1.0, min(1.0, float(sim)))
                        boost = self._calculate_keyword_boost(query, i)
                        final_sim = sim + boost
                        similarities.append((i, sim, boost, final_sim))
                except Exception:
                    pass
            similarities.sort(key=lambda x: x[3], reverse=True)
            return similarities
        except Exception:
            return []

    def as_retriever(self, search_kwargs=None):
        k = search_kwargs.get("k", 4) if search_kwargs else 4
        min_score = search_kwargs.get("min_score", 0.05) if search_kwargs else 0.05
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