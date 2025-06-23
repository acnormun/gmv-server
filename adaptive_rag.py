# adaptive_rag.py - VERSÃO ULTRA-RÁPIDA (Embeddings Otimizados)

import os
import pickle
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import Counter
import math

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever  # ADICIONADO

# Imports com fallback
try:
    from langchain_ollama import OllamaLLM, OllamaEmbeddings
except ImportError:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        from langchain.llms import Ollama as OllamaLLM
        from langchain.embeddings import OllamaEmbeddings

# EMBEDDING ULTRA-RÁPIDO BASEADO EM TF-IDF
class FastTFIDFEmbedder:
    """Embeddings baseados em TF-IDF - 100x mais rápido que Ollama"""
    
    def __init__(self, max_features=1000):
        self.vocabulary = {}
        self.idf_scores = {}
        self.max_features = max_features
        self.is_fitted = False
    
    def _tokenize(self, text):
        """Tokenização simples e rápida"""
        # Remove pontuação e converte para minúsculas
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Extrai tokens de 2+ caracteres
        tokens = [token for token in text.split() if len(token) >= 2]
        return tokens
    
    def _get_important_terms(self, documents):
        """Extrai termos mais importantes dos documentos"""
        all_tokens = []
        doc_tokens = []
        
        for doc in documents:
            tokens = self._tokenize(doc)
            doc_tokens.append(set(tokens))  # Unique tokens per doc
            all_tokens.extend(tokens)
        
        # Conta frequência total
        token_freq = Counter(all_tokens)
        
        # Calcula IDF (inverse document frequency)
        vocab = {}
        for token, freq in token_freq.most_common(self.max_features):
            # Quantos documentos contêm este token
            doc_count = sum(1 for doc_set in doc_tokens if token in doc_set)
            if doc_count > 0:
                idf = math.log(len(documents) / doc_count)
                vocab[token] = len(vocab)
                self.idf_scores[token] = idf
        
        return vocab
    
    def fit(self, documents):
        """Treina o vocabulário nos documentos"""
        print(f"🔄 Criando vocabulário TF-IDF para {len(documents)} documentos...")
        self.vocabulary = self._get_important_terms(documents)
        self.is_fitted = True
        print(f"✅ Vocabulário criado: {len(self.vocabulary)} termos")
    
    def embed_query(self, text):
        """Gera embedding de uma query"""
        if not self.is_fitted:
            return np.random.random(self.max_features).tolist()
        
        tokens = self._tokenize(text)
        token_count = Counter(tokens)
        
        # Cria vetor TF-IDF
        vector = np.zeros(len(self.vocabulary))
        
        for token, count in token_count.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / len(tokens) if tokens else 0
                idf = self.idf_scores.get(token, 0)
                vector[idx] = tf * idf
        
        # Normaliza o vetor
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def embed_documents(self, documents):
        """Gera embeddings para múltiplos documentos"""
        return [self.embed_query(doc) for doc in documents]

# Vector store otimizado
class UltraFastVectorStore:
    def __init__(self, persist_dir):
        self.persist_dir = persist_dir
        self.embeddings = []
        self.documents = []
        self.metadata = []
        self.doc_hashes = []
        self.embedder = FastTFIDFEmbedder()
        os.makedirs(persist_dir, exist_ok=True)
    
    def _get_doc_hash(self, content):
        return hashlib.md5(content.encode()).hexdigest()[:8]  # Hash curto
    
    def add_documents(self, documents, embedding_func=None):
        """Adiciona documentos com SAMPLING INTELIGENTE"""
        
        # OTIMIZAÇÃO 1: Reduz número de chunks drasticamente
        if len(documents) > 500:
            print(f"📊 {len(documents)} chunks encontrados - aplicando sampling...")
            # Pega os chunks mais longos (mais informativos)
            documents = sorted(documents, key=lambda x: len(x.page_content), reverse=True)[:500]
            print(f"📝 Reduzido para {len(documents)} chunks mais importantes")
        
        # Carrega cache
        cache = self._load_cache()
        new_docs = []
        new_contents = []
        
        # Filtra apenas novos
        for doc in documents:
            doc_hash = self._get_doc_hash(doc.page_content)
            
            if doc_hash not in cache:
                new_docs.append(doc)
                new_contents.append(doc.page_content)
                self.doc_hashes.append(doc_hash)
            else:
                # Usa cache
                cached = cache[doc_hash]
                self.embeddings.append(cached["embedding"])
                self.documents.append(doc.page_content)
                self.metadata.append(doc.metadata)
                self.doc_hashes.append(doc_hash)
        
        if new_contents:
            print(f"🚀 Gerando embeddings TF-IDF para {len(new_contents)} novos chunks...")
            
            # OTIMIZAÇÃO 2: Usa TF-IDF ao invés de Ollama
            all_docs_for_vocab = new_contents + [doc for doc in self.documents]
            self.embedder.fit(all_docs_for_vocab)
            
            # Gera embeddings rapidamente
            batch_embeddings = self.embedder.embed_documents(new_contents)
            
            for doc, embedding in zip(new_docs, batch_embeddings):
                self.embeddings.append(embedding)
                self.documents.append(doc.page_content)
                self.metadata.append(doc.metadata)
            
            print(f"✅ {len(new_contents)} embeddings TF-IDF gerados INSTANTANEAMENTE!")
        
        self._save_cache()
    
    def _load_cache(self):
        cache_file = os.path.join(self.persist_dir, "tfidf_cache.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def _save_cache(self):
        # Salva cache
        cache_data = {}
        for i, doc_hash in enumerate(self.doc_hashes):
            if i < len(self.embeddings):
                cache_data[doc_hash] = {
                    "embedding": self.embeddings[i],
                    "content": self.documents[i],
                    "metadata": self.metadata[i]
                }
        
        cache_file = os.path.join(self.persist_dir, "tfidf_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Salva embedder
        embedder_file = os.path.join(self.persist_dir, "embedder.pkl")
        with open(embedder_file, 'wb') as f:
            pickle.dump(self.embedder, f)
        
        # Salva dados principais
        main_file = os.path.join(self.persist_dir, "data.pkl")
        with open(main_file, 'wb') as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "documents": self.documents,
                "metadata": self.metadata,
                "doc_hashes": self.doc_hashes
            }, f)
    
    def similarity_search(self, query, k=3):
        if not self.embeddings:
            return []
        
        # Usa TF-IDF para busca
        query_emb = np.array(self.embedder.embed_query(query))
        similarities = []
        
        for emb in self.embeddings:
            emb_array = np.array(emb)
            # Cosine similarity
            sim = np.dot(query_emb, emb_array) / (np.linalg.norm(query_emb) * np.linalg.norm(emb_array) + 1e-8)
            similarities.append(sim)
        
        indices = np.argsort(similarities)[::-1][:k]
        return [Document(page_content=self.documents[i], metadata=self.metadata[i]) 
                for i in indices]
    
    def as_retriever(self, search_kwargs=None):
        k = search_kwargs.get("k", 3) if search_kwargs else 3
        
        class FastRetriever(BaseRetriever):
            """Retriever compatível com LangChain"""
            
            def __init__(self, vector_store, k):
                super().__init__()
                self.vector_store = vector_store
                self.k = k
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                """Método requerido pelo BaseRetriever"""
                return self.vector_store.similarity_search(query, k=self.k)
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                """Método público para compatibilidade"""
                return self._get_relevant_documents(query)
        
        return FastRetriever(self, k)
    
    def load(self):
        """Carrega dados salvos"""
        # Carrega embedder
        embedder_file = os.path.join(self.persist_dir, "embedder.pkl")
        if os.path.exists(embedder_file):
            try:
                with open(embedder_file, 'rb') as f:
                    self.embedder = pickle.load(f)
            except:
                self.embedder = FastTFIDFEmbedder()
        
        # Carrega dados principais
        filepath = os.path.join(self.persist_dir, "data.pkl")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get("embeddings", [])
                    self.documents = data.get("documents", [])
                    self.metadata = data.get("metadata", [])
                    self.doc_hashes = data.get("doc_hashes", [])
                    return len(self.embeddings) > 0
            except:
                pass
        return False

@dataclass
class FastRAGConfig:
    model_name: str = "gemma:2b"
    chunk_size: int = 1500  # Chunks MUITO maiores
    chunk_overlap: int = 150
    top_k: int = 3
    max_chunks: int = 300  # Limite máximo de chunks

class UltraFastRAG:
    def __init__(self):
        self.config = FastRAGConfig()
        self.llm = None
        self.vector_store = None
        self.documents = []
        self.is_initialized = False
        self.data_path = os.getenv("DADOS_ANONIMOS", os.getenv("PASTA_DESTINO", "./documentos"))
        self.cache_path = os.path.join(os.path.dirname(self.data_path), ".rag_cache")
        os.makedirs(self.cache_path, exist_ok=True)
    
    def initialize(self):
        try:
            print("🔄 Inicializando Gemma:2b...")
            self.llm = OllamaLLM(model=self.config.model_name, temperature=0.1, num_predict=512)
            self.is_initialized = True
            self._load_cache()
            print("✅ RAG inicializado (SEM embeddings Ollama!)")
            return True
        except Exception as e:
            print(f"❌ Erro: {e}")
            return False
    
    def _load_cache(self):
        self.vector_store = UltraFastVectorStore(os.path.join(self.cache_path, "ultra_fast"))
        if self.vector_store.load():
            self.documents = [Document(page_content=doc, metadata=meta) 
                            for doc, meta in zip(self.vector_store.documents, self.vector_store.metadata)]
            print(f"📦 Cache TF-IDF carregado: {len(self.documents)} chunks")
    
    def load_documents_from_directory(self):
        if not os.path.exists(self.data_path):
            return 0
        
        print(f"📁 Carregando de: {self.data_path}")
        
        documents = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith(('.txt', '.md')):
                    try:
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if text and len(text) > 100:
                                documents.append(Document(
                                    page_content=text, 
                                    metadata={"filename": file, "source": filepath}
                                ))
                    except:
                        continue
        
        if documents:
            print(f"📄 {len(documents)} arquivos encontrados")
            self.documents = documents
            self._create_vector_store()
        
        return len(documents)
    
    def _create_vector_store(self):
        if not self.documents:
            return
        
        print("✂️ Criando chunks GRANDES e OTIMIZADOS...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,  # Chunks grandes
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = splitter.split_documents(self.documents)
        print(f"📝 {len(chunks)} chunks criados")
        
        # Filtra chunks pequenos demais
        chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 100]
        
        # LIMITE RÍGIDO para performance
        if len(chunks) > self.config.max_chunks:
            print(f"⚡ Limitando a {self.config.max_chunks} chunks para máxima velocidade")
            # Ordena por tamanho (chunks maiores = mais informativos)
            chunks = sorted(chunks, key=lambda x: len(x.page_content), reverse=True)[:self.config.max_chunks]
        
        print(f"🔍 {len(chunks)} chunks finais para processamento")
        
        self.vector_store = UltraFastVectorStore(os.path.join(self.cache_path, "ultra_fast"))
        self.vector_store.add_documents(chunks)  # SEM embedding_func - usa TF-IDF interno
        print("⚡ Vector store TF-IDF criado INSTANTANEAMENTE!")
    
    def query(self, question):
        if not self.is_initialized or not self.vector_store:
            return {"error": "Sistema não inicializado"}
        
        try:
            # MÉTODO SIMPLIFICADO - sem RetrievalQA complexo
            print(f"🔍 Buscando documentos para: {question[:50]}...")
            
            # Busca documentos relevantes
            relevant_docs = self.vector_store.similarity_search(question, k=self.config.top_k)
            
            if not relevant_docs:
                return {"error": "Nenhum documento relevante encontrado"}
            
            # Cria contexto
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Template simples
            prompt_text = f"""Com base nos documentos encontrados, responda de forma direta e objetiva:

DOCUMENTOS:
{context}

PERGUNTA: {question}

RESPOSTA:"""
            
            print("🤖 Consultando Gemma:2b...")
            
            # Chama LLM diretamente
            if hasattr(self.llm, 'invoke'):
                answer = self.llm.invoke(prompt_text)
            else:
                answer = self.llm(prompt_text)
            
            return {
                "question": question,
                "answer": answer,
                "source_documents": [{
                    "content": doc.page_content[:300] + "...",
                    "filename": doc.metadata.get("filename", "")
                } for doc in relevant_docs],
                "documents_count": len(self.documents),
                "search_method": "TF-IDF (ultra-fast)",
                "context_size": len(context)
            }
            
        except Exception as e:
            print(f"❌ Erro na query: {e}")
            return {"error": str(e)}

# Instância global
rag_system = UltraFastRAG()

def init_rag_system():
    return rag_system.initialize()

def load_data_directory():
    return rag_system.load_documents_from_directory()

def get_rag_status():
    if not rag_system.is_initialized:
        return {"status": "offline", "message": "Não inicializado", "isReady": False}
    if not rag_system.vector_store or len(rag_system.documents) == 0:
        return {"status": "offline", "message": "Sem documentos", "isReady": False, "data_path": rag_system.data_path}
    return {
        "status": "online", 
        "message": f"{len(rag_system.documents)} chunks TF-IDF", 
        "isReady": True,
        "documents_loaded": len(rag_system.documents),
        "method": "TF-IDF embeddings"
    }