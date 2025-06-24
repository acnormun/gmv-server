# adaptive_rag.py - VERSÃO INTEGRADA E CORRIGIDA

import os
import pickle
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import math
import json

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever

# Imports com fallback robusto
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
            # Fallback para desenvolvimento/teste
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
# 🚀 EMBEDDING HÍBRIDO OTIMIZADO
# =================================================================================

class SmartTFIDFEmbedder:
    """Embeddings TF-IDF otimizados para documentos legais"""
    
    def __init__(self, max_features=1500, use_bigrams=True, min_df=2):
        self.vocabulary = {}
        self.bigram_vocabulary = {}
        self.idf_scores = {}
        self.bigram_idf_scores = {}
        self.max_features = max_features
        self.use_bigrams = use_bigrams
        self.min_df = min_df
        self.is_fitted = False
        
        # Stopwords específicas para documentos legais
        self.legal_stopwords = {
            'a', 'o', 'e', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não',
            'se', 'na', 'por', 'mais', 'das', 'dos', 'ao', 'como', 'mas', 'foi', 'ele',
            'ela', 'seu', 'sua', 'ou', 'ser', 'está', 'ter', 'que', 'são', 'tem', 'nos',
            'foi', 'pela', 'pelo', 'sobre', 'entre', 'após', 'antes', 'durante', 'sem'
        }
        
        # Cache para performance
        self._token_cache = {}
    
    def _smart_tokenize(self, text):
        """Tokenização inteligente para documentos legais"""
        if not text:
            return []
        
        # Cache para evitar reprocessamento
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        if text_hash in self._token_cache:
            return self._token_cache[text_hash]
        
        # Normalização preservando estruturas legais importantes
        text = text.lower()
        
        # Preserva números de processo (formato: NNNNNN-NN.NNNN.N.NN.NNNN)
        process_numbers = re.findall(r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}', text)
        for i, pnum in enumerate(process_numbers):
            placeholder = f"__PROCESS_{i}__"
            text = text.replace(pnum, placeholder)
        
        # Preserva valores monetários
        money_values = re.findall(r'r\$\s*\d+(?:\.\d{3})*(?:,\d{2})?', text)
        for i, money in enumerate(money_values):
            placeholder = f"__MONEY_{i}__"
            text = text.replace(money, placeholder)
        
        # Preserva datas
        dates = re.findall(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text)
        for i, date in enumerate(dates):
            placeholder = f"__DATE_{i}__"
            text = text.replace(date, placeholder)
        
        # Remove pontuação mas preserva estruturas importantes
        text = re.sub(r'[^\w\s\-_]', ' ', text)
        
        # Tokeniza
        tokens = []
        for token in text.split():
            token = token.strip('-_')
            
            # Restaura valores preservados
            if '__PROCESS_' in token:
                tokens.append('numero_processo')
            elif '__MONEY_' in token:
                tokens.append('valor_monetario')
            elif '__DATE_' in token:
                tokens.append('data_documento')
            elif (len(token) >= 3 and 
                  token not in self.legal_stopwords and
                  not token.isdigit()):
                tokens.append(token)
        
        self._token_cache[text_hash] = tokens
        return tokens
    
    def _extract_bigrams(self, tokens):
        """Extrai bigramas relevantes"""
        if len(tokens) < 2:
            return []
        
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            bigrams.append(bigram)
        return bigrams
    
    def _build_vocabulary(self, documents):
        """Constrói vocabulário otimizado"""
        print(f"🔄 Construindo vocabulário TF-IDF para {len(documents)} documentos...")
        
        all_tokens = []
        all_bigrams = []
        doc_token_sets = []
        doc_bigram_sets = []
        
        # Processa todos os documentos
        for doc in documents:
            tokens = self._smart_tokenize(doc)
            bigrams = self._extract_bigrams(tokens) if self.use_bigrams else []
            
            doc_token_sets.append(set(tokens))
            doc_bigram_sets.append(set(bigrams))
            all_tokens.extend(tokens)
            all_bigrams.extend(bigrams)
        
        # Conta frequências
        token_freq = Counter(all_tokens)
        bigram_freq = Counter(all_bigrams)
        
        # Seleciona tokens por critérios de qualidade
        vocab_candidates = []
        for token, freq in token_freq.items():
            if freq >= self.min_df:
                # Calcula document frequency
                doc_count = sum(1 for doc_set in doc_token_sets if token in doc_set)
                if doc_count > 0:
                    idf = math.log(len(documents) / doc_count)
                    # Score que privilegia termos discriminativos
                    quality_score = idf * math.log(1 + freq)
                    vocab_candidates.append((token, quality_score, idf))
        
        # Ordena e seleciona melhores
        vocab_candidates.sort(key=lambda x: x[1], reverse=True)
        
        vocabulary = {}
        idf_scores = {}
        
        max_tokens = self.max_features // (2 if self.use_bigrams else 1)
        for token, score, idf in vocab_candidates[:max_tokens]:
            vocabulary[token] = len(vocabulary)
            idf_scores[token] = idf
        
        # Processa bigramas
        bigram_vocabulary = {}
        bigram_idf_scores = {}
        
        if self.use_bigrams:
            bigram_candidates = []
            for bigram, freq in bigram_freq.items():
                if freq >= max(2, self.min_df):
                    doc_count = sum(1 for doc_set in doc_bigram_sets if bigram in doc_set)
                    if doc_count > 0:
                        idf = math.log(len(documents) / doc_count)
                        quality_score = idf * math.log(1 + freq)
                        bigram_candidates.append((bigram, quality_score, idf))
            
            bigram_candidates.sort(key=lambda x: x[1], reverse=True)
            
            max_bigrams = self.max_features // 2
            for bigram, score, idf in bigram_candidates[:max_bigrams]:
                bigram_vocabulary[bigram] = len(bigram_vocabulary)
                bigram_idf_scores[bigram] = idf
        
        print(f"✅ Vocabulário: {len(vocabulary)} tokens + {len(bigram_vocabulary)} bigramas")
        return vocabulary, idf_scores, bigram_vocabulary, bigram_idf_scores
    
    def fit(self, documents):
        """Treina o embedder"""
        self.vocabulary, self.idf_scores, self.bigram_vocabulary, self.bigram_idf_scores = self._build_vocabulary(documents)
        self.is_fitted = True
    
    def _create_vector(self, text):
        """Cria vetor TF-IDF para um texto"""
        if not self.is_fitted:
            return np.random.random(self.max_features).tolist()
        
        tokens = self._smart_tokenize(text)
        bigrams = self._extract_bigrams(tokens) if self.use_bigrams else []
        
        token_count = Counter(tokens)
        bigram_count = Counter(bigrams)
        
        # Tamanho total do vetor
        total_size = len(self.vocabulary) + len(self.bigram_vocabulary)
        vector = np.zeros(total_size)
        
        # Preenche parte dos tokens
        for token, count in token_count.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / len(tokens) if tokens else 0
                idf = self.idf_scores.get(token, 0)
                vector[idx] = tf * idf
        
        # Preenche parte dos bigramas
        if self.use_bigrams:
            offset = len(self.vocabulary)
            for bigram, count in bigram_count.items():
                if bigram in self.bigram_vocabulary:
                    idx = offset + self.bigram_vocabulary[bigram]
                    tf = count / len(bigrams) if bigrams else 0
                    idf = self.bigram_idf_scores.get(bigram, 0)
                    vector[idx] = tf * idf
        
        # Normalização
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def embed_query(self, text):
        return self._create_vector(text).tolist()
    
    def embed_documents(self, documents):
        return [self._create_vector(doc).tolist() for doc in documents]

# =================================================================================
# 🗃️ VECTOR STORE OTIMIZADO
# =================================================================================

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
        if use_ollama:
            try:
                self.ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
                print("✅ Usando Ollama embeddings")
            except Exception as e:
                print(f"⚠️ Ollama indisponível, usando TF-IDF: {e}")
                self.use_ollama = False
                self.tfidf_embedder = SmartTFIDFEmbedder()
        else:
            self.tfidf_embedder = SmartTFIDFEmbedder()
        
        os.makedirs(persist_dir, exist_ok=True)
    
    def _get_doc_hash(self, content):
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_documents(self, documents, max_docs=500):
        """Adiciona documentos com filtragem inteligente"""
        print(f"📊 Processando {len(documents)} documentos...")
        
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
            doc_hash = self._get_doc_hash(doc.page_content)
            
            if doc_hash not in cache:
                new_docs.append(doc)
                new_contents.append(doc.page_content)
                self.doc_hashes.append(doc_hash)
            else:
                # Carrega do cache
                cached = cache[doc_hash]
                self.embeddings.append(cached["embedding"])
                self.documents.append(doc.page_content)
                self.metadata.append(doc.metadata)
                self.doc_hashes.append(doc_hash)
        
        if new_contents:
            print(f"🚀 Gerando embeddings para {len(new_contents)} novos documentos...")
            
            if self.use_ollama:
                try:
                    # Usa Ollama em batches para eficiência
                    batch_size = 10
                    new_embeddings = []
                    for i in range(0, len(new_contents), batch_size):
                        batch = new_contents[i:i+batch_size]
                        batch_emb = self.ollama_embeddings.embed_documents(batch)
                        new_embeddings.extend(batch_emb)
                        print(f"   📦 Batch {i//batch_size + 1}/{(len(new_contents)-1)//batch_size + 1}")
                except Exception as e:
                    print(f"❌ Erro Ollama: {e}. Usando TF-IDF...")
                    self.use_ollama = False
                    self.tfidf_embedder = SmartTFIDFEmbedder()
                    all_docs = new_contents + [doc for doc in self.documents]
                    self.tfidf_embedder.fit(all_docs)
                    new_embeddings = self.tfidf_embedder.embed_documents(new_contents)
            else:
                # Usa TF-IDF
                all_docs = new_contents + [doc for doc in self.documents]
                self.tfidf_embedder.fit(all_docs)
                new_embeddings = self.tfidf_embedder.embed_documents(new_contents)
            
            # Adiciona os novos embeddings
            for doc, embedding in zip(new_docs, new_embeddings):
                self.embeddings.append(embedding)
                self.documents.append(doc.page_content)
                self.metadata.append(doc.metadata)
            
            print(f"✅ {len(new_contents)} embeddings gerados!")
        
        self._save_cache()
    
    def similarity_search(self, query, k=4, min_score=0.1):
        """Busca por similaridade otimizada"""
        if not self.embeddings:
            return []
        
        print(f"🔍 Buscando para: '{query[:50]}...'")
        
        # Gera embedding da query
        if self.use_ollama:
            try:
                query_embedding = self.ollama_embeddings.embed_query(query)
            except:
                query_embedding = self.tfidf_embedder.embed_query(query)
        else:
            query_embedding = self.tfidf_embedder.embed_query(query)
        
        query_vec = np.array(query_embedding)
        
        # Calcula similaridades
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            doc_vec = np.array(doc_embedding)
            
            # Similaridade coseno
            sim = np.dot(query_vec, doc_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-8
            )
            
            # Boost para matches de palavras-chave
            query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
            doc_words = set(re.findall(r'\b\w{3,}\b', self.documents[i].lower()[:500]))  # Primeiros 500 chars
            
            if query_words and doc_words:
                word_overlap = len(query_words.intersection(doc_words)) / len(query_words)
                sim += word_overlap * 0.2  # Boost de até 20%
            
            similarities.append((i, float(sim)))
        
        # Ordena e filtra
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = [(i, score) for i, score in similarities if score >= min_score]
        
        # Retorna top-k
        selected = similarities[:k]
        print(f"📊 Scores: {[f'{s:.3f}' for _, s in selected]}")
        
        return [
            Document(
                page_content=self.documents[i], 
                metadata={**self.metadata[i], "similarity_score": score}
            ) 
            for i, score in selected
        ]
    
    def _load_cache(self):
        cache_file = os.path.join(self.persist_dir, "smart_cache.pkl")
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
        
        cache_file = os.path.join(self.persist_dir, "smart_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Salva embedder TF-IDF se usado
        if not self.use_ollama:
            embedder_file = os.path.join(self.persist_dir, "tfidf_embedder.pkl")
            with open(embedder_file, 'wb') as f:
                pickle.dump(self.tfidf_embedder, f)
    
    def load(self):
        """Carrega dados salvos"""
        cache = self._load_cache()
        
        for doc_hash, data in cache.items():
            self.embeddings.append(data["embedding"])
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
            except:
                pass
        
        return len(self.embeddings) > 0
    
    def as_retriever(self, search_kwargs=None):
        """Retriever compatível com LangChain"""
        k = search_kwargs.get("k", 4) if search_kwargs else 4
        min_score = search_kwargs.get("min_score", 0.1) if search_kwargs else 0.1
        
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

# =================================================================================
# 🎯 SISTEMA RAG PRINCIPAL
# =================================================================================

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

class UltraFastRAG:
    def __init__(self, config: Optional[UltraFastRAGConfig] = None):
        self.config = config or UltraFastRAGConfig()
        self.llm = None
        self.vector_store = None
        self.documents = []
        self.is_initialized = False
        
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
            
            # Inicializa LLM
            self.llm = OllamaLLM(
                model=self.config.model_name,
                temperature=self.config.temperature,
                num_predict=400
            )
            
            # Testa conexão
            test_response = self.llm.invoke("Teste")
            print(f"✅ LLM {self.config.model_name} conectado")
            
            self.is_initialized = True
            self._load_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            print("💡 Verifique se o Ollama está rodando e os modelos instalados")
            return False
    
    def _load_cache(self):
        """Carrega cache existente"""
        self.vector_store = OptimizedVectorStore(
            os.path.join(self.cache_path, "optimized"),
            use_ollama=self.config.use_ollama_embeddings
        )
        
        if self.vector_store.load():
            self.documents = [
                Document(page_content=doc, metadata=meta) 
                for doc, meta in zip(self.vector_store.documents, self.vector_store.metadata)
            ]
            print(f"📦 Cache carregado: {len(self.documents)} documentos")
    
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
                        
                        # Cria chunks do documento
                        chunks = self.text_splitter.split_text(clean_content)
                        for chunk in chunks:
                            if len(chunk.strip()) > 100:  # Chunks muito pequenos
                                documents.append(Document(
                                    page_content=chunk.strip(),
                                    metadata=metadata.copy()
                                ))
                        
                    except Exception as e:
                        print(f"⚠️ Erro ao processar {file}: {e}")
        
        if documents:
            print(f"📄 {len(documents)} chunks carregados")
            self.documents = documents
            self._create_vector_store()
        else:
            print("⚠️ Nenhum documento válido encontrado")
        
        return len(documents)
    
    def _create_vector_store(self):
        """Cria vector store com os documentos"""
        if not self.documents:
            return
        
        print("🗃️ Criando vector store...")
        
        self.vector_store = OptimizedVectorStore(
            os.path.join(self.cache_path, "optimized"),
            use_ollama=self.config.use_ollama_embeddings
        )
        
        self.vector_store.add_documents(self.documents, max_docs=self.config.max_chunks)
        print("✅ Vector store criado!")
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Realiza consulta no sistema RAG"""
        if not self.is_initialized or not self.vector_store:
            return {"error": "Sistema não inicializado"}
        
        try:
            k = top_k or self.config.top_k
            print(f"🔍 Processando: {question[:50]}...")
            
            # Busca documentos relevantes
            relevant_docs = self.vector_store.similarity_search(question, k=k, min_score=0.15)
            
            if not relevant_docs:
                return {
                    "error": "Nenhum documento relevante encontrado",
                    "suggestion": "Tente reformular a pergunta com termos mais específicos"
                }
            
            # Prepara contexto
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.page_content.strip()
                score = doc.metadata.get("similarity_score", 0)
                
                # Adiciona metadados relevantes se existirem
                metadata_info = []
                for key in ['numero_processo', 'agravante', 'agravado', 'assuntos']:
                    if key in doc.metadata and doc.metadata[key]:
                        metadata_info.append(f"{key}: {doc.metadata[key]}")
                
                doc_text = f"DOCUMENTO {i} (relevância: {score:.2f}):\n"
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
3. Se a informação não estiver nos documentos, responda "Informação não encontrada"
4. Seja preciso e objetivo
5. Mantenha linguagem jurídica adequada

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
                    "metadata": {k: v for k, v in doc.metadata.items() if k in ['filename', 'numero_processo', 'agravante', 'agravado']},
                    "similarity_score": doc.metadata.get("similarity_score", 0)
                } for doc in relevant_docs]
            }
            
        except Exception as e:
            print(f"❌ Erro na consulta: {e}")
            return {"error": str(e)}

# =================================================================================
# 🚀 INSTÂNCIA GLOBAL E FUNÇÕES DE INTERFACE
# =================================================================================

# Instância global para compatibilidade
rag_system = UltraFastRAG()

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
            "data_path": rag_system.data_path
        }
    
    if not rag_system.vector_store or len(rag_system.documents) == 0:
        return {
            "status": "offline", 
            "message": "Nenhum documento carregado", 
            "isReady": False,
            "data_path": rag_system.data_path
        }
    
    embedding_method = "Híbrido (Ollama + TF-IDF)" if rag_system.config.use_ollama_embeddings else "TF-IDF"
    
    return {
        "status": "online", 
        "message": f"{len(rag_system.documents)} documentos carregados", 
        "isReady": True,
        "documents_loaded": len(rag_system.documents),
        "embedding_method": embedding_method,
        "model": rag_system.config.model_name,
        "data_path": rag_system.data_path
    }

def query_rag(question: str, top_k: int = 4):
    """Interface para consultas RAG"""
    return rag_system.query(question, top_k)

# =================================================================================
# 🧪 TESTE STANDALONE
# =================================================================================

if __name__ == "__main__":
    print("🧪 TESTE DO SISTEMA RAG")
    print("=" * 50)
    
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
    
    # Configura sistema
    config = UltraFastRAGConfig(
        model_name="gemma:2b",
        temperature=0.1,
        data_dir=test_dir,
        use_ollama_embeddings=True
    )
    
    test_rag = UltraFastRAG(config)
    
    print("\n🔄 Inicializando sistema...")
    if test_rag.initialize():
        print("✅ Sistema inicializado")
        
        print("\n📂 Carregando documentos...")
        docs_loaded = test_rag.load_documents_from_directory()
        print(f"✅ {docs_loaded} documentos carregados")
        
        if docs_loaded > 0:
            print("\n🧪 Executando testes...")
            
            queries = [
                "Qual é o número do processo e quem são as partes envolvidas?",
                "A terapia ABA é disponibilizada pelo SUS?",
                "Quais tratamentos o agravante necessita?"
            ]
            
            for i, query in enumerate(queries, 1):
                print(f"\n--- TESTE {i} ---")
                print(f"Pergunta: {query}")
                
                result = test_rag.query(query)
                
                if "error" in result:
                    print(f"❌ Erro: {result['error']}")
                else:
                    print(f"Resposta: {result['answer']}")
                    print(f"Fontes: {result['sources']}")
                    print(f"Documentos: {result['documents_found']}")
        else:
            print("⚠️ Nenhum documento carregado para teste")
    else:
        print("❌ Falha na inicialização")
        print("💡 Certifique-se que o Ollama está rodando:")
        print("   ollama serve")
        print("   ollama pull gemma:2b")
        print("   ollama pull nomic-embed-text")