# utils/smart_tfidf_embedder.py

import hashlib
import math
import re
from collections import Counter
from typing import List

import numpy as np


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
        
        # Cache para performance (limitado para evitar problemas)
        self._token_cache = {}
        self._cache_limit = 1000  # Limita o cache
    
    def _smart_tokenize(self, text):
        """Tokenização inteligente para documentos legais"""
        if not text or not isinstance(text, str):
            return []
        
        # Limita o tamanho do cache para evitar problemas de memória
        if len(self._token_cache) > self._cache_limit:
            self._token_cache.clear()
        
        # Cache para evitar reprocessamento
        try:
            text_hash = hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:8]
            if text_hash in self._token_cache:
                return self._token_cache[text_hash]
        except Exception:
            # Se der erro no hash, processa sem cache
            pass
        
        try:
            # Normalização preservando estruturas legais importantes
            text = str(text).lower()
            
            # Preserva números de processo (formato: NNNNNN-NN.NNNN.N.NN.NNNN)
            text = re.sub(r'\d{6,7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}', 'NUMERO_PROCESSO', text)
            
            # Preserva valores monetários
            text = re.sub(r'r\$\s*\d+(?:\.\d{3})*(?:,\d{2})?', 'VALOR_MONETARIO', text)
            
            # Preserva datas
            text = re.sub(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', 'DATA_DOCUMENTO', text)
            
            # Remove pontuação mas preserva estruturas importantes
            text = re.sub(r'[^\w\s\-_]', ' ', text)
            
            # Tokeniza
            tokens = []
            for token in text.split():
                token = token.strip('-_')
                
                if (len(token) >= 3 and 
                    token not in self.legal_stopwords and
                    not token.isdigit()):
                    tokens.append(token)
            
            # Salva no cache se possível
            try:
                if text_hash and len(self._token_cache) < self._cache_limit:
                    self._token_cache[text_hash] = tokens
            except Exception:
                pass
            
            return tokens
            
        except Exception as e:
            print(f"⚠️ Erro na tokenização: {e}")
            return []
    
    def _extract_bigrams(self, tokens):
        """Extrai bigramas relevantes"""
        if not tokens or len(tokens) < 2:
            return []
        
        try:
            bigrams = []
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                bigrams.append(bigram)
            return bigrams
        except Exception as e:
            print(f"⚠️ Erro na extração de bigramas: {e}")
            return []
    
    def _build_vocabulary(self, documents):
        """Constrói vocabulário otimizado"""
        print(f"🔄 Construindo vocabulário TF-IDF para {len(documents)} documentos...")
        
        if not documents:
            print("⚠️ Nenhum documento fornecido para construir vocabulário")
            return {}, {}, {}, {}
        
        try:
            all_tokens = []
            all_bigrams = []
            doc_token_sets = []
            doc_bigram_sets = []
            
            # Processa documentos em batches para evitar problemas de memória
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                print(f"   📦 Processando batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                for doc in batch:
                    try:
                        tokens = self._smart_tokenize(doc)
                        bigrams = self._extract_bigrams(tokens) if self.use_bigrams else []
                        
                        doc_token_sets.append(set(tokens))
                        doc_bigram_sets.append(set(bigrams))
                        all_tokens.extend(tokens)
                        all_bigrams.extend(bigrams)
                    except Exception as e:
                        print(f"⚠️ Erro ao processar documento: {e}")
                        continue
            
            if not all_tokens:
                print("⚠️ Nenhum token encontrado nos documentos")
                return {}, {}, {}, {}
            
            # Conta frequências
            token_freq = Counter(all_tokens)
            bigram_freq = Counter(all_bigrams)
            
            # Seleciona tokens por critérios de qualidade
            vocab_candidates = []
            for token, freq in token_freq.items():
                try:
                    if freq >= self.min_df:
                        # Calcula document frequency
                        doc_count = sum(1 for doc_set in doc_token_sets if token in doc_set)
                        if doc_count > 0:
                            idf = math.log(len(documents) / doc_count)
                            # Score que privilegia termos discriminativos
                            quality_score = idf * math.log(1 + freq)
                            vocab_candidates.append((token, quality_score, idf))
                except Exception as e:
                    print(f"⚠️ Erro ao processar token {token}: {e}")
                    continue
            
            # Ordena e seleciona melhores
            vocab_candidates.sort(key=lambda x: x[1], reverse=True)
            
            vocabulary = {}
            idf_scores = {}
            
            max_tokens = max(1, self.max_features // (2 if self.use_bigrams else 1))
            for token, score, idf in vocab_candidates[:max_tokens]:
                vocabulary[token] = len(vocabulary)
                idf_scores[token] = idf
            
            # Processa bigramas
            bigram_vocabulary = {}
            bigram_idf_scores = {}
            
            if self.use_bigrams and bigram_freq:
                bigram_candidates = []
                for bigram, freq in bigram_freq.items():
                    try:
                        if freq >= max(2, self.min_df):
                            doc_count = sum(1 for doc_set in doc_bigram_sets if bigram in doc_set)
                            if doc_count > 0:
                                idf = math.log(len(documents) / doc_count)
                                quality_score = idf * math.log(1 + freq)
                                bigram_candidates.append((bigram, quality_score, idf))
                    except Exception as e:
                        print(f"⚠️ Erro ao processar bigrama {bigram}: {e}")
                        continue
                
                bigram_candidates.sort(key=lambda x: x[1], reverse=True)
                
                max_bigrams = max(1, self.max_features // 2)
                for bigram, score, idf in bigram_candidates[:max_bigrams]:
                    bigram_vocabulary[bigram] = len(bigram_vocabulary)
                    bigram_idf_scores[bigram] = idf
            
            print(f"✅ Vocabulário: {len(vocabulary)} tokens + {len(bigram_vocabulary)} bigramas")
            return vocabulary, idf_scores, bigram_vocabulary, bigram_idf_scores
            
        except Exception as e:
            print(f"❌ Erro crítico na construção do vocabulário: {e}")
            return {}, {}, {}, {}
    
    def fit(self, documents):
        """Treina o embedder"""
        try:
            if not documents:
                print("⚠️ Lista de documentos vazia para treinar embedder")
                self.is_fitted = False
                return
            
            # Filtra documentos válidos
            valid_docs = []
            for doc in documents:
                if doc and isinstance(doc, str) and len(doc.strip()) > 0:
                    valid_docs.append(doc.strip())
            
            if not valid_docs:
                print("⚠️ Nenhum documento válido encontrado")
                self.is_fitted = False
                return
            
            print(f"📚 Treinando com {len(valid_docs)} documentos válidos")
            
            result = self._build_vocabulary(valid_docs)
            self.vocabulary, self.idf_scores, self.bigram_vocabulary, self.bigram_idf_scores = result
            
            # Verifica se o vocabulário foi criado com sucesso
            if self.vocabulary or self.bigram_vocabulary:
                self.is_fitted = True
                print("✅ Embedder treinado com sucesso")
            else:
                self.is_fitted = False
                print("⚠️ Falha ao criar vocabulário - embedder não foi treinado")
                
        except Exception as e:
            print(f"❌ Erro ao treinar embedder: {e}")
            self.is_fitted = False
    
    def _create_vector(self, text):
        """Cria vetor TF-IDF para um texto"""
        try:
            if not self.is_fitted:
                print("⚠️ Embedder não foi treinado - retornando vetor aleatório")
                return np.random.random(min(self.max_features, 100))
            
            tokens = self._smart_tokenize(text)
            bigrams = self._extract_bigrams(tokens) if self.use_bigrams else []
            
            # Calcula tamanho do vetor
            total_size = len(self.vocabulary) + len(self.bigram_vocabulary)
            if total_size == 0:
                print("⚠️ Vocabulário vazio - retornando vetor aleatório")
                return np.random.random(min(self.max_features, 100))
            
            vector = np.zeros(total_size)
            
            # Conta tokens
            token_count = Counter(tokens)
            bigram_count = Counter(bigrams)
            
            # Preenche parte dos tokens
            for token, count in token_count.items():
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    if idx < len(vector):  # Verificação de segurança
                        tf = count / max(1, len(tokens))  # Evita divisão por zero
                        idf = self.idf_scores.get(token, 0)
                        vector[idx] = tf * idf
            
            # Preenche parte dos bigramas
            if self.use_bigrams and self.bigram_vocabulary:
                offset = len(self.vocabulary)
                for bigram, count in bigram_count.items():
                    if bigram in self.bigram_vocabulary:
                        idx = offset + self.bigram_vocabulary[bigram]
                        if idx < len(vector):  # Verificação de segurança
                            tf = count / max(1, len(bigrams))  # Evita divisão por zero
                            idf = self.bigram_idf_scores.get(bigram, 0)
                            vector[idx] = tf * idf
            
            # Normalização segura
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
            
        except Exception as e:
            print(f"⚠️ Erro ao criar vetor: {e}")
            return np.random.random(min(self.max_features, 100))
    
    def embed_query(self, text):
        """Embeda uma query"""
        try:
            vector = self._create_vector(text)
            if isinstance(vector, np.ndarray):
                return vector.tolist()
            return list(vector) if vector is not None else []
        except Exception as e:
            print(f"⚠️ Erro ao embedar query: {e}")
            return [0.0] * min(self.max_features, 100)
    
    def embed_documents(self, documents):
        """Embeda uma lista de documentos"""
        try:
            if not documents:
                return []
            
            results = []
            for i, doc in enumerate(documents):
                try:
                    vector = self._create_vector(doc)
                    if isinstance(vector, np.ndarray):
                        results.append(vector.tolist())
                    else:
                        results.append(list(vector) if vector is not None else [])
                except Exception as e:
                    print(f"⚠️ Erro ao embedar documento {i}: {e}")
                    results.append([0.0] * min(self.max_features, 100))
            
            return results
            
        except Exception as e:
            print(f"❌ Erro ao embedar documentos: {e}")
            return []