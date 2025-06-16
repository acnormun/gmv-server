#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adaptive_rag.py - Sistema RAG Adaptativo para GMV
Utilitário modular para integração com app.py
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging
from datetime import datetime

# ==========================================
# 🔧 CONFIGURAÇÃO E ESTRUTURAS
# ==========================================

class QueryType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical" 
    OPINION = "opinion"
    CONTEXTUAL = "contextual"

@dataclass
class ProcessDocument:
    """Representa um documento de processo anonimizado"""
    numero_processo: str
    tema: str
    data_distribuicao: str
    responsavel: str
    status: str
    ultima_atualizacao: str
    suspeitos: List[str]
    comentarios: str
    markdown_content: str

@dataclass
class RAGResult:
    """Resultado da consulta RAG"""
    query: str
    response: str
    retrieved_chunks: List[Dict]
    confidence_score: float
    strategy_used: str
    processing_time: float
    
    def to_dict(self):
        """Converte para dicionário para JSON"""
        return {
            'query': self.query,
            'response': self.response,
            'retrieved_chunks': self.retrieved_chunks,
            'confidence_score': self.confidence_score,
            'strategy_used': self.strategy_used,
            'processing_time': self.processing_time
        }

# ==========================================
# 🧠 SISTEMA DE EMBEDDINGS OFFLINE
# ==========================================

class AdvancedTfidfEmbeddings:
    """Sistema avançado de embeddings usando TF-IDF + features customizadas"""
    
    def __init__(self, max_features: int = 3000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words=None,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.9,
            lowercase=True,
            strip_accents='unicode',
            sublinear_tf=True
        )
        self.is_fitted = False
        
        # Palavras-chave jurídicas com pesos
        self.domain_keywords = {
            'lavagem': 15.0, 'dinheiro': 12.0, 'fraude': 14.0, 'corrupcao': 14.0,
            'suspeito': 10.0, 'investigacao': 8.0, 'processo': 6.0, 'crime': 12.0,
            'financeiro': 8.0, 'bancario': 8.0, 'movimentacao': 7.0, 'operacao': 7.0,
            'empresa': 5.0, 'pessoa': 4.0, 'documento': 6.0, 'evidencia': 10.0,
            'prova': 10.0, 'analise': 6.0, 'relatorio': 5.0, 'conclusao': 8.0,
            'apreensao': 9.0, 'bloqueio': 8.0, 'indisponibilidade': 9.0,
            'organizacao': 7.0, 'quadrilha': 9.0, 'esquema': 8.0,
            'falsificacao': 11.0, 'adulteracao': 10.0, 'propina': 12.0
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Pré-processamento do texto"""
        if not text:
            return ""
        
        text = re.sub(r'[^\w\s áéíóúâêîôûàèìòùãõçÁÉÍÓÚÂÊÎÔÛÀÈÌÒÙÃÕÇ]', ' ', text.lower())
        text = re.sub(r'\d{4}\.\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4}', 'NUMERO_PROCESSO', text)
        text = re.sub(r'r\$\s*[\d.,]+', 'VALOR_MONETARIO', text)
        text = re.sub(r'\d{1,2}\/\d{1,2}\/\d{4}', 'DATA', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def fit(self, texts: List[str]):
        """Treina o modelo com os textos"""
        if not texts:
            raise ValueError("Lista de textos vazia")
        
        processed_texts = [self._preprocess_text(text) for text in texts]
        processed_texts = [t for t in processed_texts if t.strip()]
        
        if not processed_texts:
            raise ValueError("Todos os textos ficaram vazios após processamento")
        
        self.tfidf.fit(processed_texts)
        self.is_fitted = True
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Gera embeddings para os textos"""
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute .fit() primeiro.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self._preprocess_text(text) for text in texts]
        tfidf_embeddings = self.tfidf.transform(processed_texts).toarray()
        domain_features = self._extract_domain_features(texts)
        
        combined_embeddings = np.hstack([
            tfidf_embeddings,
            domain_features * 0.2
        ])
        
        return combined_embeddings
    
    def _extract_domain_features(self, texts: List[str]) -> np.ndarray:
        """Extrai features específicas do domínio jurídico"""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            feat = []
            
            feat.append(len(text))
            feat.append(len(text.split()))
            feat.append(len(re.findall(r'\d', text)) / max(len(text), 1))
            
            for keyword, weight in self.domain_keywords.items():
                count = len(re.findall(r'\b' + keyword, text_lower))
                feat.append(count * weight)
            
            feat.append(1 if re.search(r'\d{4}\.\d{2}\.\d{4}', text) else 0)
            feat.append(1 if re.search(r'r\$|real|reais', text_lower) else 0)
            feat.append(1 if re.search(r'\d{1,2}\/\d{1,2}\/\d{4}', text) else 0)
            feat.append(len(re.findall(r'\b[A-Z]{2,}\b', text)))
            
            features.append(feat)
        
        features_array = np.array(features, dtype=float)
        
        for i in range(features_array.shape[1]):
            col = features_array[:, i]
            if col.std() > 0:
                features_array[:, i] = (col - col.mean()) / col.std()
        
        return features_array

# ==========================================
# 🤖 SISTEMA RAG ADAPTATIVO
# ==========================================

class GMVAdaptiveRAG:
    """Sistema RAG adaptativo para dados do GMV"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Sistema de embeddings offline
        self.embedding_model = AdvancedTfidfEmbeddings()
        
        # Storage
        self.documents: List[ProcessDocument] = []
        self.chunk_embeddings: np.ndarray = None
        self.chunk_metadata: List[Dict] = []
        self.all_chunks: List[str] = []
        
        # Configuração de logging
        self.logger = logging.getLogger(__name__)
        
        # Cache
        self.query_cache: Dict[str, RAGResult] = {}
        self.is_initialized = False
    
    def initialize(self, triagem_path: str, pasta_destino: str, pasta_dat: str = None) -> bool:
        """Inicializa o sistema carregando dados"""
        try:
            self.logger.info("🔄 Inicializando sistema RAG...")
            
            num_docs = self.load_gmv_data(triagem_path, pasta_destino, pasta_dat or "")
            
            if num_docs > 0:
                self.is_initialized = True
                self.logger.info(f"✅ Sistema RAG inicializado com {num_docs} documentos")
                return True
            else:
                self.logger.error("❌ Falha na inicialização - nenhum documento carregado")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erro na inicialização: {str(e)}")
            return False
    
    def load_gmv_data(self, triagem_path: str, pasta_destino: str, pasta_dat: str) -> int:
        """Carrega dados do sistema GMV"""
        try:
            processos = self._extrair_tabela_triagem(triagem_path)
            
            if not processos:
                self.logger.warning("⚠️ Nenhum processo encontrado na tabela de triagem")
                return 0
            
            for processo in processos:
                numero = processo['numeroProcesso'].replace('/', '-')
                markdown_path = os.path.join(pasta_destino, f"{numero}.md")
                
                if os.path.exists(markdown_path):
                    with open(markdown_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                    
                    doc = ProcessDocument(
                        numero_processo=processo['numeroProcesso'],
                        tema=processo['tema'],
                        data_distribuicao=processo['dataDistribuicao'],
                        responsavel=processo['responsavel'],
                        status=processo['status'],
                        ultima_atualizacao=processo['ultimaAtualizacao'],
                        suspeitos=processo['suspeitos'].split(',') if processo['suspeitos'] else [],
                        comentarios=processo.get('comentarios', ''),
                        markdown_content=markdown_content
                    )
                    
                    self.documents.append(doc)
            
            if self.documents:
                self._process_documents()
                return len(self.documents)
            else:
                return 0
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {str(e)}")
            return 0
    
    def _extrair_tabela_triagem(self, arquivo_md: str) -> List[Dict]:
        """Extrai dados da tabela de triagem markdown"""
        processos = []
        
        try:
            with open(arquivo_md, 'r', encoding='utf-8') as f:
                conteudo = f.read()
            
            linhas = conteudo.split('\n')
            
            for linha in linhas:
                linha = linha.strip()
                if linha.startswith('|') and not linha.startswith('|---') and 'Nº Processo' not in linha:
                    colunas = [col.strip() for col in linha.split('|')[1:-1]]
                    
                    if len(colunas) >= 7:
                        processo = {
                            'numeroProcesso': colunas[0],
                            'tema': colunas[1],
                            'dataDistribuicao': colunas[2],
                            'responsavel': colunas[3],
                            'status': colunas[4],
                            'ultimaAtualizacao': colunas[5],
                            'suspeitos': colunas[6],
                            'comentarios': colunas[7] if len(colunas) > 7 else ''
                        }
                        processos.append(processo)
                        
        except Exception as e:
            self.logger.error(f"Erro ao extrair tabela: {str(e)}")
            
        return processos
    
    def _process_documents(self):
        """Processa documentos em chunks e gera embeddings"""
        self.logger.info("🔄 Processando documentos em chunks...")
        
        all_chunks = []
        all_metadata = []
        
        for doc in self.documents:
            full_text = f"""
            PROCESSO: {doc.numero_processo}
            TEMA: {doc.tema}
            DATA: {doc.data_distribuicao}
            RESPONSAVEL: {doc.responsavel}
            STATUS: {doc.status}
            SUSPEITOS: {', '.join(doc.suspeitos) if doc.suspeitos else 'Nenhum'}
            COMENTARIOS: {doc.comentarios}
            
            CONTEUDO:
            {doc.markdown_content}
            """
            
            chunks = self._create_chunks_with_overlap(full_text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'document_id': doc.numero_processo,
                    'chunk_id': i,
                    'tema': doc.tema,
                    'status': doc.status,
                    'suspeitos': doc.suspeitos,
                    'data_distribuicao': doc.data_distribuicao
                })
        
        if all_chunks:
            self.all_chunks = all_chunks
            self.chunk_metadata = all_metadata
            
            # Treina e gera embeddings
            self.embedding_model.fit(all_chunks)
            self.chunk_embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
            
            self.logger.info(f"✅ {len(all_chunks)} chunks processados")
    
    def _create_chunks_with_overlap(self, text: str) -> List[str]:
        """Cria chunks com overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classifica o tipo de consulta"""
        query_lower = query.lower()
        
        factual_patterns = [
            r'\b(qual|quais|quando|onde|quem|quantos|numero|data|nome)\b',
            r'\b(processo.*numero|numero.*processo)\b',
            r'\b(responsavel|status|tema)\b'
        ]
        
        analytical_patterns = [
            r'\b(compare|analise|diferenca|relacao|tendencia|padrao)\b',
            r'\b(maior|menor|melhor|pior|mais|menos)\b',
            r'\b(como.*difere|porque.*diferente)\b'
        ]
        
        contextual_patterns = [
            r'\b(contexto|situacao|cenario|ambiente|circunstancia)\b',
            r'\b(explique.*porque|razao.*por)\b',
            r'\b(considerando|levando.*conta)\b'
        ]
        
        opinion_patterns = [
            r'\b(opini[aã]o|acho|acredito|parece|talvez|provavel)\b',
            r'\b(tendencia|futuro|prever|expectativa)\b'
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                return QueryType.FACTUAL
                
        for pattern in analytical_patterns:
            if re.search(pattern, query_lower):
                return QueryType.ANALYTICAL
                
        for pattern in contextual_patterns:
            if re.search(pattern, query_lower):
                return QueryType.CONTEXTUAL
                
        for pattern in opinion_patterns:
            if re.search(pattern, query_lower):
                return QueryType.OPINION
        
        return QueryType.FACTUAL
    
    def _retrieval_strategy(self, query: str, query_type: QueryType, k: int = 5) -> List[Dict]:
        """Executa estratégia de recuperação baseada no tipo de consulta"""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        if query_type == QueryType.ANALYTICAL:
            # Para consultas analíticas, busca mais diversidade
            top_indices = np.argsort(similarities)[-k*2:][::-1]
            
            if len(top_indices) > k:
                top_embeddings = self.chunk_embeddings[top_indices]
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(top_embeddings)
                
                final_indices = []
                for cluster_id in range(k):
                    cluster_indices = top_indices[clusters == cluster_id]
                    if len(cluster_indices) > 0:
                        best_in_cluster = cluster_indices[np.argmax(similarities[cluster_indices])]
                        final_indices.append(best_in_cluster)
            else:
                final_indices = top_indices
                
        elif query_type == QueryType.CONTEXTUAL:
            # Para consultas contextuais, aplica boost baseado em metadados
            boosted_similarities = similarities.copy()
            
            for i, metadata in enumerate(self.chunk_metadata):
                if metadata['suspeitos']:
                    boosted_similarities[i] *= 1.2
                if metadata['status'].lower() in ['suspeito', 'investigação']:
                    boosted_similarities[i] *= 1.1
            
            final_indices = np.argsort(boosted_similarities)[-k:][::-1]
            
        elif query_type == QueryType.OPINION:
            # Para consultas de opinião, prioriza comentários
            weighted_similarities = similarities.copy()
            
            for i, metadata in enumerate(self.chunk_metadata):
                doc = next((d for d in self.documents if d.numero_processo == metadata['document_id']), None)
                if doc and doc.comentarios.strip():
                    weighted_similarities[i] *= 1.3
            
            final_indices = np.argsort(weighted_similarities)[-k:][::-1]
            
        else:  # FACTUAL
            # Para consultas factuais, busca direta por similaridade
            final_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in final_indices:
            results.append({
                'chunk_id': idx,
                'content': self.all_chunks[idx],
                'similarity': similarities[idx],
                'metadata': self.chunk_metadata[idx]
            })
            
        return sorted(results, key=lambda x: x['similarity'], reverse=True)
    
    def _generate_response(self, query: str, retrieved_chunks: List[Dict], strategy: str) -> str:
        """Gera resposta baseada nos chunks recuperados"""
        
        if not retrieved_chunks:
            return "Desculpe, não encontrei informações relevantes para sua consulta."
        
        if strategy == QueryType.FACTUAL.value:
            response = f"Com base nos dados dos processos:\n\n"
            
            for chunk in retrieved_chunks[:3]:
                metadata = chunk['metadata']
                response += f"• **Processo {metadata['document_id']}**\n"
                response += f"  - Tema: {metadata['tema']}\n"
                response += f"  - Status: {metadata['status']}\n"
                response += f"  - Confiança: {chunk['similarity']:.3f}\n\n"
                
        elif strategy == QueryType.ANALYTICAL.value:
            response = f"Análise dos dados encontrados:\n\n"
            
            temas = {}
            for chunk in retrieved_chunks:
                tema = chunk['metadata']['tema']
                if tema not in temas:
                    temas[tema] = []
                temas[tema].append(chunk)
            
            for tema, chunks in temas.items():
                response += f"**{tema}:** {len(chunks)} processo(s) relacionado(s)\n"
                
        elif strategy == QueryType.CONTEXTUAL.value:
            response = f"Considerando o contexto dos processos:\n\n"
            
            suspeitos_encontrados = set()
            for chunk in retrieved_chunks:
                suspeitos_encontrados.update(chunk['metadata']['suspeitos'])
            
            if suspeitos_encontrados:
                response += f"Suspeitos identificados: {', '.join(list(suspeitos_encontrados)[:5])}\n\n"
                
        else:  # OPINION
            response = f"Com base nos padrões observados:\n\n"
            
            status_count = {}
            for chunk in retrieved_chunks:
                status = chunk['metadata']['status']
                status_count[status] = status_count.get(status, 0) + 1
            
            response += "Distribuição por status:\n"
            for status, count in status_count.items():
                response += f"• {status}: {count} processo(s)\n"
        
        response += f"\n---\n*Baseado em {len(retrieved_chunks)} chunk(s) de dados anonimizados*"
        
        return response
    
    def query(self, query_text: str, k: int = 5, use_cache: bool = True) -> RAGResult:
        """Executa consulta usando Adaptive RAG"""
        if not self.is_initialized:
            raise ValueError("Sistema RAG não foi inicializado. Execute initialize() primeiro.")
        
        start_time = datetime.now()
        
        # Verifica cache
        if use_cache and query_text in self.query_cache:
            self.logger.info("🔄 Usando resultado do cache")
            return self.query_cache[query_text]
        
        # Classifica tipo de consulta
        query_type = self._classify_query_type(query_text)
        self.logger.info(f"🔍 Tipo de consulta detectado: {query_type.value}")
        
        # Executa estratégia de recuperação
        retrieved_chunks = self._retrieval_strategy(query_text, query_type, k)
        
        # Gera resposta
        response = self._generate_response(query_text, retrieved_chunks, query_type.value)
        
        # Calcula confiança
        if retrieved_chunks:
            avg_similarity = np.mean([chunk['similarity'] for chunk in retrieved_chunks])
            confidence = min(avg_similarity * 1.2, 1.0)
        else:
            confidence = 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Cria resultado
        result = RAGResult(
            query=query_text,
            response=response,
            retrieved_chunks=retrieved_chunks,
            confidence_score=confidence,
            strategy_used=query_type.value,
            processing_time=processing_time
        )
        
        # Salva no cache
        if use_cache:
            self.query_cache[query_text] = result
        
        self.logger.info(f"✅ Consulta processada em {processing_time:.2f}s com confiança {confidence:.3f}")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do sistema"""
        if not self.documents:
            return {"error": "Nenhum documento carregado"}
        
        status_count = {}
        tema_count = {}
        suspeitos_count = {}
        
        for doc in self.documents:
            status_count[doc.status] = status_count.get(doc.status, 0) + 1
            tema_count[doc.tema] = tema_count.get(doc.tema, 0) + 1
            
            for suspeito in doc.suspeitos:
                if suspeito.strip():
                    suspeitos_count[suspeito] = suspeitos_count.get(suspeito, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunk_metadata) if self.chunk_metadata else 0,
            "cache_size": len(self.query_cache),
            "is_initialized": self.is_initialized,
            "status_distribution": status_count,
            "tema_distribution": tema_count,
            "top_suspeitos": dict(sorted(suspeitos_count.items(), key=lambda x: x[1], reverse=True)[:10])
        }

# ==========================================
# 🎯 INSTÂNCIA GLOBAL DO RAG
# ==========================================

# Instância global para uso na aplicação Flask
rag_instance = GMVAdaptiveRAG()

def initialize_rag(triagem_path: str, pasta_destino: str, pasta_dat: str = None) -> bool:
    """Inicializa a instância global do RAG"""
    return rag_instance.initialize(triagem_path, pasta_destino, pasta_dat)

def query_rag(query: str, k: int = 5) -> Dict:
    """Executa consulta na instância global do RAG"""
    try:
        result = rag_instance.query(query, k=k)
        return result.to_dict()
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "response": "Erro ao processar consulta",
            "confidence_score": 0.0,
            "processing_time": 0.0
        }

def get_rag_statistics() -> Dict:
    """Retorna estatísticas da instância global do RAG"""
    return rag_instance.get_statistics()