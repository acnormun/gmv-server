import os
import pandas as pd
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging
import requests
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
    llm_model: str
    
    def to_dict(self):
        """Converte para dicionário para JSON"""
        return {
            'query': self.query,
            'response': self.response,
            'retrieved_chunks': self.retrieved_chunks,
            'confidence_score': self.confidence_score,
            'strategy_used': self.strategy_used,
            'processing_time': self.processing_time,
            'llm_model': self.llm_model
        }

# ==========================================
# 🦙 CLIENTE LLAMA 3.1:8B OTIMIZADO
# ==========================================

class Llama31Client:
    """Cliente otimizado para Llama 3.1:8B via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model_name = "llama3.1:8b"
        self.is_available = False
        
        # Configurações otimizadas para Llama 3.1:8B
        self.generation_config = {
            "temperature": 0.1,          # Baixa para respostas mais consistentes
            "top_p": 0.9,               # Boa diversidade controlada
            "top_k": 40,                # Vocabulário focado
            "repeat_penalty": 1.1,      # Evita repetições
            "num_predict": 2048,        # Respostas detalhadas
            "stop": ["###", "---", "```"]  # Stopwords para melhor controle
        }
        
        self._check_and_setup()
    
    def _check_and_setup(self):
        """Verifica e configura Llama 3.1:8B"""
        print("🦙 Configurando Llama 3.1:8B...")
        
        # Verifica se Ollama está rodando
        if not self._check_ollama_running():
            raise ConnectionError(
                "❌ Ollama não está rodando!\n"
                "💡 Para resolver:\n"
                "   1. Instale: https://ollama.ai\n" 
                "   2. Execute: ollama serve\n"
                "   3. Baixe o modelo: ollama pull llama3.1:8b"
            )
        
        # Verifica se modelo está disponível
        if not self._check_model_available():
            print("📥 Baixando Llama 3.1:8B (isso pode demorar alguns minutos)...")
            if not self._download_model():
                raise RuntimeError("❌ Falha ao baixar Llama 3.1:8B")
        
        # Testa o modelo
        if self._test_model():
            self.is_available = True
            print("✅ Llama 3.1:8B configurado e testado com sucesso!")
        else:
            raise RuntimeError("❌ Llama 3.1:8B não está funcionando corretamente")
    
    def _check_ollama_running(self) -> bool:
        """Verifica se Ollama está rodando"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _check_model_available(self) -> bool:
        """Verifica se Llama 3.1:8B está disponível"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                return self.model_name in available_models
        except:
            pass
        return False
    
    def _download_model(self) -> bool:
        """Baixa Llama 3.1:8B"""
        try:
            print("🔄 Iniciando download do Llama 3.1:8B...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model_name},
                timeout=1800  # 30 minutos timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Erro no download: {e}")
            return False
    
    def _test_model(self) -> bool:
        """Testa Llama 3.1:8B"""
        try:
            response = self.generate(
                prompt="Diga apenas 'Teste OK' em português.",
                max_tokens=10
            )
            return "ok" in response.lower()
        except:
            return False
    
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = None) -> str:
        """Gera resposta usando Llama 3.1:8B"""
        if not self.is_available:
            raise ValueError("Llama 3.1:8B não está disponível")
        
        # Prepara configuração
        config = self.generation_config.copy()
        if max_tokens:
            config["num_predict"] = max_tokens
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": config
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get('response', '').strip()
            else:
                print(f"❌ Erro na geração: {response.status_code}")
                return "Erro na geração de resposta"
                
        except Exception as e:
            print(f"❌ Erro na geração: {e}")
            return "Erro de conectividade"
    
    def get_info(self) -> Dict:
        """Retorna informações do modelo"""
        return {
            "model": self.model_name,
            "status": "disponível" if self.is_available else "indisponível",
            "config": self.generation_config,
            "url": self.base_url
        }

# ==========================================
# 🧠 SISTEMA DE EMBEDDINGS
# ==========================================

class MiniLMEmbeddings:
    """Sistema de embeddings otimizado para português"""
    
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo de embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"🔄 Carregando {self.model_name}...")
            
            self.model = SentenceTransformer(self.model_name)
            self.is_fitted = True
            
            print(f"✅ Embeddings carregados!")
            
        except ImportError:
            raise ImportError(
                "sentence-transformers não instalado.\n"
                "Execute: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar embeddings: {e}")
    
    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Gera embeddings"""
        if not self.is_fitted:
            raise ValueError("Modelo não carregado")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings
            
        except Exception as e:
            print(f"❌ Erro ao gerar embeddings: {e}")
            raise

# ==========================================
# 🤖 RAG ADAPTATIVO COM LLAMA 3.1:8B
# ==========================================

class GMVAdaptiveRAGLlama31:
    """Sistema RAG adaptativo otimizado para Llama 3.1:8B"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Sistemas de IA
        self.embedding_model = MiniLMEmbeddings()
        self.llm_client = Llama31Client()
        
        # Storage
        self.documents: List[ProcessDocument] = []
        self.chunk_embeddings: np.ndarray = None
        self.chunk_metadata: List[Dict] = []
        self.all_chunks: List[str] = []
        
        # Configuração
        self.logger = logging.getLogger(__name__)
        self.query_cache: Dict[str, RAGResult] = {}
        self.is_initialized = False
        
        # Prompts otimizados para Llama 3.1:8B
        self.prompt_templates = {
            QueryType.FACTUAL: """Você é um assistente especializado em análise de processos jurídicos do GMV.

INSTRUÇÕES:
- Responda de forma direta e factual
- Cite informações específicas dos documentos
- Use dados concretos (números, datas, nomes)
- Mantenha o foco na pergunta

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA: {query}

RESPOSTA:""",

            QueryType.ANALYTICAL: """Você é um analista especializado em processos jurídicos do GMV.

INSTRUÇÕES:
- Faça uma análise detalhada e comparativa
- Identifique padrões e tendências
- Compare dados entre diferentes processos
- Forneça insights relevantes

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA: {query}

ANÁLISE:""",

            QueryType.CONTEXTUAL: """Você é um investigador experiente do GMV.

INSTRUÇÕES:
- Explique o contexto e circunstâncias
- Relacione informações entre documentos
- Descreva o cenário completo
- Identifique conexões importantes

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA: {query}

CONTEXTO:""",

            QueryType.OPINION: """Você é um consultor jurídico especializado do GMV.

INSTRUÇÕES:
- Forneça avaliação baseada em padrões observados
- Identifique tendências e indicadores
- Baseie-se nos dados disponíveis
- Seja objetivo nas conclusões

CONTEXTO DOS DOCUMENTOS:
{context}

PERGUNTA: {query}

AVALIAÇÃO:"""
        }
        
        # Prompt de sistema otimizado para Llama 3.1
        self.system_prompt = """Você é um assistente especializado em análise de processos jurídicos do Gabinete de Monitoramento e Vigilância (GMV).

CARACTERÍSTICAS:
- Responda sempre em português brasileiro
- Seja preciso e objetivo
- Cite informações específicas dos documentos
- Mantenha confidencialidade dos dados
- Foque apenas nas informações fornecidas
- Use linguagem técnica apropriada

FORMATO DE RESPOSTA:
- Seja claro e estruturado
- Use bullets quando apropriado
- Evite especulações
- Baseie-se somente nos dados fornecidos"""
    
    def initialize(self, triagem_path: str, pasta_destino: str, pasta_dat: str = None) -> bool:
        """Inicializa o sistema"""
        try:
            self.logger.info("🔄 Inicializando RAG com Llama 3.1:8B...")
            
            num_docs = self.load_gmv_data(triagem_path, pasta_destino, pasta_dat or "")
            
            if num_docs > 0:
                self.is_initialized = True
                self.logger.info(f"✅ RAG inicializado com {num_docs} documentos")
                print(f"🦙 Usando Llama 3.1:8B para geração de respostas")
                return True
            else:
                self.logger.error("Falha - nenhum documento carregado")
                return False
                
        except Exception as e:
            self.logger.error(f"Erro na inicialização: {e}")
            return False
    
    def load_gmv_data(self, triagem_path: str, pasta_destino: str, pasta_dat: str) -> int:
        """Carrega dados do GMV"""
        try:
            processos = self._extrair_tabela_triagem(triagem_path)
            
            if not processos:
                self.logger.warning("⚠️ Nenhum processo na triagem")
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
            self.logger.error(f"Erro ao carregar dados: {e}")
            return 0
    
    def _extrair_tabela_triagem(self, arquivo_md: str) -> List[Dict]:
        """Extrai dados da tabela de triagem"""
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
            self.logger.error(f"Erro ao extrair tabela: {e}")
            
        return processos
    
    def _process_documents(self):
        """Processa documentos em chunks"""
        self.logger.info("🔄 Processando documentos...")
        
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
            
            print("🔄 Gerando embeddings...")
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
        """Classifica tipo de consulta"""
        query_lower = query.lower()
        
        # Padrões otimizados para português brasileiro
        factual_patterns = [
            r'\b(qual|quais|quando|onde|quem|quantos|numero|data|nome)\b',
            r'\b(processo.*numero|numero.*processo)\b',
            r'\b(responsavel|status|tema|data)\b',
            r'\b(quanto|quantas|que.*e|o.*que)\b'
        ]
        
        analytical_patterns = [
            r'\b(compare|analise|diferenca|relacao|tendencia|padrao)\b',
            r'\b(maior|menor|melhor|pior|mais|menos)\b',
            r'\b(como.*difere|porque.*diferente|em.*relacao)\b',
            r'\b(distribui[cç][aã]o|frequencia|incidencia)\b'
        ]
        
        contextual_patterns = [
            r'\b(contexto|situacao|cenario|ambiente|circunstancia)\b',
            r'\b(explique.*porque|razao.*por|motivo)\b',
            r'\b(considerando|levando.*conta|dado.*que)\b',
            r'\b(historico|antecedentes|origem)\b'
        ]
        
        opinion_patterns = [
            r'\b(opini[aã]o|acho|acredito|parece|talvez|provavel)\b',
            r'\b(tendencia|futuro|prever|expectativa|perspectiva)\b',
            r'\b(avalia[cç][aã]o|julgamento|considera[cç][aã]o)\b'
        ]
        
        # Classifica por prioridade
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
        
        return QueryType.FACTUAL  # Default
    
    def _retrieval_strategy(self, query: str, query_type: QueryType, k: int = 5) -> List[Dict]:
        """Estratégia de recuperação adaptativa"""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        if query_type == QueryType.ANALYTICAL:
            # Diversidade para análises
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
            # Boost para suspeitos e investigações
            boosted_similarities = similarities.copy()
            
            for i, metadata in enumerate(self.chunk_metadata):
                if metadata['suspeitos']:
                    boosted_similarities[i] *= 1.3
                if metadata['status'].lower() in ['suspeito', 'investigação', 'investigacao']:
                    boosted_similarities[i] *= 1.2
            
            final_indices = np.argsort(boosted_similarities)[-k:][::-1]
            
        else:  # FACTUAL e OPINION
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
    
    def _generate_response_with_llama(self, query: str, retrieved_chunks: List[Dict], strategy: str) -> str:
        """Gera resposta usando Llama 3.1:8B"""
        
        if not retrieved_chunks:
            return "Desculpe, não encontrei informações relevantes para sua consulta."
        
        # Prepara contexto otimizado para Llama 3.1
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk['metadata']
            context_parts.append(f"""
[DOCUMENTO {i+1}]
Processo: {metadata['document_id']}
Tema: {metadata['tema']}
Status: {metadata['status']}
Relevância: {chunk['similarity']:.3f}

Conteúdo:
{chunk['content'][:600]}
""")
        
        context = "\n".join(context_parts)
        
        # Seleciona template
        query_type = QueryType(strategy)
        prompt_template = self.prompt_templates.get(query_type, self.prompt_templates[QueryType.FACTUAL])
        
        # Gera prompt final
        final_prompt = prompt_template.format(context=context, query=query)
        
        # Gera resposta com Llama 3.1:8B
        try:
            response = self.llm_client.generate(
                prompt=final_prompt,
                system_prompt=self.system_prompt
            )
            
            # Adiciona metadados
            response += f"\n\n---\n"
            response += f"*Gerado por: Llama 3.1:8B*\n"
            response += f"*Baseado em {len(retrieved_chunks)} documentos*\n"
            response += f"*Estratégia: {strategy}*"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Erro na geração: {e}")
            return f"Erro ao gerar resposta: {str(e)}"
    
    def query(self, query_text: str, k: int = 5, use_cache: bool = True) -> RAGResult:
        """Executa consulta RAG"""
        if not self.is_initialized:
            raise ValueError("Sistema não inicializado. Execute initialize() primeiro.")
        
        start_time = datetime.now()
        
        # Cache
        if use_cache and query_text in self.query_cache:
            self.logger.info("🔄 Cache hit")
            return self.query_cache[query_text]
        
        # Classifica consulta
        query_type = self._classify_query_type(query_text)
        self.logger.info(f"🔍 Query tipo: {query_type.value}")
        
        # Recupera chunks
        retrieved_chunks = self._retrieval_strategy(query_text, query_type, k)
        
        # Gera resposta
        response = self._generate_response_with_llama(query_text, retrieved_chunks, query_type.value)
        
        # Calcula confiança
        if retrieved_chunks:
            avg_similarity = np.mean([chunk['similarity'] for chunk in retrieved_chunks])
            confidence = min(avg_similarity * 1.1, 1.0)
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
            processing_time=processing_time,
            llm_model="llama3.1:8b"
        )
        
        # Cache
        if use_cache:
            self.query_cache[query_text] = result
        
        self.logger.info(f"✅ Processado em {processing_time:.2f}s - confiança {confidence:.3f}")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Estatísticas do sistema"""
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
            "embedding_model": "all-MiniLM-L12-v2",
            "llm_model": "llama3.1:8b",
            "llm_info": self.llm_client.get_info(),
            "status_distribution": status_count,
            "tema_distribution": tema_count,
            "top_suspeitos": dict(sorted(suspeitos_count.items(), key=lambda x: x[1], reverse=True)[:10])
        }
    
    def test_llama_connection(self) -> Dict:
        """Testa conexão com Llama 3.1:8B"""
        test_queries = [
            "Diga apenas 'Conectado' em português",
            "Responda '2+2=4' apenas",
            "Teste de português: responda 'OK'"
        ]
        
        results = {}
        for query in test_queries:
            try:
                start = datetime.now()
                response = self.llm_client.generate(query, max_tokens=10)
                end = datetime.now()
                
                results[query] = {
                    "response": response,
                    "time": (end - start).total_seconds(),
                    "success": len(response) > 0
                }
            except Exception as e:
                results[query] = {
                    "error": str(e),
                    "success": False
                }
        
        return results

# ==========================================
# 🎯 INSTÂNCIA GLOBAL
# ==========================================

# Instância global otimizada para Llama 3.1:8B
rag_instance = GMVAdaptiveRAGLlama31()

def initialize_rag(triagem_path: str, pasta_destino: str, pasta_dat: str = None) -> bool:
    """Inicializa RAG com Llama 3.1:8B"""
    return rag_instance.initialize(triagem_path, pasta_destino, pasta_dat)

def query_rag(query: str, k: int = 5) -> Dict:
    """Executa consulta"""
    try:
        result = rag_instance.query(query, k=k)
        return result.to_dict()
    except Exception as e:
        return {
            "error": str(e),
            "query": query,
            "response": "Erro ao processar consulta",
            "confidence_score": 0.0,
            "processing_time": 0.0,
            "llm_model": "llama3.1:8b"
        }

def get_rag_statistics() -> Dict:
    """Estatísticas do RAG"""
    return rag_instance.get_statistics()

def test_llama_connection() -> Dict:
    """Testa Llama 3.1:8B"""
    return rag_instance.test_llama_connection()