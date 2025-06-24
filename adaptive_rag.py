# adaptive_rag.py - VERSÃO INTEGRADA COM CONVERSATIONAL RAG

import os
import pickle
import hashlib
import re
import datetime
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter
import math
import json
import logging

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever

logger = logging.getLogger(__name__)

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
# 💬 SISTEMA CONVERSACIONAL INTEGRADO
# =================================================================================

class ConversationalLayer:
    def __init__(self):
        self.conversation_patterns = self._load_conversation_patterns()
        self.context_memory = {}
        
    def _load_conversation_patterns(self) -> Dict[str, Dict]:
        return {
            'greetings': {
                'patterns': [
                    r'\b(oi|olá|ola|hey|e ai|eai|salve)\b',
                    r'\bbom\s+dia\b',
                    r'\bboa\s+tarde\b',
                    r'\bboa\s+noite\b',
                    r'\bbom\s+final\s+de\s+semana\b',
                    r'\bcomo\s+(vai|está|esta|ta)\b',
                    r'\btudo\s+(bem|bom|certo)\b',
                    r'\be\s+aí\b',
                    r'\bbeleza\b',
                    r'\btranquilo\b'
                ],
                'responses': [
                    "Olá! 😊 Como posso ajudar você hoje?",
                    "Oi! Tudo bem? Em que posso ser útil?",
                    "Olá! Prazer em falar contigo. Como posso ajudar?",
                    "Oi! 👋 Pronto para ajudar. O que você precisa?",
                    "Olá! Espero que esteja tendo um ótimo dia. Como posso ajudar?"
                ],
                'time_specific': {
                    'morning': [
                        "Bom dia! ☀️ Como posso ajudar você hoje?",
                        "Bom dia! Espero que tenha uma manhã produtiva. Em que posso ajudar?",
                        "Oi, bom dia! Pronto para te ajudar. O que precisa?"
                    ],
                    'afternoon': [
                        "Boa tarde! 🌤️ Como posso ser útil?",
                        "Boa tarde! Em que posso ajudar você hoje?",
                        "Oi, boa tarde! O que você gostaria de saber?"
                    ],
                    'evening': [
                        "Boa noite! 🌙 Como posso ajudar?",
                        "Boa noite! Em que posso ser útil?",
                        "Oi, boa noite! O que você precisa?"
                    ]
                }
            },
            'farewells': {
                'patterns': [
                    r'\b(tchau|bye|adeus|até|falou|valeu)\b',
                    r'\bobrigad[oa]\b',
                    r'\bté\s+mais\b',
                    r'\baté\s+(logo|mais|breve)\b',
                    r'\bum\s+abraço\b',
                    r'\bfico\s+por\s+aqui\b'
                ],
                'responses': [
                    "Até mais! 👋 Foi um prazer ajudar!",
                    "Tchau! Qualquer coisa, é só chamar! 😊",
                    "Até logo! Espero ter ajudado!",
                    "Valeu! Volte sempre que precisar! 🤝",
                    "Tchau! Tenha um ótimo dia! ☀️"
                ]
            },
            'thanks': {
                'patterns': [
                    r'\bobrigad[oa]\b',
                    r'\bvaleu\b',
                    r'\b(muito\s+)?obrigad[oa]\b',
                    r'\bmto\s+obrigad[oa]\b',
                    r'\bagradeço\b',
                    r'\bthanks?\b'
                ],
                'responses': [
                    "Por nada! 😊 Fico feliz em ajudar!",
                    "De nada! É sempre um prazer ajudar!",
                    "Magina! Para isso estou aqui! 🤝",
                    "Não há de quê! Precisando, é só falar!",
                    "Disponha! Qualquer dúvida, estou aqui!"
                ]
            },
            'how_are_you': {
                'patterns': [
                    r'\bcomo\s+(você\s+)?(está|esta|vai|anda)\b',
                    r'\btudo\s+(bem|bom|certo)\s+com\s+você\b',
                    r'\bcomo\s+(você\s+)?se\s+sente\b',
                    r'\bcomo\s+(anda|vai)\s+você\b'
                ],
                'responses': [
                    "Estou bem, obrigado por perguntar! 😊 Como você está?",
                    "Tudo certo por aqui! E você, como anda?",
                    "Estou ótimo e pronto para ajudar! Como posso ser útil?",
                    "Muito bem, obrigado! E você, tudo bem?",
                    "Estou excelente! Pronto para responder suas dúvidas! 💪"
                ]
            },
            'help_requests': {
                'patterns': [
                    r'\b(me\s+)?ajuda\b',
                    r'\bpreciso\s+de\s+ajuda\b',
                    r'\bpode\s+me\s+ajudar\b',
                    r'\bo\s+que\s+(você\s+)?(faz|pode)\b',
                    r'\bcomo\s+(funciona|usar|utilizar)\b',
                    r'\bquais\s+suas\s+funções\b'
                ],
                'responses': [
                    "Claro! Posso ajudar com informações sobre processos jurídicos, documentos legais e muito mais. O que você gostaria de saber?",
                    "É claro que posso ajudar! Sou especializado em documentos legais e processos. Qual sua dúvida?",
                    "Com certeza! Estou aqui para ajudar com questões jurídicas e análise de documentos. Em que posso ser útil?",
                    "Claro! Posso responder perguntas sobre processos, analisar documentos e fornecer informações jurídicas. O que precisa?",
                    "Sem problemas! Especializo-me em assistência jurídica e análise documental. Como posso ajudar?"
                ]
            },
            'compliments': {
                'patterns': [
                    r'\b(você é|és)\s+(bom|ótimo|excelente|top|massa)\b',
                    r'\bmuito\s+(bom|útil|eficiente)\b',
                    r'\bparabéns\b',
                    r'\bgostei\s+da\s+resposta\b',
                    r'\bvocê\s+é\s+inteligente\b'
                ],
                'responses': [
                    "Muito obrigado! 😊 Fico feliz em ser útil!",
                    "Que bom que gostou! É um prazer ajudar!",
                    "Obrigado pelo elogio! Sempre dou meu melhor! 💪",
                    "Agradeço! É sempre gratificante saber que ajudei!",
                    "Obrigado! Continuo melhorando para servir melhor! 🚀"
                ]
            },
            'personal_questions': {
                'patterns': [
                    r'\bqual\s+(seu|teu)\s+nome\b',
                    r'\bcomo\s+(você\s+)?se\s+chama\b',
                    r'\bquem\s+(você\s+)?é\b',
                    r'\bo\s+que\s+(você\s+)?é\b',
                    r'\bvocê\s+é\s+(um\s+)?robô\b'
                ],
                'responses': [
                    "Sou seu assistente jurídico especializado em documentos legais! 🤖⚖️ Estou aqui para ajudar com processos e questões jurídicas.",
                    "Me chamo Assistente GMV! Sou especializado em análise de documentos jurídicos e processos legais. Como posso ajudar?",
                    "Sou um assistente inteligente focado em questões jurídicas! Minha especialidade é analisar documentos e processos legais.",
                    "Sou o Assistente do sistema GMV, especializado em direito e análise documental! Em que posso ser útil?",
                    "Sou seu assistente jurídico digital! 💼 Especializado em processos, documentos legais e orientação jurídica."
                ]
            },
            'confirmations': {
                'patterns': [
                    r'\b(sim|claro|certo|ok|okay|beleza|perfeito)\b',
                    r'\b(pode|vamos)\s+(seguir|continuar)\b',
                    r'\bentendi\b',
                    r'\bcontinua\b'
                ],
                'responses': [
                    "Perfeito! Como posso ajudar?",
                    "Ótimo! O que você gostaria de saber?",
                    "Certo! Qual sua próxima pergunta?",
                    "Beleza! Em que mais posso ser útil?",
                    "Entendido! Qual sua dúvida?"
                ]
            }
        }
    
    def detect_conversation_type(self, text: str) -> Optional[str]:
        text_lower = text.lower().strip()
        text_clean = re.sub(r'[!?.,;]', '', text_lower)
        for conv_type, data in self.conversation_patterns.items():
            for pattern in data['patterns']:
                if re.search(pattern, text_clean, re.IGNORECASE):
                    return conv_type
        return None
    
    def get_time_period(self) -> str:
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    def generate_conversational_response(self, text: str, conv_type: str) -> str:
        conv_data = self.conversation_patterns[conv_type]
        if conv_type == 'greetings' and 'time_specific' in conv_data:
            time_period = self.get_time_period()
            if time_period in conv_data['time_specific']:
                return random.choice(conv_data['time_specific'][time_period])
        return random.choice(conv_data['responses'])
    
    def should_use_conversational_response(self, text: str) -> bool:
        if len(text.strip()) <= 30:
            return True
        conv_type = self.detect_conversation_type(text)
        if conv_type:
            text_without_conv = text.lower()
            patterns = self.conversation_patterns[conv_type]['patterns']
            for pattern in patterns:
                text_without_conv = re.sub(pattern, '', text_without_conv, flags=re.IGNORECASE)
            remaining_words = len([w for w in text_without_conv.split() if len(w) > 2])
            return remaining_words <= 2
        return False

class SmartRAGHandler:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.conversational = ConversationalLayer()
        self.conversation_history = []
        
    def process_query(self, question: str, user_id: str = "default") -> Dict[str, Any]:
        question_clean = question.strip()
        conv_type = self.conversational.detect_conversation_type(question_clean)
        
        if conv_type and self.conversational.should_use_conversational_response(question_clean):
            response = self.conversational.generate_conversational_response(question_clean, conv_type)
            return {
                "answer": response,
                "type": "conversational",
                "conversation_type": conv_type,
                "sources": [],
                "documents_found": 0,
                "is_natural_response": True,
                "suggestion": "Faça uma pergunta sobre processos ou documentos jurídicos para que eu possa ajudar melhor!"
            }
        elif conv_type:
            question_technical = self._extract_technical_part(question_clean, conv_type)
            if question_technical and len(question_technical.strip()) > 5:
                # Usa o método base para evitar recursão
                if hasattr(self.rag_system, 'base_query'):
                    rag_result = self.rag_system.base_query(question_technical)
                else:
                    rag_result = self._call_original_rag(question_technical)
                
                if "error" not in rag_result:
                    conv_prefix = self._get_conversational_prefix(conv_type)
                    rag_result["answer"] = f"{conv_prefix}\n\n{rag_result['answer']}"
                    rag_result["type"] = "hybrid"
                    rag_result["conversation_type"] = conv_type
                    rag_result["is_natural_response"] = True
                return rag_result
            else:
                response = self.conversational.generate_conversational_response(question_clean, conv_type)
                return {
                    "answer": response,
                    "type": "conversational",
                    "conversation_type": conv_type,
                    "sources": [],
                    "documents_found": 0,
                    "is_natural_response": True
                }
        else:
            # Usa o método base para evitar recursão
            if hasattr(self.rag_system, 'base_query'):
                rag_result = self.rag_system.base_query(question_clean)
            else:
                rag_result = self._call_original_rag(question_clean)
            
            if "error" not in rag_result:
                rag_result["type"] = "technical"
                rag_result["is_natural_response"] = False
                rag_result["answer"] = self._humanize_technical_response(rag_result["answer"])
            return rag_result
    
    def _call_original_rag(self, question: str) -> Dict[str, Any]:
        """Chama o método RAG original diretamente para evitar recursão"""
        try:
            # Chama o método básico do UltraFastRAG diretamente
            if not self.rag_system.is_initialized or not self.rag_system.vector_store:
                return {"error": "Sistema não inicializado"}
            
            k = self.rag_system.config.top_k
            print(f"🔍 Processamento RAG direto: {question[:50]}...")
            
            # Busca documentos relevantes
            relevant_docs = self.rag_system.vector_store.similarity_search(question, k=k, min_score=0.15)
            
            if not relevant_docs:
                return {
                    "error": "Nenhum documento relevante encontrado",
                    "suggestion": "Tente reformular a pergunta com termos mais específicos"
                }
            
            # Prepara contexto (versão simplificada)
            context_parts = []
            sources = set()
            
            for i, doc in enumerate(relevant_docs, 1):
                content = doc.page_content.strip()
                context_parts.append(f"DOCUMENTO {i}:\n{content}")
                
                if 'source' in doc.metadata:
                    sources.add(doc.metadata['source'])
                elif 'filename' in doc.metadata:
                    sources.add(doc.metadata['filename'])
            
            context = "\n\n".join(context_parts)
            
            # Template simplificado
            prompt_template = f"""Com base nos documentos fornecidos, responda a pergunta de forma clara e objetiva.

DOCUMENTOS:
{context}

PERGUNTA: {question}

RESPOSTA:"""
            
            # Chama LLM
            if hasattr(self.rag_system.llm, 'invoke'):
                answer = self.rag_system.llm.invoke(prompt_template)
            else:
                answer = self.rag_system.llm(prompt_template)
            
            return {
                "answer": answer.strip(),
                "sources": list(sources),
                "documents_found": len(relevant_docs)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_technical_part(self, text: str, conv_type: str) -> str:
        patterns = self.conversational.conversation_patterns[conv_type]['patterns']
        technical_text = text.lower()
        for pattern in patterns:
            technical_text = re.sub(pattern, '', technical_text, flags=re.IGNORECASE)
        technical_text = re.sub(r'\b(e|mas|então|aí|né|assim|tipo)\b', '', technical_text)
        return technical_text.strip()
    
    def _get_conversational_prefix(self, conv_type: str) -> str:
        prefixes = {
            'greetings': random.choice([
                "Olá! 😊", 
                "Oi!", 
                "Olá, que bom falar contigo!"
            ]),
            'help_requests': random.choice([
                "Claro, posso ajudar!", 
                "Com certeza!", 
                "É claro que posso ajudar!"
            ]),
            'thanks': random.choice([
                "De nada! 😊", 
                "Por nada!", 
                "Fico feliz em ajudar!"
            ])
        }
        return prefixes.get(conv_type, "")
    
    def _humanize_technical_response(self, answer: str) -> str:
        if len(answer) < 50:
            friendly_endings = [
                " Espero ter esclarecido sua dúvida! 😊",
                " Precisa de mais alguma informação?",
                " Posso detalhar mais algum ponto específico?",
                " Isso responde sua pergunta?"
            ]
            answer += random.choice(friendly_endings)
        elif "conforme" in answer.lower() or "nos termos" in answer.lower():
            if not any(emoji in answer for emoji in ['😊', '👍', '💡', '⚖️']):
                answer += " \n\n💡 Espero ter ajudado! Qualquer dúvida, é só perguntar."
        return answer

def enhance_rag_with_conversation(rag_system):
    """Melhora o sistema RAG com capacidades conversacionais"""
    # Salva referência para o método original
    original_query_method = rag_system.query
    smart_handler = SmartRAGHandler(rag_system)
    
    # Método que usa o original diretamente (evita recursão)
    def base_rag_query(question: str, top_k: int = 4) -> Dict[str, Any]:
        """Chama o método RAG original diretamente"""
        return original_query_method(question, top_k)
    
    # Atualiza o handler para usar o método base
    smart_handler.rag_system.base_query = base_rag_query
    
    def enhanced_query(question: str, top_k: int = 4, user_id: str = "default") -> Dict[str, Any]:
        try:
            return smart_handler.process_query(question, user_id)
        except Exception as e:
            logger.error(f"Erro no processamento conversacional: {e}")
            # Usa o método base em caso de erro
            return base_rag_query(question, top_k)
    
    rag_system.query = enhanced_query
    rag_system.conversational_handler = smart_handler
    logger.info("✅ Sistema RAG melhorado com capacidades conversacionais!")
    return rag_system

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
        query_embedding = None
        
        if self.use_ollama:
            try:
                query_embedding = self.ollama_embeddings.embed_query(query)
            except Exception as e:
                print(f"⚠️ Erro Ollama embedding: {e}")
                self.use_ollama = False
        
        if not self.use_ollama:
            # Inicializa TF-IDF se necessário
            if not hasattr(self, 'tfidf_embedder') or self.tfidf_embedder is None:
                print("🔄 Inicializando TF-IDF embedder...")
                self.tfidf_embedder = SmartTFIDFEmbedder()
                # Treina com documentos existentes
                if self.documents:
                    self.tfidf_embedder.fit(self.documents)
            
            query_embedding = self.tfidf_embedder.embed_query(query)
        
        if query_embedding is None:
            print("❌ Erro ao gerar embedding da query")
            return []
        
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
# 🎯 SISTEMA RAG PRINCIPAL COM INTEGRAÇÃO CONVERSACIONAL
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
    enable_conversational: bool = True  # Nova opção

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
            
            # Integra sistema conversacional se habilitado
            if self.config.enable_conversational:
                self._integrate_conversational()
            
            return True
            
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            print("💡 Verifique se o Ollama está rodando e os modelos instalados")
            return False
    
    def _integrate_conversational(self):
        """Integra o sistema conversacional"""
        try:
            enhance_rag_with_conversation(self)
            print("✅ Sistema conversacional integrado!")
        except Exception as e:
            print(f"⚠️ Erro ao integrar sistema conversacional: {e}")
            print("🔄 Continuando com sistema RAG básico")
    
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
        """Realiza consulta no sistema RAG (método básico, pode ser sobrescrito pelo conversacional)"""
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
    conversational_status = "Ativo" if rag_system.config.enable_conversational and hasattr(rag_system, 'conversational_handler') else "Inativo"
    
    return {
        "status": "online", 
        "message": f"{len(rag_system.documents)} documentos carregados", 
        "isReady": True,
        "documents_loaded": len(rag_system.documents),
        "embedding_method": embedding_method,
        "conversational": conversational_status,
        "model": rag_system.config.model_name,
        "data_path": rag_system.data_path
    }

def query_rag(question: str, top_k: int = 4):
    """Interface para consultas RAG (agora com capacidades conversacionais)"""
    return rag_system.query(question, top_k)

# =================================================================================
# 🧪 TESTE STANDALONE
# =================================================================================

def test_conversational_responses():
    """Teste específico para respostas conversacionais"""
    conv_layer = ConversationalLayer()
    test_cases = [
        "Oi!",
        "Bom dia!",
        "Como você está?",
        "Obrigado pela ajuda",
        "Você pode me ajudar?",
        "Tchau!",
        "Qual seu nome?",
        "Oi, bom dia! Preciso saber sobre um processo específico",
        "Olá! Gostaria de entender sobre TEA e terapia ABA",
        "Valeu pelas informações! Qual o valor da causa do processo 1005888?"
    ]
    
    print("🧪 TESTE DE RESPOSTAS CONVERSACIONAIS")
    print("=" * 60)
    
    for test in test_cases:
        conv_type = conv_layer.detect_conversation_type(test)
        should_conv = conv_layer.should_use_conversational_response(test)
        print(f"\n📝 Input: '{test}'")
        print(f"🎯 Tipo: {conv_type}")
        print(f"🗣️ Conversacional: {should_conv}")
        
        if conv_type and should_conv:
            response = conv_layer.generate_conversational_response(test, conv_type)
            print(f"💬 Resposta: {response}")
        else:
            print(f"🔍 Resposta: [Processamento técnico RAG]")

if __name__ == "__main__":
    print("🧪 TESTE DO SISTEMA RAG ADAPTATIVO COM CONVERSACIONAL")
    print("=" * 60)
    
    # Primeiro testa as respostas conversacionais
    test_conversational_responses()
    
    print("\n" + "="*60)
    
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
    
    # Configura sistema com conversacional habilitado
    config = UltraFastRAGConfig(
        model_name="gemma:2b",
        temperature=0.1,
        data_dir=test_dir,
        use_ollama_embeddings=True,
        enable_conversational=True  # Habilita sistema conversacional
    )
    
    test_rag = UltraFastRAG(config)
    
    print("\n🔄 Inicializando sistema...")
    if test_rag.initialize():
        print("✅ Sistema inicializado")
        
        print("\n📂 Carregando documentos...")
        docs_loaded = test_rag.load_documents_from_directory()
        print(f"✅ {docs_loaded} documentos carregados")
        
        if docs_loaded > 0:
            print("\n🧪 Executando testes mistos (conversacional + técnico)...")
            
            queries = [
                "Oi! Bom dia!",  # Conversacional
                "Qual é o número do processo e quem são as partes envolvidas?",  # Técnica
                "Obrigado! A terapia ABA é disponibilizada pelo SUS?",  # Híbrida
                "Quais tratamentos o agravante necessita?",  # Técnica
                "Valeu pelas informações! Tchau!"  # Conversacional
            ]
            
            for i, query in enumerate(queries, 1):
                print(f"\n--- TESTE {i} ---")
                print(f"Pergunta: {query}")
                
                result = test_rag.query(query)
                
                if "error" in result:
                    print(f"❌ Erro: {result['error']}")
                else:
                    print(f"Resposta: {result['answer']}")
                    print(f"Tipo: {result.get('type', 'técnico')}")
                    if result.get('sources'):
                        print(f"Fontes: {result['sources']}")
                    if result.get('documents_found'):
                        print(f"Documentos: {result['documents_found']}")
        else:
            print("⚠️ Nenhum documento carregado para teste")
    else:
        print("❌ Falha na inicialização")
        print("💡 Certifique-se que o Ollama está rodando:")
        print("   ollama serve")
        print("   ollama pull gemma:2b")
        print("   ollama pull nomic-embed-text")