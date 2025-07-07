import re
import datetime
import random
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

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
                rag_result = self.rag_system.query(question_technical)
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
            rag_result = self.rag_system.query(question_clean)
            if "error" not in rag_result:
                rag_result["type"] = "technical"
                rag_result["is_natural_response"] = False
                rag_result["answer"] = self._humanize_technical_response(rag_result["answer"])
            return rag_result
    
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
    original_query = rag_system.query
    smart_handler = SmartRAGHandler(rag_system)
    def enhanced_query(question: str, top_k: int = 4, user_id: str = "default") -> Dict[str, Any]:
        try:
            return smart_handler.process_query(question, user_id)
        except Exception as e:
            logger.error(f"Erro no processamento conversacional: {e}")
            return original_query(question, top_k)
    rag_system.query = enhanced_query
    rag_system.conversational_handler = smart_handler
    return rag_system