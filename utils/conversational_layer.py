import datetime
import random 
from typing import Dict, Optional
import re

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