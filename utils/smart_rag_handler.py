import random
import re
from typing import Any, Dict
from utils.conversational_layer import ConversationalLayer


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
