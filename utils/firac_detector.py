import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FIRACStructure:
    facts: str = ""
    issues: str = ""
    rules: str = ""
    application: str = ""
    conclusion: str = ""
    numero_processo: str = ""
    resumo: str = ""

class FIRACDetector:
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.firac_patterns = self._build_firac_patterns()
        
    def _build_firac_patterns(self) -> Dict[str, List[str]]:
        return {
            'firac_requests': [
                r'\b(firac|f\.?i\.?r\.?a\.?c\.?)\b.*?\b(processo|agravo)\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})\b',
                r'\b(faça|gere|crie|elabore|monte)\s+.*?(firac|f\.?i\.?r\.?a\.?c\.?)\s+.*?(processo|do processo)\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})?',
                r'\b(quero|preciso|solicito)\s+.*?(firac|f\.?i\.?r\.?a\.?c\.?)\s+.*?(processo|número)\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})?',
                r'\b(análise|analise)\s+(firac|f\.?i\.?r\.?a\.?c\.?)\s+.*?(processo)\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})?',
                r'\b(estrutur[ae]|format[eo])\s+.*(firac|f\.?i\.?r\.?a\.?c\.?)',
                r'\b(me\s+)?(mostre|apresente|exiba)\s+.*?(firac|f\.?i\.?r\.?a\.?c\.?)',
                r'\bfirac\s+(do|para|sobre)\s+(processo|agravo|caso)',
                r'\b(como|qual)\s+.*?firac.*?(processo|caso)'
            ],
            'numero_processo': [
                r'(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})',
                r'processo\s*n[úu]mero\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})',
                r'n[úuº°]\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})'
            ]
        }
    
    def detect_firac_request(self, text: str) -> Dict[str, Any]:
        text_clean = text.lower().strip()
        
        result = {
            'is_firac_request': False,
            'numero_processo': None,
            'request_type': None,
            'confidence': 0.0
        }
        
        for pattern in self.firac_patterns['firac_requests']:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                result['is_firac_request'] = True
                result['request_type'] = 'firac_analysis'
                result['confidence'] = 0.9
                
                numero_match = self._extract_numero_processo(text)
                if numero_match:
                    result['numero_processo'] = numero_match
                    result['confidence'] = 1.0
                
                logger.info(f"FIRAC solicitado detectado: {match.group(0)}")
                break
        
        return result
    
    def _extract_numero_processo(self, text: str) -> Optional[str]:
        for pattern in self.firac_patterns['numero_processo']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        return None
    
    def generate_firac(self, numero_processo: str = None, query_text: str = "") -> FIRACStructure:
        if not self.rag_system:
            logger.error("Sistema RAG não disponível para gerar FIRAC")
            return FIRACStructure(
                resumo="Sistema RAG não disponível para análise"
            )
        
        try:
            if not numero_processo:
                numero_processo = self._extract_numero_processo(query_text)
            
            if not numero_processo:
                return FIRACStructure(
                    resumo="Número do processo não identificado. Por favor, informe o número completo do processo."
                )
            
            logger.info(f"Gerando FIRAC para processo: {numero_processo}")
            
            search_query = f"processo {numero_processo} fatos questões direito aplicação decisão"
            rag_result = self.rag_system.query(search_query, top_k=6)
            
            if not rag_result.get('context_used'):
                return FIRACStructure(
                    numero_processo=numero_processo,
                    resumo=f"Processo {numero_processo} não encontrado na base de dados."
                )
            
            context = rag_result['context_used']
            firac = self._extract_firac_components(context, numero_processo)
            
            if hasattr(self.rag_system, 'llm'):
                firac = self._enhance_firac_with_llm(firac, context)
            
            logger.info(f"FIRAC gerado com sucesso para {numero_processo}")
            return firac
            
        except Exception as e:
            logger.error(f"Erro ao gerar FIRAC: {str(e)}")
            return FIRACStructure(
                numero_processo=numero_processo or "Não identificado",
                resumo=f"Erro ao gerar FIRAC: {str(e)}"
            )
    
    def _extract_firac_components(self, context: str, numero_processo: str) -> FIRACStructure:
        firac = FIRACStructure(numero_processo=numero_processo)
        
        patterns = {
            'facts': [
                r'(?:fatos?|alegou|alega|alegam|ocorr[eê]ncia|situação)[\s\S]*?(?=(?:quest[aã]o|direito|lei|artigo|c[oó]digo)|\n\n|$)',
                r'(?:RELATÓRIO|Relatório)[\s\S]*?(?=(?:FUNDAMENTAÇÃO|VOTO|quest[aã]o)|\n\n|$)',
                r'(?:autor|agravante|agravado).*?(?:requer|solicita|pede)[\s\S]*?(?=(?:quest[aã]o|direito|lei)|\n\n|$)'
            ],
            'issues': [
                r'(?:quest[aã]o|problema|controv[eé]rsia|disputa|litígio|ponto controvertido)[\s\S]*?(?=(?:direito|lei|artigo|aplica)|\n\n|$)',
                r'(?:QUESTÃO|Quest[aã]o|PROBLEMA)[\s\S]*?(?=(?:DIREITO|REGRA|LEI)|\n\n|$)'
            ],
            'rules': [
                r'(?:artigo|art\.?|lei|c[oó]digo|constituição|súmula|jurisprud[eê]ncia)[\s\S]*?(?=(?:aplicando|assim|portanto|logo)|\n\n|$)',
                r'(?:FUNDAMENT[AO]|Fundamentação|EMENTA)[\s\S]*?(?=(?:DISPOSITIVO|CONCLUSÃO)|\n\n|$)',
                r'(?:CF|CTN|CPC|CC).*?art\.?\s*\d+[\s\S]*?(?=(?:aplicando|assim)|\n\n|$)'
            ],
            'application': [
                r'(?:aplicando|considerando|assim|logo|portanto|neste caso|no caso)[\s\S]*?(?=(?:conclusão|decide|julgo)|\n\n|$)',
                r'(?:ANÁLISE|APLICAÇÃO|Análise)[\s\S]*?(?=(?:CONCLUSÃO|DISPOSITIVO)|\n\n|$)'
            ],
            'conclusion': [
                r'(?:conclui|decide|julgo|determino|defiro|indefiro|procedente|improcedente)[\s\S]*?(?=\n\n|$)',
                r'(?:DISPOSITIVO|CONCLUSÃO|DECIDIU|Conclusão)[\s\S]*?(?=\n\n|$)',
                r'(?:recurso.*?(?:provido|desprovido|conhecido))[\s\S]*?(?=\n\n|$)'
            ]
        }
        
        for component, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, context, re.IGNORECASE | re.MULTILINE)
                if matches:
                    content = ' '.join(matches)
                    content = re.sub(r'\s+', ' ', content.strip())[:800]
                    setattr(firac, component, content)
                    break
        
        return firac
    
    def _enhance_firac_with_llm(self, firac: FIRACStructure, context: str) -> FIRACStructure:
        try:
            prompt = f"""
            Analise o seguinte processo e estruture um FIRAC completo:

            PROCESSO: {firac.numero_processo}

            CONTEXTO:
            {context[:2000]}

            Forneça uma análise FIRAC estruturada:

            **FATOS (F):**
            [Descreva os fatos principais do caso de forma objetiva]

            **QUESTÕES JURÍDICAS (I):**
            [Identifique as questões de direito a serem resolvidas]

            **REGRAS APLICÁVEIS (R):**
            [Cite as leis, artigos e jurisprudências aplicáveis]

            **APLICAÇÃO (A):**
            [Analise como as regras se aplicam aos fatos]

            **CONCLUSÃO (C):**
            [Apresente a decisão e seus fundamentos]

            Seja objetivo e use linguagem jurídica adequada.
            """
            
            if hasattr(self.rag_system.llm, 'invoke'):
                response = self.rag_system.llm.invoke(prompt)
            else:
                response = self.rag_system.llm(prompt)
            
            enhanced_firac = self._parse_llm_firac_response(response, firac)
            return enhanced_firac
            
        except Exception as e:
            logger.error(f"Erro ao melhorar FIRAC com LLM: {str(e)}")
            return firac
    
    def _parse_llm_firac_response(self, response: str, original_firac: FIRACStructure) -> FIRACStructure:
        if isinstance(response, object) and hasattr(response, 'content'):
            response = response.content
        elif not isinstance(response, str):
            response = str(response)
        
        sections = {
            'facts': r'\*\*FATOS.*?\*\*\s*:?\s*(.*?)(?=\*\*|$)',
            'issues': r'\*\*QUESTÕES.*?\*\*\s*:?\s*(.*?)(?=\*\*|$)',
            'rules': r'\*\*REGRAS.*?\*\*\s*:?\s*(.*?)(?=\*\*|$)',
            'application': r'\*\*APLICAÇÃO.*?\*\*\s*:?\s*(.*?)(?=\*\*|$)',
            'conclusion': r'\*\*CONCLUSÃO.*?\*\*\s*:?\s*(.*?)(?=\*\*|$)'
        }
        
        enhanced_firac = original_firac
        
        for component, pattern in sections.items():
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 10:
                    setattr(enhanced_firac, component, content)
        
        return enhanced_firac
    
    def format_firac_response(self, firac: FIRACStructure) -> str:
        if not firac.facts and not firac.issues and not firac.rules:
            return firac.resumo or "FIRAC não pôde ser gerado para este processo."
        
        response = f"## 📋 ANÁLISE FIRAC\n\n"
        response += f"**Processo:** {firac.numero_processo}\n\n"
        
        if firac.facts:
            response += f"### 📌 FATOS (F)\n{firac.facts}\n\n"
        
        if firac.issues:
            response += f"### ❓ QUESTÕES JURÍDICAS (I)\n{firac.issues}\n\n"
        
        if firac.rules:
            response += f"### ⚖️ REGRAS APLICÁVEIS (R)\n{firac.rules}\n\n"
        
        if firac.application:
            response += f"### 🔍 APLICAÇÃO (A)\n{firac.application}\n\n"
        
        if firac.conclusion:
            response += f"### ✅ CONCLUSÃO (C)\n{firac.conclusion}\n\n"
        
        response += "---\n*Análise gerada automaticamente pelo sistema RAG*"
        
        return response

class FIRACEnabledRAG:
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.firac_detector = FIRACDetector(rag_system)
    
    def enhanced_query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        firac_detection = self.firac_detector.detect_firac_request(question)
        
        if firac_detection['is_firac_request']:
            logger.info("Solicitação de FIRAC detectada!")
            
            firac = self.firac_detector.generate_firac(
                numero_processo=firac_detection['numero_processo'],
                query_text=question
            )
            
            formatted_response = self.firac_detector.format_firac_response(firac)
            
            return {
                "answer": formatted_response,
                "type": "firac_analysis",
                "numero_processo": firac.numero_processo,
                "firac_structure": firac,
                "confidence": firac_detection['confidence'],
                "sources": [],
                "documents_found": 1 if firac.facts else 0,
                "is_structured_analysis": True
            }
        
        return self.rag_system.query(question, top_k)

def create_firac_enabled_rag(rag_system):
    return FIRACEnabledRAG(rag_system)