import os
import re
import unicodedata
import spacy
from functools import lru_cache
from typing import Dict, Tuple, Set, List
import gc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class AnonimizadorOtimizado:
    """Sistema de anonimização otimizado usando spaCy"""
    
    def __init__(self, caminho_palavras_descartadas="utils/palavras_descartadas.txt"):
        print("🚀 Inicializando AnonimizadorOtimizado...")
        
        # Carrega modelo spaCy apenas uma vez com componentes otimizados
        self.nlp = self._carregar_modelo_otimizado()
        
        # Cache para normalizações e mapeamentos
        self.cache_normalizacao = {}
        self.cache_suspeitos = None
        
        # Palavras descartadas carregadas do arquivo txt
        self.palavras_descartadas = self._carregar_palavras_descartadas(caminho_palavras_descartadas)
        
        # Padrões regex compilados para melhor performance
        self.padroes_regex = self._compilar_padroes()
        
        print(" AnonimizadorOtimizado inicializado com sucesso")
    
    def _carregar_modelo_otimizado(self):
        """Carrega o modelo spaCy com configurações otimizadas"""
        try:
            print("📦 Carregando modelo spaCy...")
            # Desabilita componentes desnecessários para performance
            nlp = spacy.load("pt_core_news_sm", 
                           disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
            
            # Aumenta limite para textos grandes
            nlp.max_length = 2_000_000
            
            print(" Modelo spaCy carregado com sucesso")
            return nlp
        except Exception as e:
            print(f"Erro ao carregar modelo spaCy: {e}")
            print(" Continuando sem spaCy - funcionalidade limitada")
            return None
    
    @lru_cache(maxsize=2000)
    def normalizar(self, texto: str) -> str:
        """Normaliza texto com cache para evitar reprocessamento"""
        if not texto:
            return ""
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode().lower().strip()
    
    def _carregar_palavras_descartadas(self, caminho="utils/palavras_descartadas.txt") -> Set[str]:
        """Carrega palavras descartadas de arquivo txt como set para busca rápida"""
        palavras = set()
        
        # Tenta diferentes caminhos possíveis
        caminhos_possiveis = [
            caminho,
            "utils/palavras_descartadas.txt",
            "palavras_descartadas.txt",
            os.path.join(".", "palavras_descartadas.txt"),
            os.path.join("utils", "palavras_descartadas.txt")
        ]
        
        arquivo_encontrado = None
        for caminho_teste in caminhos_possiveis:
            if os.path.exists(caminho_teste):
                arquivo_encontrado = caminho_teste
                print(f"📄 Arquivo de palavras descartadas encontrado: {os.path.abspath(caminho_teste)}")
                break
        
        if not arquivo_encontrado:
            print(f" Arquivo de palavras descartadas não encontrado. Usando lista padrão.")
            
            # Lista mínima de fallback
            palavras = {
                'E', 'EM', 'NO', 'NA', 'DOS', 'DAS', 'DE', 'DO', 'DA', 'AOS', 'AO',
                'COM', 'SEM', 'POR', 'PARA', 'ANTE', 'APÓS', 'APOS', 'ATÉ', 'ATE',
                'CONTRA', 'DESDE', 'ENTRE', 'PERANTE', 'SEGUNDO', 'SOBRE', 'CONFORME',
                'TRIBUNAL', 'VARA', 'PROCESSO', 'JUIZ', 'JUÍZA', 'DOUTOR', 'DOUTORA',
                'ARTIGO', 'PARÁGRAFO', 'INCISO', 'ALÍNEA', 'LETRA', 'ITEM'
            }
            return palavras
        
        try:
            total_palavras = 0
            with open(arquivo_encontrado, "r", encoding="utf-8") as f:
                for linha in f:
                    palavra = linha.strip().upper()
                    
                    # Pula linhas vazias e comentários
                    if not palavra or palavra.startswith('#'):
                        continue
                    
                    # Adiciona palavra em maiúscula E sua versão normalizada
                    palavras.add(palavra)
                    palavras.add(self.normalizar(palavra))
                    total_palavras += 1
            
            print(f"📊 Carregadas {total_palavras} palavras descartadas ({len(palavras)} incluindo normalizadas)")
            
        except Exception as e:
            print(f"Erro ao ler arquivo {arquivo_encontrado}: {e}")
            # Lista mínima de fallback
            palavras = {
                'E', 'EM', 'NO', 'NA', 'DOS', 'DAS', 'DE', 'DO', 'DA', 'AOS', 'AO',
                'COM', 'SEM', 'POR', 'PARA', 'TRIBUNAL', 'VARA', 'PROCESSO'
            }
        
        return palavras
    
    def _compilar_padroes(self) -> Dict[str, re.Pattern]:
        """Compila padrões regex uma única vez para melhor performance"""
        return {
            'cpf': re.compile(r'\b\d{3}\.?\d{3}\.?\d{3}[-\.]?\d{2}\b'),
            'rg': re.compile(r'\b\d{1,2}\.?\d{3}\.?\d{3}[-\.]?\d{1,2}\b'),
            'cnpj': re.compile(r'\b\d{2}\.?\d{3}\.?\d{3}[\/\.]?\d{4}[-\.]?\d{2}\b'),
            'data': re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}\b'),
            'data_iso': re.compile(r'\b\d{4}[-\.]\d{1,2}[-\.]\d{1,2}\b'),
            'email': re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b'),
            'cep': re.compile(r'\b\d{5}[-\.]?\d{3}\b'),
            'telefone': re.compile(r'\b(?:\(?\d{2}\)?\s?)?(?:9?\d{4,5})[-\.\s]?\d{4}\b'),
            'processo': re.compile(r'\b\d{7}[-\.]?\d{2}[-\.]?\d{4}[-\.]?\d[-\.]?\d{2}[-\.]?\d{4}\b')
        }
    
    def carregar_suspeitos_mapeados(self, caminho="utils/suspeitos.txt") -> Dict[str, Tuple[str, str]]:
        """Carrega mapeamento de suspeitos com cache"""
        if self.cache_suspeitos is not None:
            return self.cache_suspeitos
        
        mapa = {}
        
        # Tenta diferentes caminhos
        caminhos_possiveis = [
            caminho,
            "utils/suspeitos.txt",
            "suspeitos.txt",
            "./utils/suspeitos.txt"
        ]
        
        arquivo_encontrado = None
        for caminho_teste in caminhos_possiveis:
            if os.path.exists(caminho_teste):
                arquivo_encontrado = caminho_teste
                break
        
        if not arquivo_encontrado:
            print(f" Arquivo de suspeitos não encontrado. Continuando sem lista específica.")
            self.cache_suspeitos = mapa
            return mapa
        
        try:
            print(f"📄 Carregando suspeitos de: {arquivo_encontrado}")
            with open(arquivo_encontrado, "r", encoding="utf-8") as f:
                total_suspeitos = 0
                for linha in f:
                    if "|" in linha:
                        partes = linha.strip().split("|")
                        if len(partes) >= 2:
                            ident = partes[0].strip()
                            nome = partes[1].strip()
                            
                            # Adiciona mapeamento completo
                            chave_nome_completo = self.normalizar(nome)
                            mapa[chave_nome_completo] = (ident, nome)
                            
                            # Adiciona também primeiro + último nome
                            palavras_nome = nome.strip().split()
                            if len(palavras_nome) >= 2:
                                chave_nome_sobrenome = self.normalizar(f"{palavras_nome[0]} {palavras_nome[-1]}")
                                mapa[chave_nome_sobrenome] = (ident, nome)
                            
                            total_suspeitos += 1
            
            print(f" Carregados {total_suspeitos} suspeitos com {len(mapa)} mapeamentos")
            
        except Exception as e:
            print(f"Erro ao carregar suspeitos: {e}")
        
        self.cache_suspeitos = mapa
        return mapa
    
    def extrair_nomes_spacy_otimizado(self, texto: str, debug=False) -> List[str]:
        """Extrai nomes usando spaCy + regex híbrido para melhor cobertura"""
        nomes_spacy = set()
        nomes_regex = set()
        
        # MÉTODO 1: spaCy (se disponível)
        if self.nlp:
            try:
                # Processa texto em lotes se for muito grande
                if len(texto) > 500000:  # 500KB
                    nomes_spacy = set(self._processar_texto_grande(texto))
                else:
                    doc = self.nlp(texto)
                    for ent in doc.ents:
                        if ent.label_ in ["PER", "PERSON"]:
                            nome = ent.text.strip()
                            if debug:
                                print(f"🔍 spaCy detectou: '{nome}' (label: {ent.label_})")
                            
                            if self._validar_nome(nome):
                                nomes_spacy.add(nome)
                                if debug:
                                    print(f" spaCy ACEITO: '{nome}'")
                            else:
                                if debug:
                                    print(f"spaCy REJEITADO: '{nome}' - {self._obter_motivo_rejeicao(nome)}")
                
                if debug:
                    print(f"📊 spaCy encontrou: {len(nomes_spacy)} nomes")
                    
            except Exception as e:
                print(f"Erro ao extrair nomes com spaCy: {e}")
        
        # MÉTODO 2: Regex robusto (sempre executa para complementar)
        nomes_regex = set(self._extrair_nomes_regex_melhorado(texto, debug))
        
        if debug:
            print(f"📊 Regex encontrou: {len(nomes_regex)} nomes")
        
        # COMBINA os resultados (união dos dois métodos)
        todos_nomes = nomes_spacy.union(nomes_regex)
        
        if debug:
            print(f"📊 TOTAL COMBINADO: {len(todos_nomes)} nomes únicos")
            if todos_nomes:
                print(f"   Nomes finais: {sorted(list(todos_nomes))}")
        
        return list(todos_nomes)
    
    def _extrair_nomes_regex_melhorado(self, texto: str, debug=False) -> List[str]:
        """Método melhorado usando regex para capturar nomes que o spaCy pode perder"""
        nomes_encontrados = set()
        
        # PADRÃO 1: Nomes próprios normais (primeira letra maiúscula)
        # Ex: João Silva, Maria de Souza, Carlos Eduardo
        padrao1 = r'\b[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ][a-záàâãéèêíìôõóòúç]+(?:\s+(?:da|de|do|dos|das|e)\s+)?(?:\s+[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ][a-záàâãéèêíìôõóòúç]+)+\b'
        nomes1 = re.findall(padrao1, texto)
        
        # PADRÃO 2: Nomes em CAIXA ALTA (muito comum em documentos jurídicos)
        # Ex: JOÃO SILVA, MARIA DE SOUZA, DEIVIDY SANTOS
        padrao2 = r'\b[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ]{2,}(?:\s+(?:DA|DE|DO|DOS|DAS|E)\s+)?(?:\s+[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ]{2,})+\b'
        nomes2 = re.findall(padrao2, texto)
        
        # PADRÃO 3: Nomes com primeira e última palavra maiúscula (padrão misto)
        # Ex: JOÃO da Silva, MARIA de Souza
        padrao3 = r'\b[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ]{2,}\s+(?:da|de|do|dos|das|e\s+)?[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ]{2,}\b'
        nomes3 = re.findall(padrao3, texto)
        
        # PADRÃO 4: Nomes únicos mais longos (que podem ser apelidos/nomes não convencionais)
        # Ex: Deividy, Thallysson, Wyllyams
        padrao4 = r'\b[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ][a-záàâãéèêíìôõóòúçyYwW]{4,}\b'
        nomes4 = re.findall(padrao4, texto)
        
        # PADRÃO 5: Nomes seguidos de vírgula ou ponto (comum em listas)
        # Ex: "João Silva," ou "Maria Santos."
        padrao5 = r'\b[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ][a-záàâãéèêíìôõóòúç]+(?:\s+[A-ZÁÀÂÃÉÈÊÍÌÔÕÓÒÚÇ][a-záàâãéèêíìôõóòúç]+)+(?=[,\.])'
        nomes5 = re.findall(padrao5, texto)
        
        # Combina todos os padrões
        todos_padroes = nomes1 + nomes2 + nomes3 + nomes4 + nomes5
        
        if debug:
            print(f"🔍 REGEX - Padrão 1 (normais): {len(nomes1)} - {nomes1[:3] if nomes1 else []}")
            print(f"🔍 REGEX - Padrão 2 (MAIÚSCULA): {len(nomes2)} - {nomes2[:3] if nomes2 else []}")
            print(f"🔍 REGEX - Padrão 3 (misto): {len(nomes3)} - {nomes3[:3] if nomes3 else []}")
            print(f"🔍 REGEX - Padrão 4 (únicos): {len(nomes4)} - {nomes4[:3] if nomes4 else []}")
            print(f"🔍 REGEX - Padrão 5 (pontuados): {len(nomes5)} - {nomes5[:3] if nomes5 else []}")
        
        # Valida cada nome encontrado
        for nome in todos_padroes:
            nome_limpo = nome.strip()
            if self._validar_nome_melhorado(nome_limpo, debug):
                nomes_encontrados.add(nome_limpo)
                if debug:
                    print(f" REGEX ACEITO: '{nome_limpo}'")
            else:
                if debug:
                    print(f"REGEX REJEITADO: '{nome_limpo}' - {self._obter_motivo_rejeicao(nome_limpo)}")
        
        return list(nomes_encontrados)
    
    def _validar_nome_melhorado(self, nome: str, debug=False) -> bool:
        """Validação melhorada para nomes, incluindo casos especiais"""
        if not nome or len(nome) < 2:
            return False
        
        # Normaliza para comparação
        nome_normalizado = self.normalizar(nome)
        nome_upper = nome.upper()
        
        # FILTRO 1: Palavras descartadas básicas
        if nome_upper in self.palavras_descartadas or nome_normalizado in self.palavras_descartadas:
            if debug:
                print(f"   Rejeitado por palavra descartada: {nome}")
            return False
        
        # FILTRO 2: Para nomes compostos, verifica cada palavra
        palavras_do_nome = nome.split()
        if len(palavras_do_nome) > 1:
            palavras_descartadas_count = 0
            for palavra in palavras_do_nome:
                palavra_upper = palavra.upper()
                palavra_norm = self.normalizar(palavra)
                if palavra_upper in self.palavras_descartadas or palavra_norm in self.palavras_descartadas:
                    palavras_descartadas_count += 1
            
            # Se mais de 50% das palavras são descartadas, rejeita
            if palavras_descartadas_count > len(palavras_do_nome) * 0.5:
                if debug:
                    print(f"   Rejeitado por muitas palavras descartadas: {palavras_descartadas_count}/{len(palavras_do_nome)}")
                return False
        
        # FILTRO 3: Padrões que não são nomes
        filtros_basicos = [
            lambda x: all(c in '.,;:-_()[]{}' for c in x),  # Apenas pontuação
            lambda x: re.match(r'^\d+$', x),  # Apenas números
            lambda x: re.match(r'^[IVX]+$', x.upper()),  # Números romanos
            lambda x: x.upper() in ['SIM', 'NÃO', 'NAO'],  # Respostas
            lambda x: re.match(r'.*@.*', x),  # Emails
            lambda x: re.match(r'^www\.', x, re.IGNORECASE),  # URLs
            lambda x: re.match(r'^http', x, re.IGNORECASE),  # URLs
            lambda x: re.match(r'^\d+[A-Z]*$', x),  # Números com letras
            lambda x: len([c for c in x if c.isdigit()]) > len(x) * 0.7,  # Muitos números
        ]
        
        for filtro in filtros_basicos:
            if filtro(nome):
                if debug:
                    print(f"   Rejeitado por filtro básico: {nome}")
                return False
        
        # FILTRO 4: Siglas muito longas (mas permite nomes como "DA SILVA")
        if (len(nome) >= 6 and  # Aumentei para 6 para permitir nomes como "SILVA"
            nome.isupper() and 
            not any(c.islower() for c in nome) and
            nome.count('.') == 0 and
            nome.count(' ') == 0):  # Siglas longas SEM espaços
            if debug:
                print(f"   Rejeitado por ser sigla longa: {nome}")
            return False
        
        # FILTRO 5: Palavras muito comuns em documentos jurídicos (adicional)
        palavras_juridicas_extras = {
            'AUTOS', 'FOLHA', 'FOLHAS', 'PÁGINA', 'PAGINA', 'PÁGINAS', 'PAGINAS',
            'VERSO', 'FRENTE', 'DOCUMENTO', 'ANEXO', 'ANEXOS', 'CÓPIA', 'COPIA',
            'ORIGINAL', 'PROTOCOLO', 'NÚMERO', 'NUMERO', 'DATA', 'HORA',
            'LOCAL', 'ASSINATURA', 'RUBRICA', 'CARIMBO', 'SELO'
        }
        
        if nome_upper in palavras_juridicas_extras:
            if debug:
                print(f"   Rejeitado por termo jurídico: {nome}")
            return False
        
        # FILTRO 6: Aceita nomes que passaram em todos os filtros
        # MAS adiciona verificação especial para nomes únicos muito curtos
        if len(palavras_do_nome) == 1 and len(nome) < 4:  # Nome único muito curto
            if debug:
                print(f"   Rejeitado por ser muito curto: {nome}")
            return False
        
        return True
    
    def _validar_nome(self, nome: str) -> bool:
        """Valida se um nome deve ser considerado para anonimização - usa versão melhorada"""
        return self._validar_nome_melhorado(nome, debug=False)
    
    def _obter_motivo_rejeicao(self, nome: str) -> str:
        """Retorna o motivo específico da rejeição de um nome"""
        if not nome or len(nome) < 2:
            return "muito curto (< 2 chars)"
        
        nome_normalizado = self.normalizar(nome)
        nome_upper = nome.upper()
        
        # Verifica palavra descartada
        if nome_upper in self.palavras_descartadas or nome_normalizado in self.palavras_descartadas:
            return "palavra descartada"
        
        # Verifica palavras compostas
        palavras_do_nome = nome.split()
        if len(palavras_do_nome) > 1:
            palavras_descartadas_count = 0
            for palavra in palavras_do_nome:
                palavra_upper = palavra.upper()
                palavra_norm = self.normalizar(palavra)
                if palavra_upper in self.palavras_descartadas or palavra_norm in self.palavras_descartadas:
                    palavras_descartadas_count += 1
            
            if palavras_descartadas_count > len(palavras_do_nome) * 0.5:
                return f"muitas palavras descartadas ({palavras_descartadas_count}/{len(palavras_do_nome)})"
        
        # Verifica filtros específicos
        if all(c in '.,;:-_()[]{}' for c in nome):
            return "apenas pontuação"
        if re.match(r'^\d+$', nome):
            return "apenas números"
        if re.match(r'^[IVX]+$', nome.upper()):
            return "números romanos"
        if nome.upper() in ['SIM', 'NÃO', 'NAO']:
            return "resposta comum"
        if re.match(r'.*@.*', nome):
            return "formato de email"
        if re.match(r'^www\.', nome, re.IGNORECASE):
            return "URL (www)"
        if re.match(r'^http', nome, re.IGNORECASE):
            return "URL (http)"
        if re.match(r'^\d+[A-Z]*$', nome):
            return "número com letras"
        if len([c for c in nome if c.isdigit()]) > len(nome) * 0.7:
            return "muitos números"
        
        # Verifica siglas longas
        if (len(nome) >= 6 and nome.isupper() and not any(c.islower() for c in nome) 
            and nome.count('.') == 0 and nome.count(' ') == 0):
            return "sigla longa"
        
        # Verifica palavras jurídicas extras
        palavras_juridicas_extras = {
            'AUTOS', 'FOLHA', 'FOLHAS', 'PÁGINA', 'PAGINA', 'PÁGINAS', 'PAGINAS',
            'VERSO', 'FRENTE', 'DOCUMENTO', 'ANEXO', 'ANEXOS', 'CÓPIA', 'COPIA',
            'ORIGINAL', 'PROTOCOLO', 'NÚMERO', 'NUMERO', 'DATA', 'HORA'
        }
        if nome_upper in palavras_juridicas_extras:
            return "termo jurídico"
        
        # Verifica se é muito curto para nome único
        if len(palavras_do_nome) == 1 and len(nome) < 4:
            return "nome único muito curto"
        
        return "filtro desconhecido"
    
    def anonimizar_texto_otimizado(self, texto: str) -> str:
        """Anonimiza padrões usando regex compilados"""
        substituicoes = {
            'cpf': '[CPF_REMOVIDO]',
            'rg': '[RG_REMOVIDO]',
            'cnpj': '[CNPJ_REMOVIDO]',
            'data': '[DATA_REMOVIDA]',
            'data_iso': '[DATA_REMOVIDA]',
            'email': '[EMAIL_REMOVIDO]',
            'cep': '[CEP_REMOVIDO]',
            'telefone': '[TELEFONE_REMOVIDO]',
            'processo': '[PROCESSO_REMOVIDO]'
        }
        
        for chave, substituto in substituicoes.items():
            texto = self.padroes_regex[chave].sub(substituto, texto)
        
        return texto
    
    def anonimizar_com_identificadores(self, texto: str, mapa_suspeitos: Dict, debug=False) -> Tuple[str, Dict]:
        """Anonimiza texto com identificadores otimizado"""
        
        print(f"🔒 Iniciando anonimização do texto...")
        
        # PRIMEIRO: Anonimiza padrões (CPF, RG, emails, etc.) ANTES de processar nomes
        texto_com_padroes = self.anonimizar_texto_otimizado(texto)
        
        # SEGUNDO: Extrai e processa nomes
        nomes = self.extrair_nomes_spacy_otimizado(texto, debug=debug)
        reverso = {}
        substituidos = set()
        contador = 1
        
        print(f"📝 Nomes detectados: {len(nomes)}")
        if debug and nomes:
            print(f"   Nomes: {nomes}")
        
        # TERCEIRO: Processa suspeitos conhecidos
        for nome in nomes:
            nome_norm = self.normalizar(nome)
            if nome_norm in mapa_suspeitos:
                ident, nome_real = mapa_suspeitos[nome_norm]
                if ident not in reverso:
                    reverso[ident] = nome_real
                
                # Substitui no texto
                padrao = re.compile(rf'\b{re.escape(nome)}\b', flags=re.IGNORECASE)
                texto_com_padroes, n = padrao.subn(ident, texto_com_padroes)
                if n > 0:
                    substituidos.add(nome)
                    print(f"🎯 SUSPEITO: {nome} → {ident} ({n}x)")
        
        # QUARTO: Processa nomes comuns
        for nome in nomes:
            if nome in substituidos:
                continue
            
            ident = f"PESSOA_{contador:03d}"
            padrao = re.compile(rf'\b{re.escape(nome)}\b', flags=re.IGNORECASE)
            texto_com_padroes, n = padrao.subn(ident, texto_com_padroes)
            if n > 0:
                reverso[ident] = nome
                print(f"👤 NOME: {nome} → {ident} ({n}x)")
                contador += 1
        
        print(f" Anonimização concluída: {len(reverso)} substituições")
        
        return texto_com_padroes, reverso
    
    def _processar_texto_grande(self, texto: str) -> List[str]:
        """Processa textos grandes em chunks para evitar problemas de memória"""
        chunk_size = 100000  # 100KB por chunk
        chunks = [texto[i:i+chunk_size] for i in range(0, len(texto), chunk_size)]
        
        todos_nomes = set()
        
        for chunk in chunks:
            try:
                doc = self.nlp(chunk)
                for ent in doc.ents:
                    if ent.label_ in ["PER", "PERSON"]:
                        nome = ent.text.strip()
                        if self._validar_nome(nome):
                            todos_nomes.add(nome)
                
                # Força garbage collection entre chunks
                del doc
                gc.collect()
                
            except Exception as e:
                print(f"Erro ao processar chunk: {e}")
                continue
        
        return list(todos_nomes)