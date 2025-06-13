# anonimiza_md_otimizado.py
import os
import re
import unicodedata
from dotenv import load_dotenv
from pathlib import Path
import spacy
from functools import lru_cache
from typing import Dict, Tuple, Set, List
import gc

# === Carrega .env ===
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
PASTA_DESTINO = os.getenv("PASTA_DESTINO", ".")

class AnonimizadorOtimizado:
    def __init__(self, caminho_palavras_descartadas="palavras_descartadas.txt"):
        # Carrega modelo spaCy apenas uma vez com componentes otimizados
        self.nlp = self._carregar_modelo_otimizado()
        
        # Cache para normalizações e mapeamentos
        self.cache_normalizacao = {}
        self.cache_suspeitos = None
        
        # Palavras descartadas carregadas do arquivo txt
        self.palavras_descartadas = self._carregar_palavras_descartadas(caminho_palavras_descartadas)
        
        # Padrões regex compilados para melhor performance
        self.padroes_regex = self._compilar_padroes()
    
    def _carregar_modelo_otimizado(self):
        """Carrega o modelo spaCy com configurações otimizadas"""
        try:
            # Desabilita componentes desnecessários para performance
            nlp = spacy.load("pt_core_news_sm", 
                           disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
            
            # Aumenta limite para textos grandes
            nlp.max_length = 2_000_000
            
            # Adiciona EntityRuler para padrões específicos
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "CPF", "pattern": [{"TEXT": {"REGEX": r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"}}]},
                {"label": "RG", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}\.?\d{3}\.?\d{3}[-\.]?\d{1,2}"}}]},
                {"label": "CNPJ", "pattern": [{"TEXT": {"REGEX": r"\d{2}\.?\d{3}\.?\d{3}[\/\.]?\d{4}-?\d{2}"}}]},
                {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]},
                {"label": "TELEFONE", "pattern": [{"TEXT": {"REGEX": r"(?:\(?\d{2}\)?\s?)?(?:9?\d{4,5})[-\.\s]?\d{4}"}}]}
            ]
            ruler.add_patterns(patterns)
            
            return nlp
        except Exception as e:
            print(f"Erro ao carregar modelo spaCy: {e}")
            return None
    
    @lru_cache(maxsize=2000)
    def normalizar(self, texto: str) -> str:
        """Normaliza texto com cache para evitar reprocessamento"""
        if not texto:
            return ""
        return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode().lower().strip()
    
    def _carregar_palavras_descartadas(self, caminho="palavras_descartadas.txt") -> Set[str]:
        """Carrega palavras descartadas de arquivo txt como set para busca rápida"""
        palavras = set()
        
        # Debug: mostra caminho absoluto e verifica se existe
        caminho_absoluto = os.path.abspath(caminho)
        print(f"🔍 DEBUG: Tentando carregar palavras de: {caminho_absoluto}")
        print(f"🔍 DEBUG: Arquivo existe? {os.path.exists(caminho_absoluto)}")
        
        # Tenta diferentes caminhos possíveis
        caminhos_possiveis = [
            caminho,
            os.path.join("utils", "palavras_descartadas.txt"),
            os.path.join(".", "palavras_descartadas.txt"),
            "palavras_descartadas.txt"
        ]
        
        arquivo_encontrado = None
        for caminho_teste in caminhos_possiveis:
            if os.path.exists(caminho_teste):
                arquivo_encontrado = caminho_teste
                print(f"✓ Arquivo encontrado em: {os.path.abspath(caminho_teste)}")
                break
        
        if not arquivo_encontrado:
            print(f"❌ Arquivo não encontrado em nenhum dos caminhos:")
            for c in caminhos_possiveis:
                print(f"   - {os.path.abspath(c)}")
            print(f"⚠️ Usando lista padrão básica.")
            
            # Lista mínima de fallback
            palavras = {
                'E', 'EM', 'NO', 'NA', 'DOS', 'DAS', 'DE', 'DO', 'DA', 'AOS', 'AO',
                'COM', 'SEM', 'POR', 'PARA', 'ANTE', 'APÓS', 'APOS', 'ATÉ', 'ATE',
                'CONTRA', 'DESDE', 'ENTRE', 'PERANTE', 'SEGUNDO', 'SOBRE', 'CONFORME',
                'TRIBUNAL', 'VARA', 'PROCESSO', 'JUIZ', 'JUÍZA', 'DOUTOR', 'DOUTORA'
            }
            return palavras
        
        try:
            total_linhas = 0
            linhas_vazias = 0
            comentarios = 0
            palavras_adicionadas = 0
            
            with open(arquivo_encontrado, "r", encoding="utf-8") as f:
                for numero_linha, linha in enumerate(f, 1):
                    total_linhas += 1
                    linha_original = linha
                    
                    # Remove espaços e quebras de linha
                    palavra = linha.strip().upper()
                    
                    # Debug das primeiras 5 linhas
                    if numero_linha <= 5:
                        print(f"🔍 Linha {numero_linha}: '{linha_original.strip()}' -> '{palavra}'")
                    
                    # Verifica se é linha vazia
                    if not palavra:
                        linhas_vazias += 1
                        continue
                    
                    # Verifica se é comentário
                    if palavra.startswith('#'):
                        comentarios += 1
                        continue
                    
                    # Adiciona palavra em maiúscula E sua versão normalizada
                    palavras.add(palavra)
                    palavras.add(self.normalizar(palavra))
                    palavras_adicionadas += 1
            
            print(f"📊 Estatísticas do arquivo:")
            print(f"   - Total de linhas: {total_linhas}")
            print(f"   - Linhas vazias: {linhas_vazias}")
            print(f"   - Comentários: {comentarios}")
            print(f"   - Palavras processadas: {palavras_adicionadas}")
            print(f"✓ Set final com {len(palavras)} entradas (incluindo versões normalizadas)")
            
            # Mostra algumas palavras como exemplo
            if palavras:
                exemplo_palavras = sorted(list(palavras))[:10]
                print(f"📝 Exemplos: {', '.join(exemplo_palavras)}")
            
        except Exception as e:
            print(f"❌ Erro ao ler arquivo {arquivo_encontrado}: {e}")
            import traceback
            traceback.print_exc()
            
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
    
    def carregar_suspeitos_mapeados(self, caminho="suspeitos.txt") -> Dict[str, Tuple[str, str]]:
        """Carrega mapeamento de suspeitos com cache"""
        if self.cache_suspeitos is not None:
            return self.cache_suspeitos
        
        mapa = {}
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                for linha in f:
                    if "|" in linha:
                        ident, nome = linha.strip().split("|", 1)
                        partes = nome.strip().split()
                        if len(partes) >= 2:
                            chave_nome_completo = self.normalizar(nome)
                            chave_nome_sobrenome = self.normalizar(f"{partes[0]} {partes[-1]}")
                            mapa[chave_nome_completo] = (ident, nome)
                            mapa[chave_nome_sobrenome] = (ident, nome)
        except FileNotFoundError:
            print(f"Arquivo {caminho} não encontrado. Continuando sem lista de suspeitos.")
        
        self.cache_suspeitos = mapa
        return mapa
    
    def extrair_nomes_spacy_otimizado(self, texto: str, debug=False) -> List[str]:
        """Extrai nomes usando spaCy com otimizações"""
        if not self.nlp:
            return []
        
        try:
            # Processa texto em lotes se for muito grande
            if len(texto) > 500000:  # 500KB
                return self._processar_texto_grande(texto)
            
            doc = self.nlp(texto)
            nomes = set()  # Usa set para eliminar duplicatas automaticamente
            nomes_rejeitados = set()  # Para debug
            
            for ent in doc.ents:
                if ent.label_ in ["PER", "PERSON"]:  # Inclui ambas as variações
                    nome = ent.text.strip()
                    
                    if debug:
                        print(f"🔍 spaCy detectou: '{nome}' (label: {ent.label_})")
                    
                    # Filtros otimizados
                    if self._validar_nome(nome):
                        nomes.add(nome)
                        if debug:
                            print(f"✅ ACEITO: '{nome}'")
                    else:
                        nomes_rejeitados.add(nome)
                        if debug:
                            print(f"❌ REJEITADO: '{nome}' - Motivo: {self._obter_motivo_rejeicao(nome)}")
            
            if debug:
                print(f"\n📊 RESUMO:")
                print(f"   - Nomes aceitos: {len(nomes)}")
                print(f"   - Nomes rejeitados: {len(nomes_rejeitados)}")
                if nomes:
                    print(f"   - Aceitos: {sorted(list(nomes))}")
                if nomes_rejeitados:
                    print(f"   - Rejeitados: {sorted(list(nomes_rejeitados))}")
            
            return list(nomes)
        
        except Exception as e:
            print(f"Erro ao extrair nomes: {e}")
            return []
    
    def _obter_motivo_rejeicao(self, nome: str) -> str:
        """Retorna o motivo da rejeição de um nome (para debug)"""
        if not nome or len(nome) < 2:
            return "muito curto"
        
        nome_normalizado = self.normalizar(nome)
        nome_upper = nome.upper()
        
        if nome_upper in self.palavras_descartadas or nome_normalizado in self.palavras_descartadas:
            return "palavra descartada"
        
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
        
        if all(c in '.,;:-_()[]{}' for c in nome):
            return "apenas pontuação"
        if re.match(r'^\d+$', nome):
            return "apenas números"
        if re.match(r'^[IVX]+$', nome.upper()):
            return "números romanos"
        if nome.upper() in ['SIM', 'NÃO', 'NAO']:
            return "resposta comum"
        if re.match(r'.*@.*', nome):
            return "email"
        if re.match(r'^www\.', nome, re.IGNORECASE):
            return "URL"
        if re.match(r'^http', nome, re.IGNORECASE):
            return "URL"
        if re.match(r'^\d+[A-Z]*$', nome):
            return "número com letras"
        if len([c for c in nome if c.isdigit()]) > len(nome) * 0.7:
            return "muitos números"
        if (len(nome) >= 4 and nome.isupper() and not any(c.islower() for c in nome) and nome.count('.') == 0):
            return "sigla longa"
        
        return "filtro desconhecido"
    
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
    
    def _validar_nome(self, nome: str) -> bool:
        """Valida se um nome deve ser considerado para anonimização"""
        if not nome or len(nome) < 2:  # Relaxa para 2 caracteres
            return False
        
        # Normaliza para comparação (sem acentos, minúsculo)
        nome_normalizado = self.normalizar(nome)
        nome_upper = nome.upper()
        
        # Verifica se é palavra descartada (usando ambas as formas)
        if nome_upper in self.palavras_descartadas or nome_normalizado in self.palavras_descartadas:
            return False
        
        # Verifica cada palavra individualmente se for nome composto
        palavras_do_nome = nome.split()
        if len(palavras_do_nome) > 1:
            # Se MAIS DE 50% das palavras estão na lista de descartadas, rejeita
            palavras_descartadas_count = 0
            for palavra in palavras_do_nome:
                palavra_upper = palavra.upper()
                palavra_norm = self.normalizar(palavra)
                if palavra_upper in self.palavras_descartadas or palavra_norm in self.palavras_descartadas:
                    palavras_descartadas_count += 1
            
            # Se mais de 50% das palavras são descartadas, rejeita o nome todo
            if palavras_descartadas_count > len(palavras_do_nome) * 0.5:
                return False
        
        # Filtros para padrões que definitivamente não são nomes
        if (all(c in '.,;:-_()[]{}' for c in nome) or  # Apenas pontuação
            re.match(r'^\d+$', nome) or  # Apenas números
            re.match(r'^[IVX]+$', nome.upper()) or  # Números romanos puros
            nome.upper() in ['SIM', 'NÃO', 'NAO'] or  # Respostas comuns
            re.match(r'.*@.*', nome) or  # Emails
            re.match(r'^www\.', nome, re.IGNORECASE) or  # URLs
            re.match(r'^http', nome, re.IGNORECASE) or  # URLs
            re.match(r'^\d+[A-Z]*$', nome) or  # Números com letras (1A, 2B)
            len([c for c in nome if c.isdigit()]) > len(nome) * 0.7):  # Muito número no nome
            return False
        
        # Aceita siglas curtas que podem ser nomes/sobrenomes (ex: DA, DE, etc. já estão na lista)
        # Mas rejeita siglas muito longas em maiúscula que claramente não são nomes
        if (len(nome) >= 4 and 
            nome.isupper() and 
            not any(c.islower() for c in nome) and
            nome.count('.') == 0):  # Siglas longas sem pontos
            return False
        
        return True

    def anonimizar_texto_otimizado(self, texto: str) -> str:
        """Anonimiza padrões usando regex compilados"""
        substituicoes = {
            'cpf': '[CPF]',
            'rg': '[RG]',
            'cnpj': '[CNPJ]',
            'data': '[DATA]',
            'data_iso': '[DATA_ISO]',
            'email': '[EMAIL]',
            'cep': '[CEP]',
            'telefone': '[TELEFONE]',
            'processo': '[PROCESSO]'
        }
        
        for chave, substituto in substituicoes.items():
            texto = self.padroes_regex[chave].sub(substituto, texto)
        
        return texto
    
    def anonimizar_com_identificadores(self, texto: str, mapa_suspeitos: Dict, debug=False) -> Tuple[str, Dict]:
        """Anonimiza texto com identificadores otimizado"""
        
        # PRIMEIRO: Anonimiza padrões (CPF, RG, emails, etc.) ANTES de processar nomes
        if debug:
            print(f"🔍 TEXTO ORIGINAL (primeiros 200 chars):\n{texto[:200]}...")
        
        texto_com_padroes = self.anonimizar_texto_otimizado(texto)
        
        if debug:
            print(f"🔍 APÓS ANONIMIZAR PADRÕES:\n{texto_com_padroes[:200]}...")
        
        # SEGUNDO: Extrai e processa nomes
        nomes = self.extrair_nomes_spacy_otimizado(texto, debug=debug)
        reverso = {}
        substituidos = set()
        contador = 1
        
        if debug:
            print(f"\n🔍 INICIANDO ANONIMIZAÇÃO DE NOMES:")
            print(f"   - Total de nomes detectados: {len(nomes)}")
            print(f"   - Nomes: {nomes}")
        
        # Terceiro passa: suspeitos conhecidos
        for nome in nomes:
            nome_norm = self.normalizar(nome)
            if nome_norm in mapa_suspeitos:
                ident, nome_real = mapa_suspeitos[nome_norm]
                if ident not in reverso:
                    reverso[ident] = nome_real
                
                # Usa regex compilado para substituição mais eficiente
                padrao = re.compile(rf'\b{re.escape(nome)}\b', flags=re.IGNORECASE)
                texto_com_padroes, n = padrao.subn(f"{nome_real} ({ident})", texto_com_padroes)
                if n > 0:
                    substituidos.add(nome)
                    print(f"SUSPEITO: {nome} → {ident} ({n}x)")
        
        # Quarta passa: nomes comuns
        for nome in nomes:
            if nome in substituidos:
                continue
            
            ident = f"#NOME_{contador:03}"
            padrao = re.compile(rf'\b{re.escape(nome)}\b', flags=re.IGNORECASE)
            texto_com_padroes, n = padrao.subn(ident, texto_com_padroes)
            if n > 0:
                reverso[ident] = nome
                print(f"Nome comum: {nome} → {ident} ({n}x)")
                contador += 1
        
        return texto_com_padroes, reverso
    
    def processar_lote(self, textos: List[str], mapa_suspeitos: Dict) -> List[Tuple[str, Dict]]:
        """Processa múltiplos textos em lote para melhor performance"""
        resultados = []
        
        for i, texto in enumerate(textos):
            try:
                resultado = self.anonimizar_com_identificadores(texto, mapa_suspeitos)
                resultados.append(resultado)
                
                if i % 10 == 0:  # Feedback a cada 10 textos
                    print(f"Processados {i+1}/{len(textos)} textos")
                
            except Exception as e:
                print(f"Erro ao processar texto {i}: {e}")
                resultados.append((texto, {}))  # Retorna original em caso de erro
        
        return resultados

def salvar_anonimizacao_md_otimizada(conteudo_md: str, nome_base: str, 
                                   anonimizador: AnonimizadorOtimizado = None,
                                   caminho_palavras="palavras_descartadas.txt"):
    """Salva anonimização usando a classe otimizada"""
    print(f"Usando PASTA_DESTINO: {PASTA_DESTINO}")
    pasta_anon = os.path.join(PASTA_DESTINO, "anonimizados")
    pasta_mapas = os.path.join(PASTA_DESTINO, "mapas")
    os.makedirs(pasta_anon, exist_ok=True)
    os.makedirs(pasta_mapas, exist_ok=True)
    
    # Cria anonimizador se não foi fornecido
    if anonimizador is None:
        anonimizador = AnonimizadorOtimizado(caminho_palavras)
    
    mapa_suspeitos = anonimizador.carregar_suspeitos_mapeados("suspeitos.txt")
    texto_anon, mapa_reverso = anonimizador.anonimizar_com_identificadores(conteudo_md, mapa_suspeitos)
    
    caminho_md = os.path.join(pasta_anon, f"{nome_base}_anon.md")
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(texto_anon)
    
    if mapa_reverso:
        caminho_mapa = os.path.join(pasta_mapas, f"{nome_base}_mapa.md")
        with open(caminho_mapa, "w", encoding="utf-8") as f:
            f.write("| Identificador | Nome Original |\n")
            f.write("|---------------|----------------|\n")
            for ident, nome in sorted(mapa_reverso.items()):
                f.write(f"| {ident} | {nome} |\n")
    
    print(f"Arquivos salvos:\n- {caminho_md}\n- {caminho_mapa if mapa_reverso else '(sem mapa)'}")

# === Exemplo de uso otimizado ===
if __name__ == "__main__":
    # Instancia o anonimizador uma vez (especificando arquivo de palavras se necessário)
    anonimizador = AnonimizadorOtimizado("palavras_descartadas.txt")
    
    markdown_exemplo = """
    # Relatório
    
    Participaram da reunião: JOÃO DA SILVA, Maria Lima, Fernanda dos Santos, Dr. Roberto.
    O TRIBUNAL decidiu que o PROCESSO deve ser analisado.
    Contato: maria@email.com, telefone: (11) 99999-9999
    CPF: 123.456.789-00, RG: 12.345.678-9
    Data de nascimento: 15/03/1985
    CEP: 01234-567
    
    CARLOS EDUARDO PEREIRA compareceu na audiência.
    MARIA JOSÉ SANTOS apresentou a defesa.
    O JUIZ determinou que a VARA deve proceder.
    
    Processo nº: 1234567-89.2023.8.26.0001
    """
    
    print("=== TESTE COM DEBUG ===")
    mapa_suspeitos = anonimizador.carregar_suspeitos_mapeados("suspeitos.txt")
    texto_anon, mapa_reverso = anonimizador.anonimizar_com_identificadores(
        markdown_exemplo, mapa_suspeitos, debug=True
    )
    
    print(f"\n=== RESULTADO ===")
    print(f"Texto original:\n{markdown_exemplo}")
    print(f"\nTexto anonimizado:\n{texto_anon}")
    print(f"\nMapeamento: {mapa_reverso}")
    
    # Processa um único texto (sem debug)
    print(f"\n=== PROCESSAMENTO NORMAL ===")
    salvar_anonimizacao_md_otimizada(markdown_exemplo, "relatorio_teste", anonimizador)
    
    # Exemplo de processamento em lote
    textos_exemplo = [markdown_exemplo] * 3  # Simula 3 textos
    resultados_lote = anonimizador.processar_lote(textos_exemplo, mapa_suspeitos)
    
    print(f"Processados {len(resultados_lote)} textos em lote")