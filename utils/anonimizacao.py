# anonimiza_md.py
import os
import re
import unicodedata
from dotenv import load_dotenv
from pathlib import Path
import spacy

# === Carrega .env ===
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
PASTA_DESTINO = os.getenv("PASTA_DESTINO", ".")

# === Carrega modelo spaCy uma vez ===
nlp = spacy.load("pt_core_news_sm")
nlp.max_length = 2_000_000

def normalizar(texto: str) -> str:
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode().lower().strip()

def carregar_suspeitos_mapeados(caminho="suspeitos.txt") -> dict:
    mapa = {}
    with open(caminho, "r", encoding="utf-8") as f:
        for linha in f:
            if "|" in linha:
                ident, nome = linha.strip().split("|", 1)
                partes = nome.strip().split()
                if len(partes) >= 2:
                    chave_nome_completo = normalizar(nome)
                    chave_nome_sobrenome = normalizar(f"{partes[0]} {partes[-1]}")
                    mapa[chave_nome_completo] = (ident, nome)
                    mapa[chave_nome_sobrenome] = (ident, nome)
    return mapa

# === Nova função com spaCy ===
def extrair_nomes_spacy(texto: str):
    # Palavras a serem descartadas (mesmo conjunto da lista completa)
    palavras_descartadas = {
        # Palavras originais
        'E', 'EM', 'NO', 'NA', 'DOS', 'DAS', 'DE', 'DO', 'DA', 'AOS', 'AO',
        
        # Termos institucionais
        'GABINETE', 'CÂMARA', 'CAMARA',
        
        # Documentos e procedimentos
        'ATA', 'ATAS',
        'PLANOS', 'PLANO',
        'MATÉRIAS', 'MATERIAS', 'MATÉRIA', 'MATERIA',
        'CONTRARRAZÕES', 'CONTRARRAZOES', 'CONTRARRAZÃO', 'CONTRARRAZAO',
        'DOCUMENTO', 'DOCUMENTOS',
        'JUNTADA', 'JUNTADAS',
        'DECISÃO', 'DECISAO', 'DECISÕES', 'DECISOES',
        
        # Status e observações
        'RESSALVADO', 'RESSALVADA', 'RESSALVADOS', 'RESSALVADAS',
        
        # Links e URLs
        'HTTPS', 'HTTP', 'WWW', 'COM', 'BR', 'ORG', 'GOV',
        
        # Tratamentos e cargos
        'SRA', 'SR', 'SRS', 'SRAS',
        'SENHOR', 'SENHORA', 'SENHORES', 'SENHORAS',
        'DESEMBARGADOR', 'DESEMBARGADORA', 'DESEMBARGADORES', 'DESEMBARGADORAS',
        'JUIZ', 'JUÍZA', 'JUIZA', 'JUÍZES', 'JUIZES', 'JUÍZAS', 'JUIZAS',
        'DIREITO',
        
        # Pronomes de tratamento específicos
        'V', 'EXA', 'EXÂ', 'EXCELÊNCIA', 'EXCELENCIA', 'EXCELENTÍSSIMO', 'EXCELENTISSIMO',
        'EXCELENTÍSSIMA', 'EXCELENTISSIMA', 'EXMO', 'EXMA', 'ILUSTRÍSSIMO', 'ILUSTRISSIMO',
        'ILUSTRÍSSIMA', 'ILUSTRISSIMA', 'ILMO', 'ILMA', 'MERITÍSSIMO', 'MERITISSIMO',
        'MERITÍSSIMA', 'MERITISSIMA', 'MM', 'MMO', 'MMA', 'DOUTOR', 'DOUTORA',
        'DR', 'DRA', 'PROFESSORA', 'PROFESSOR', 'PROF', 'PROFA',
        
        # Termos processuais comuns
        'PROCESSO', 'PROCESSOS',
        'RECURSO', 'RECURSOS',
        'APELAÇÃO', 'APELACAO', 'APELAÇÕES', 'APELACOES',
        'AGRAVO', 'AGRAVOS',
        'EMBARGOS', 'EMBARGO',
        'MANDADO', 'MANDADOS',
        'HABEAS', 'CORPUS',
        'SENTENÇA', 'SENTENCA', 'SENTENÇAS', 'SENTENCAS',
        'ACÓRDÃO', 'ACORDAO', 'ACÓRDÃOS', 'ACORDAOS',
        'DESPACHO', 'DESPACHOS',
        
        # Atos processuais e documentos
        'PETIÇÃO', 'PETICAO', 'PETIÇÕES', 'PETICOES', 'INICIAL',
        'CONTESTAÇÃO', 'CONTESTACAO', 'CONTESTAÇÕES', 'CONTESTACOES',
        'TRÉPLICA', 'TREPLICA', 'TRÉPLICAS', 'TREPLICAS',
        'DÚPLICA', 'DUPLICA', 'DÚPLICAS', 'DUPLICAS',
        'AUDIÊNCIA', 'AUDIENCIA', 'AUDIÊNCIAS', 'AUDIENCIAS',
        'INTIMAÇÃO', 'INTIMACAO', 'INTIMAÇÕES', 'INTIMACOES',
        'CITAÇÃO', 'CITACAO', 'CITAÇÕES', 'CITACOES',
        'NOTIFICAÇÃO', 'NOTIFICACAO', 'NOTIFICAÇÕES', 'NOTIFICACOES',
        'PUBLICAÇÃO', 'PUBLICACAO', 'PUBLICAÇÕES', 'PUBLICACOES',
        'DISTRIBUIÇÃO', 'DISTRIBUICAO', 'DISTRIBUIÇÕES', 'DISTRIBUICOES',
        'CONCLUSÃO', 'CONCLUSAO', 'CONCLUSÕES', 'CONCLUSOES',
        'REMESSA', 'REMESSAS', 'DEVOLUÇÃO', 'DEVOLUCAO', 'DEVOLUÇÕES', 'DEVOLUCOES',
        'BAIXA', 'BAIXAS', 'ARQUIVAMENTO', 'ARQUIVAMENTOS',
        'SUSPENSÃO', 'SUSPENSAO', 'SUSPENSÕES', 'SUSPENSOES',
        'SOBRESTAMENTO', 'SOBRESTAMENTOS',
        
        # Termos de direito processual
        'AÇÃO', 'ACAO', 'AÇÕES', 'ACOES',
        'AUTOR', 'AUTORA', 'AUTORES', 'AUTORAS',
        'RÉU', 'REU', 'RÉS', 'RES', 'RÉUS', 'REUS',
        'REQUERENTE', 'REQUERENTES', 'REQUERIDO', 'REQUERIDA', 'REQUERIDOS', 'REQUERIDAS',
        'APELANTE', 'APELANTES', 'APELADO', 'APELADA', 'APELADOS', 'APELADAS',
        'AGRAVANTE', 'AGRAVANTES', 'AGRAVADO', 'AGRAVADA', 'AGRAVADOS', 'AGRAVADAS',
        'EMBARGANTE', 'EMBARGANTES', 'EMBARGADO', 'EMBARGADA', 'EMBARGADOS', 'EMBARGADAS',
        'RECORRENTE', 'RECORRENTES', 'RECORRIDO', 'RECORRIDA', 'RECORRIDOS', 'RECORRIDAS',
        'IMPETRANTE', 'IMPETRANTES', 'IMPETRADO', 'IMPETRADA', 'IMPETRADOS', 'IMPETRADAS',
        'EXECUTADO', 'EXECUTADA', 'EXECUTADOS', 'EXECUTADAS',
        'EXEQUENTE', 'EXEQUENTES',
        'TERCEIRO', 'TERCEIRA', 'TERCEIROS', 'TERCEIRAS',
        'INTERESSADO', 'INTERESSADA', 'INTERESSADOS', 'INTERESSADAS',
        'ASSISTENTE', 'ASSISTENTES', 'OPOENTE', 'OPOENTES',
        'LITISCONSORTE', 'LITISCONSORTES',
        
        # Institutos jurídicos
        'COMPETÊNCIA', 'COMPETENCIA', 'INCOMPETÊNCIA', 'INCOMPETENCIA',
        'JURISDIÇÃO', 'JURISDICAO', 'JURISDIÇÕES', 'JURISDICOES',
        'LEGITIMIDADE', 'ILEGITIMIDADE',
        'INTERESSE', 'INTERESSES', 'POSSIBILIDADE', 'IMPOSSIBILIDADE',
        'PRECLUSÃO', 'PRECLUSAO', 'PRECLUSÕES', 'PRECLUSOES',
        'PRESCRIÇÃO', 'PRESCRICAO', 'PRESCRIÇÕES', 'PRESCRICOES',
        'DECADÊNCIA', 'DECADENCIA', 'DECADÊNCIAS', 'DECADENCIAS',
        'PREJUDICIAL', 'PREJUDICIAIS', 'PRELIMINAR', 'PRELIMINARES',
        'MÉRITO', 'MERITO', 'MÉRIOS', 'MERIOS',
        'PROVA', 'PROVAS', 'EVIDÊNCIA', 'EVIDENCIA', 'EVIDÊNCIAS', 'EVIDENCIAS',
        'TESTEMUNHA', 'TESTEMUNHAS', 'DEPOIMENTO', 'DEPOIMENTOS',
        'PERÍCIA', 'PERICIA', 'PERÍCIAS', 'PERICIAS', 'PERITO', 'PERITOS', 'PERITA', 'PERITAS',
        'LAUDO', 'LAUDOS', 'EXAME', 'EXAMES', 'VISTORIA', 'VISTORIAS',
        
        # Decisões e julgamentos
        'PROCEDENTE', 'IMPROCEDENTE', 'PARCIALMENTE',
        'DEFERIDO', 'INDEFERIDO', 'DEFERIR', 'INDEFERIR',
        'PROVIDO', 'DESPROVIDO', 'PROVER', 'DESPROVER',
        'CONHECIDO', 'DESCONHECIDO', 'CONHECER', 'DESCONHECER',
        'ACOLHIDO', 'REJEITADO', 'ACOLHER', 'REJEITAR',
        'CONCEDIDO', 'DENEGADO', 'CONCEDER', 'DENEGAR',
        'JULGADO', 'JULGADA', 'JULGADOS', 'JULGADAS', 'JULGAR',
        'DECIDIDO', 'DECIDIDA', 'DECIDIDOS', 'DECIDIDAS', 'DECIDIR',
        'RESOLVIDO', 'RESOLVIDA', 'RESOLVIDOS', 'RESOLVIDAS', 'RESOLVER',
        'EXTINTO', 'EXTINTA', 'EXTINTOS', 'EXTINTAS', 'EXTINÇÃO', 'EXTINCAO',
        
        # Fundamentos legais
        'LEI', 'LEIS', 'CÓDIGO', 'CODIGO', 'CÓDIGOS', 'CODIGOS',
        'CONSTITUIÇÃO', 'CONSTITUICAO', 'CONSTITUCIONAL', 'INCONSTITUCIONAL',
        'DECRETO', 'DECRETOS', 'PORTARIA', 'PORTARIAS',
        'RESOLUÇÃO', 'RESOLUCAO', 'RESOLUÇÕES', 'RESOLUCOES',
        'INSTRUÇÃO', 'INSTRUCAO', 'INSTRUÇÕES', 'INSTRUCOES', 'NORMATIVA', 'NORMATIVAS',
        'SÚMULA', 'SUMULA', 'SÚMULAS', 'SUMULAS',
        'JURISPRUDÊNCIA', 'JURISPRUDENCIA', 'JURISPRUDÊNCIAS', 'JURISPRUDENCIAS',
        'PRECEDENTE', 'PRECEDENTES', 'ORIENTAÇÃO', 'ORIENTACAO', 'ORIENTAÇÕES', 'ORIENTACOES',
        'ENTENDIMENTO', 'ENTENDIMENTOS', 'POSICIONAMENTO', 'POSICIONAMENTOS',
        
        # Parágrafos e referências
        'PARÁGRAFO', 'PARAGRAFO', 'PARÁGRAFOS', 'PARAGRAFOS',
        'ARTIGO', 'ARTIGOS', 'ART',
        'INCISO', 'INCISOS',
        'ALÍNEA', 'ALINEA', 'ALÍNEAS', 'ALINEAS',
        
        # Outras palavras comuns em processos
        'TRIBUNAL', 'TRIBUNAIS',
        'VARA', 'VARAS',
        'FORO', 'FOROS',
        'COMARCA', 'COMARCAS',
        'MINISTÉRIO', 'MINISTERIO', 'PÚBLICO', 'PUBLICO',
        'DEFENSORIA', 'PÚBLICA', 'PUBLICA',
        'ADVOCACIA', 'GERAL',
        'UNIÃO', 'UNIAO',
        'ESTADO', 'ESTADOS',
        'MUNICÍPIO', 'MUNICIPIO', 'MUNICÍPIOS', 'MUNICIPIOS',
        'FEDERAL', 'ESTADUAL', 'MUNICIPAL',
        
        # Órgãos de segurança pública
        'POLÍCIA', 'POLICIA', 'MILITAR', 'CIVIL', 'FEDERAL',
        'EXÉRCITO', 'EXERCITO', 'MARINHA', 'AERONÁUTICA', 'AERONAUTICA',
        'BOMBEIROS', 'BOMBEIRO', 'GUARDA', 'GUARDAS',
        'DELEGACIA', 'DELEGACIAS', 'DELEGADO', 'DELEGADA', 'DELEGADOS', 'DELEGADAS',
        'INSPETOR', 'INSPETORA', 'INSPETORES', 'INSPETORAS',
        'INVESTIGADOR', 'INVESTIGADORA', 'INVESTIGADORES', 'INVESTIGADORAS',
        'ESCRIVÃO', 'ESCRIVAO', 'ESCRIVÃ', 'ESCRIVA', 'ESCRIVÃES', 'ESCRIVAES',
        'PERITO', 'PERITA', 'PERITOS', 'PERITAS', 'CRIMINAL', 'CRIMINAIS',
        'AGENTE', 'AGENTES', 'OFICIAL', 'OFICIAIS',
        'COMANDANTE', 'COMANDANTES', 'CORONEL', 'CORONÉIS', 'CORONEIS',
        'MAJOR', 'MAJORES', 'CAPITÃO', 'CAPITAO', 'CAPITÃES', 'CAPITAES',
        'TENENTE', 'TENENTES', 'SARGENTO', 'SARGENTOS', 'CABO', 'CABOS',
        'SOLDADO', 'SOLDADOS', 'SOLDADA', 'SOLDADAS',
        
        # Tribunais superiores e justiças
        'STF', 'SUPREMO', 'TRIBUNAL', 'FEDERAL',
        'STJ', 'SUPERIOR', 'JUSTIÇA', 'JUSTICA',
        'TST', 'TRABALHO', 'TSE', 'ELEITORAL', 'STM', 'MILITAR',
        'TJ', 'TJSP', 'TJRJ', 'TJMG', 'TJRS', 'TJPR', 'TJSC', 'TJGO', 'TJBA',
        'TJPE', 'TJCE', 'TJPA', 'TJMA', 'TJPB', 'TJES', 'TJPI', 'TJAL', 'TJSE',
        'TJRN', 'TJMT', 'TJMS', 'TJRO', 'TJAC', 'TJAP', 'TJRR', 'TJAM', 'TJTO',
        'TJDF', 'TJDFT',
        'TRF', 'REGIONAL', 'TRT', 'TRE',
        'PRIMEIRA', 'SEGUNDA', 'TERCEIRA', 'QUARTA', 'QUINTA', 'SEXTA',
        'SÉTIMA', 'SETIMA', 'OITAVA', 'NONA', 'DÉCIMA', 'DECIMA',
        'INSTÂNCIA', 'INSTANCIA', 'INSTÂNCIAS', 'INSTANCIAS',
        'GRAU', 'GRAUS', 'TURMA', 'TURMAS', 'CÂMARA', 'CAMARA', 'CÂMARAS', 'CAMARAS',
        
        # Estados brasileiros
        'ACRE', 'ALAGOAS', 'AMAPÁ', 'AMAPA', 'AMAZONAS', 'BAHIA',
        'CEARÁ', 'CEARA', 'DISTRITO', 'ESPÍRITO', 'ESPIRITO', 'SANTO',
        'GOIÁS', 'GOIAS', 'MARANHÃO', 'MARANHAO', 'MATO', 'GROSSO', 'SUL',
        'MINAS', 'GERAIS', 'PARÁ', 'PARA', 'PARAÍBA', 'PARAIBA',
        'PARANÁ', 'PARANA', 'PERNAMBUCO', 'PIAUÍ', 'PIAUI',
        'RIO', 'JANEIRO', 'GRANDE', 'NORTE', 'RONDÔNIA', 'RONDONIA',
        'RORAIMA', 'SANTA', 'CATARINA', 'SÃO', 'SAO', 'PAULO',
        'SERGIPE', 'TOCANTINS',
        
        # Siglas dos estados
        'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
        'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
        'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO',
        
        # Principais capitais e cidades
        'BRASÍLIA', 'BRASILIA', 'BELO', 'HORIZONTE', 'SALVADOR', 'FORTALEZA',
        'RECIFE', 'CURITIBA', 'MANAUS', 'BELÉM', 'BELEM', 'GOIÂNIA', 'GOIANIA',
        'VITÓRIA', 'VITORIA', 'NATAL', 'JOÃO', 'JOAO', 'PESSOA', 'ARACAJU',
        'MACEIÓ', 'MACEIO', 'TERESINA', 'CUIABÁ', 'CUIABA', 'CAMPO',
        'FLORIANÓPOLIS', 'FLORIANOPOLIS', 'MACAPÁ', 'MACAPA',
        'RIO', 'BRANCO', 'BOA', 'VISTA', 'PALMAS', 'PORTO', 'ALEGRE',
        
        # Termos geográficos
        'CIDADE', 'CIDADES', 'CAPITAL', 'CAPITAIS', 'INTERIOR',
        'REGIÃO', 'REGIAO', 'REGIÕES', 'REGIOES', 'ZONA', 'ZONAS',
        'BAIRRO', 'BAIRROS', 'DISTRITO', 'DISTRITOS',
        'CENTRO', 'CENTROS', 'ÁREA', 'AREA', 'ÁREAS', 'AREAS',
        'LOCAL', 'LOCAIS', 'LUGAR', 'LUGARES', 'ENDEREÇO', 'ENDERECO',
        'ENDEREÇOS', 'ENDERECOS', 'LOGRADOURO', 'LOGRADOUROS',
        'RUA', 'RUAS', 'AVENIDA', 'AVENIDAS', 'ALAMEDA', 'ALAMEDAS',
        'PRAÇA', 'PRACA', 'PRAÇAS', 'PRACAS', 'LARGO', 'LARGOS',
        'TRAVESSA', 'TRAVESSAS', 'VIELA', 'VIELAS', 'ESTRADA', 'ESTRADAS',
        
        # Outros órgãos públicos importantes
        'RECEITA', 'FEDERAL', 'FAZENDA', 'PREVIDÊNCIA', 'PREVIDENCIA', 'SOCIAL',
        'INSS', 'FGTS', 'CAIXA', 'ECONÔMICA', 'ECONOMICA',
        'BANCO', 'BRASIL', 'CENTRAL', 'BACEN',
        'ANVISA', 'ANATEL', 'ANEEL', 'ANP', 'ANAC', 'ANTAQ', 'ANTT',
        'IBAMA', 'INCRA', 'FUNAI', 'IPHAN',
        'TCU', 'TCE', 'CONTROLE', 'CONTAS',
        'CGU', 'CONTROLADORIA', 'TRANSPARÊNCIA', 'TRANSPARENCIA',
        'AGU', 'PROCURADORIA', 'PROCURADORIAS',
        'PROCURADOR', 'PROCURADORA', 'PROCURADORES', 'PROCURADORAS',
        'PROMOTOR', 'PROMOTORA', 'PROMOTORES', 'PROMOTORAS',
        'DEFENSOR', 'DEFENSORA', 'DEFENSORES', 'DEFENSORAS',
        
        # Autarquias e fundações
        'AUTARQUIA', 'AUTARQUIAS', 'FUNDAÇÃO', 'FUNDACAO', 'FUNDAÇÕES', 'FUNDACOES',
        'EMPRESA', 'EMPRESAS', 'SOCIEDADE', 'SOCIEDADES',
        'COMPANHIA', 'COMPANHIAS', 'CIA',
        'SECRETARIA', 'SECRETARIAS', 'SECRETÁRIO', 'SECRETARIO',
        'SECRETÁRIA', 'SECRETARIA', 'SECRETÁRIOS', 'SECRETARIOS', 'SECRETÁRIAS', 'SECRETARIAS',
        'DEPARTAMENTO', 'DEPARTAMENTOS', 'DIRETORIA', 'DIRETORIAS',
        'COORDENAÇÃO', 'COORDENACAO', 'COORDENAÇÕES', 'COORDENACOES',
        'COORDENADOR', 'COORDENADORA', 'COORDENADORES', 'COORDENADORAS',
        'GERÊNCIA', 'GERENCIA', 'GERÊNCIAS', 'GERENCIAS',
        'GERENTE', 'GERENTES', 'DIRETOR', 'DIRETORA', 'DIRETORES', 'DIRETORAS',
        
        # Conectivos e preposições adicionais
        'COM', 'SEM', 'POR', 'PARA', 'ANTE', 'APÓS', 'APOS',
        'ATÉ', 'ATE', 'CONTRA', 'DESDE', 'ENTRE', 'PERANTE',
        'SEGUNDO', 'SOBRE', 'CONFORME', 'MEDIANTE',
        
        # Outros termos que podem aparecer
        'RELATÓRIO', 'RELATORIO', 'RELATÓRIOS', 'RELATORIOS',
        'VOTO', 'VOTOS',
        'PARECER', 'PARECERES',
        'CERTIDÃO', 'CERTIDAO', 'CERTIDÕES', 'CERTIDOES',
        'OFÍCIO', 'OFICIO', 'OFÍCIOS', 'OFICIOS',
        
        # Termos jurídicos específicos
        'ABSOLUTAMENTE', 'INCAPAZ', 'INCAPAZES',
        'QUADRO', 'QUADROS', 'ACESSO', 'ACESSOS',
        'CURSO', 'CURSOS', 'SUPERIOR', 'SUPERIORES',
        'ACÓRDÃO', 'ACORDAO', 'ACÓRDÃOS', 'ACORDAOS',
        'LECIONA', 'LECIONAM', 'LECIONAR',
        'SEÇÃO', 'SECAO', 'SEÇÕES', 'SECOES',
        'SERÁ', 'SERA', 'SERÃO', 'SERAO',
        'FONE', 'TELEFONE', 'TELEFONES',
        'ALEGA', 'ALEGAM', 'ALEGAR', 'ALEGAÇÃO', 'ALEGACAO', 'ALEGAÇÕES', 'ALEGACOES',
        'ADVOGADO', 'ADVOGADOS', 'ADVOGADA', 'ADVOGADAS',
        
        # Respostas e termos comuns
        'SIM', 'NÃO', 'NAO',
        'NUM', 'NUMS', 'NÚMERO', 'NUMERO', 'NÚMEROS', 'NUMEROS',
        
        # Números romanos
        'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
        'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
        'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXX', 'XL', 'L', 'LX', 'LXX',
        'LXXX', 'XC', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM', 'M',
        
        # Verbos processuais comuns
        'REQUER', 'REQUEREM', 'REQUERER', 'REQUERIMENTO', 'REQUERIMENTOS',
        'SOLICITA', 'SOLICITAM', 'SOLICITAR', 'SOLICITAÇÃO', 'SOLICITACAO',
        'PEDE', 'PEDEM', 'PEDIR', 'PEDIDO', 'PEDIDOS',
        'PROTESTA', 'PROTESTAM', 'PROTESTAR',
        'CONTESTA', 'CONTESTAM', 'CONTESTAR', 'CONTESTAÇÃO', 'CONTESTACAO',
        'IMPUGNA', 'IMPUGNAM', 'IMPUGNAR', 'IMPUGNAÇÃO', 'IMPUGNACAO',
        'INFORMA', 'INFORMAM', 'INFORMAR', 'INFORMAÇÃO', 'INFORMACAO', 'INFORMAÇÕES', 'INFORMACOES',
        'ESCLARECE', 'ESCLARECEM', 'ESCLARECER', 'ESCLARECIMENTO', 'ESCLARECIMENTOS',
        'MANIFESTA', 'MANIFESTAM', 'MANIFESTAR', 'MANIFESTAÇÃO', 'MANIFESTACAO',
        'DECLARA', 'DECLARAM', 'DECLARAR', 'DECLARAÇÃO', 'DECLARACAO',
        'COMPROVA', 'COMPROVAM', 'COMPROVAR', 'COMPROVAÇÃO', 'COMPROVACAO',
        'DEMONSTRA', 'DEMONSTRAM', 'DEMONSTRAR', 'DEMONSTRAÇÃO', 'DEMONSTRACAO',
        
        # Termos de tempo e localização
        'HOJE', 'ONTEM', 'AMANHÃ', 'AMANHA',
        'AGORA', 'ANTES', 'DEPOIS', 'DURANTE',
        'AQUI', 'ALI', 'LÁ', 'LA', 'ONDE', 'QUANDO',
        'SEMPRE', 'NUNCA', 'AINDA', 'JÁ', 'JA',
        
        # Advérbios e conectivos processuais
        'PORTANTO', 'CONTUDO', 'TODAVIA', 'ENTRETANTO',
        'ASSIM', 'DESSA', 'DESTA', 'DESSE', 'DESTE',
        'FORMA', 'MODO', 'MANEIRA',
        'VEZ', 'VEZES', 'PRIMEIRA', 'SEGUNDA', 'TERCEIRA',
        'ÚLTIMO', 'ULTIMO', 'ÚLTIMA', 'ULTIMA',
        'PRÓXIMO', 'PROXIMO', 'PRÓXIMA', 'PROXIMA',
        'ANTERIOR', 'POSTERIORES', 'POSTERIOR',
        
        # Expressões jurídicas específicas
        'ISTO', 'ISSO', 'AQUILO', 'ESTE', 'ESTA', 'ESTES', 'ESTAS',
        'ESSE', 'ESSA', 'ESSES', 'ESSAS', 'AQUELE', 'AQUELA', 'AQUELES', 'AQUELAS',
        'QUAL', 'QUAIS', 'QUANTO', 'QUANTA', 'QUANTOS', 'QUANTAS',
        'CUJO', 'CUJA', 'CUJOS', 'CUJAS',
        'MEDIANTE', 'ATRAVÉS', 'ATRAVES', 'PERANTE', 'DIANTE',
        'FACE', 'VISTA', 'LUZ', 'RAZÃO', 'RAZAO', 'ORDEM',
        'TERMOS', 'TERMO', 'TEOR', 'FORÇA', 'FORCA', 'VIGÊNCIA', 'VIGENCIA',
        'AMPARO', 'BASE', 'FUNDAMENTO', 'FUNDAMENTOS', 'PRINCÍPIO', 'PRINCIPIO',
        'PRINCÍPIOS', 'PRINCIPIOS', 'REGRA', 'REGRAS', 'NORMA', 'NORMAS',
        
        # Valores e quantias
        'VALOR', 'VALORES', 'QUANTIA', 'QUANTIAS', 'IMPORTÂNCIA', 'IMPORTANCIA',
        'MONTANTE', 'MONTANTES', 'SOMA', 'SOMAS', 'TOTAL', 'TOTAIS',
        'DANO', 'DANOS', 'PREJUÍZO', 'PREJUIZO', 'PREJUÍZOS', 'PREJUIZOS',
        'LUCRO', 'LUCROS', 'GANHO', 'GANHOS', 'PERDA', 'PERDAS',
        'DÍVIDA', 'DIVIDA', 'DÍVIDAS', 'DIVIDAS', 'DÉBITO', 'DEBITO', 'DÉBITOS', 'DEBITOS',
        'CRÉDITO', 'CREDITO', 'CRÉDITOS', 'CREDITOS',
        'MULTA', 'MULTAS', 'CORREÇÃO', 'CORRECAO', 'CORREÇÕES', 'CORRECOES',
        'JUROS', 'HONORÁRIOS', 'HONORARIOS', 'CUSTAS', 'TAXA', 'TAXAS',
        
        # Tempo processual
        'PRAZO', 'PRAZOS', 'TERMO', 'TERMOS', 'INÍCIO', 'INICIO', 'FIM',
        'DIES', 'DATA', 'DATAS', 'PERÍODO', 'PERIODO', 'PERÍODOS', 'PERIODOS',
        'ANO', 'ANOS', 'MÊS', 'MES', 'MESES', 'DIA', 'DIAS',
        'HORA', 'HORAS', 'MINUTO', 'MINUTOS', 'MOMENTO', 'MOMENTOS',
        'OPORTUNIDADE', 'OPORTUNIDADES', 'OCASIÃO', 'OCASIAO', 'OCASIÕES', 'OCASIOES',
        
        # Termos quantitativos
        'TODOS', 'TODAS', 'TODO', 'TODA',
        'ALGUNS', 'ALGUMAS', 'ALGUM', 'ALGUMA',
        'NENHUM', 'NENHUMA', 'NENHUNS', 'NENHUMAS',
        'MUITO', 'MUITA', 'MUITOS', 'MUITAS',
        'POUCO', 'POUCA', 'POUCOS', 'POUCAS',
        'MAIS', 'MENOS', 'MAIOR', 'MENOR',
        'MELHOR', 'PIOR', 'IGUAL', 'DIFERENTE'
    }
    
    doc = nlp(texto)
    nomes = []
    
    for ent in doc.ents:
        if ent.label_ == "PER":  # Apenas entidades identificadas como pessoas
            nome = ent.text.strip()
            
            # Verifica se não é uma palavra a ser descartada
            if nome.upper() not in palavras_descartadas:
                # Filtros adicionais para garantir qualidade
                if (len(nome) >= 3 and  # Nome deve ter pelo menos 3 caracteres
                    not nome.isdigit() and  # Não pode ser apenas números
                    not all(c in '.,;:-_()[]{}' for c in nome)):  # Não pode ser apenas pontuação
                    nomes.append(nome)
    
    return list(set(nomes))  # Remove duplicatas

def anonimizar_texto(texto: str) -> str:
    substituicoes = {
        r'\b\d{3}\.??\d{3}\.??\d{3}-??\d{2}\b': '[CPF]',
        r'\b\d{2}\.??\d{3}\.??\d{3}/??\d{4}-??\d{2}\b': '[CNPJ]',
        r'\b\d{2}/\d{2}/\d{4}\b': '[DATA]',
        r'\b\d{4}-\d{2}-\d{2}\b': '[DATA_ISO]',
        r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b': '[EMAIL]',
        r'\b\d{5}-\d{3}\b': '[CEP]',
        r'\b(?:\(?\d{2}\)?\s?)?(?:9?\d{4})-?\d{4}\b': '[TELEFONE]',
    }
    for padrao, substituto in substituicoes.items():
        texto = re.sub(padrao, substituto, texto)
    return texto

def anonimizar_com_identificadores(texto: str, mapa_suspeitos: dict) -> tuple[str, dict]:
    nomes = extrair_nomes_spacy(texto)
    reverso = {}
    substituidos = set()
    contador = 1

    for nome in nomes:
        nome_norm = normalizar(nome)
        if nome_norm in mapa_suspeitos:
            ident, nome_real = mapa_suspeitos[nome_norm]
            if ident not in reverso:
                reverso[ident] = nome_real
            padrao = re.compile(rf'\b{re.escape(nome)}\b', flags=re.IGNORECASE)
            texto, n = padrao.subn(f"{nome_real} ({ident})", texto)
            if n > 0:
                substituidos.add(nome)
                print(f"SUSPEITO: {nome} → {ident} ({n}x)")

    for nome in nomes:
        if nome in substituidos:
            continue
        ident = f"#NOME_{contador:03}"
        padrao = re.compile(rf'\b{re.escape(nome)}\b', flags=re.IGNORECASE)
        texto, n = padrao.subn(ident, texto)
        if n > 0:
            reverso[ident] = nome
            print(f"Nome comum: {nome} → {ident} ({n}x)")
            contador += 1

    return anonimizar_texto(texto), reverso

def salvar_anonimizacao_md(conteudo_md: str, nome_base: str, mapa_suspeitos: dict):
    print(f"Usando PASTA_DESTINO: {PASTA_DESTINO}")
    pasta_anon = os.path.join(PASTA_DESTINO, "anonimizados")
    pasta_mapas = os.path.join(PASTA_DESTINO, "mapas")
    os.makedirs(pasta_anon, exist_ok=True)
    os.makedirs(pasta_mapas, exist_ok=True)

    texto_anon, mapa_reverso = anonimizar_com_identificadores(conteudo_md, mapa_suspeitos)

    caminho_md = os.path.join(pasta_anon, f"{nome_base}_anon.md")
    with open(caminho_md, "w", encoding="utf-8") as f:
        f.write(texto_anon)

    if mapa_reverso:
        caminho_mapa = os.path.join(pasta_mapas, f"{nome_base}_mapa.md")
        with open(caminho_mapa, "w", encoding="utf-8") as f:
            f.write("| Identificador | Nome Original |\n")
            f.write("|---------------|----------------|\n")
            for ident, nome in mapa_reverso.items():
                f.write(f"| {ident} | {nome} |\n")

    print(f"Arquivos salvos:\n- {caminho_md}\n- {caminho_mapa if mapa_reverso else '(sem mapa)'}")

# === Exemplo de uso ===
if __name__ == "__main__":
    markdown_exemplo = """
    # Relatório

    Participaram da reunião: JOÃO DA SILVA, Maria Lima, Fernanda dos Santos.
    Contato: maria@email.com
    """

    mapa = carregar_suspeitos_mapeados("suspeitos.txt")
    salvar_anonimizacao_md(markdown_exemplo, "relatorio_teste", mapa)
