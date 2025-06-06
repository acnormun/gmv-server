import re

def normalizar(nome):
    return re.sub(r'\s+', ' ', nome.strip().lower())

def carregar_suspeitos(caminho_txt):
    with open(caminho_txt, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    return [linha.strip().split('|')[1] for linha in linhas if '|' in linha]

def extrair_nomes(texto):
    padrao_nome = r'\b(?:[A-ZÁ-Ú]{2,}|[A-ZÁ-Ú][a-zá-ú]{2,})(?:\s+(?:[dD][aeo]s?|[Dd]e|[Dd]o|[Dd]a)?\s*(?:[A-ZÁ-Ú]{2,}|[A-ZÁ-Ú][a-zá-ú]{2,})){0,4}\b'
    nomes = re.findall(padrao_nome, texto)
    palavras_descartadas = {'E', 'EM', 'NO', 'NA', 'DOS', 'DAS', 'DE', 'DO', 'DA', 'AOS', 'AO'}
    nomes_filtrados = [n.strip() for n in set(nomes) if n.upper() not in palavras_descartadas and len(n.strip()) >= 3]
    return sorted(nomes_filtrados, key=len, reverse=True)

def encontrar_suspeitos(markdown_text, caminho_suspeitos_txt):
    nomes_extraidos = extrair_nomes(markdown_text)
    print('nomes', nomes_extraidos)
    suspeitos_lista = carregar_suspeitos(caminho_suspeitos_txt)
    print('suspeitos lista', suspeitos_lista)
    nomes_extraidos_norm = [normalizar(n) for n in nomes_extraidos]
    print('nomes norm', nomes_extraidos_norm)
    suspeitos_norm = {normalizar(n): n for n in suspeitos_lista}
    print('suspeitos norm', suspeitos_norm)
    suspeicao = []
    for nome_norm in nomes_extraidos_norm:
        if nome_norm in suspeitos_norm:
            suspeicao.append(suspeitos_norm[nome_norm])
    print('suspeitos', suspeicao)
    return suspeicao
