from flask import current_app
import os
import re
import logging
from datetime import datetime
from utils.auxiliar import limpar, extrair_tabela_md
from utils.suspeicao import encontrar_suspeitos

logger = logging.getLogger(__name__)

def atualizar_processo(numero, data):
    path_triagem = current_app.config['PATH_TRIAGEM']
    pasta_dest = current_app.config['PASTA_DESTINO']
    pasta_dat = current_app.config['PASTA_DAT']
    processos = extrair_tabela_md(path_triagem)
    processo_existente = next((p for p in processos if p['numeroProcesso'] == numero), None)
    processos = [p for p in processos if p['numeroProcesso'] != numero]
    markdown = data.get('markdown', '')
    suspeitos_existentes = processo_existente.get('suspeitos', '') if processo_existente else ''
    suspeitos_calculados = ''
    if markdown.strip():
        try:
            suspeitos_lista = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
            suspeitos_calculados = ', '.join(suspeitos_lista)
        except:
            suspeitos_calculados = suspeitos_existentes
    else:
        suspeitos_calculados = suspeitos_existentes
    nome_base = numero.replace('/', '-')
    if markdown.strip():
        os.makedirs(pasta_dest, exist_ok=True)
        with open(os.path.join(pasta_dest, f"{nome_base}.md"), 'w', encoding='utf-8') as f:
            f.write(markdown)
    dat_base64 = data.get('dat')
    if dat_base64:
        os.makedirs(pasta_dat, exist_ok=True)
        with open(os.path.join(pasta_dat, f"{nome_base}.dat"), 'w', encoding='utf-8') as f:
            f.write(dat_base64)
    ultima_att = datetime.now().strftime('%Y-%m-%d')
    atualizado = {
        "numeroProcesso": limpar(data['numeroProcesso']),
        "tema": limpar(data['tema']),
        "dataDistribuicao": limpar(data['dataDistribuicao']),
        "responsavel": limpar(data['responsavel']),
        "status": limpar(data['status']),
        "ultimaAtualizacao": ultima_att,
        "suspeitos": suspeitos_calculados,
        "prioridade": limpar(data.get('prioridade', 'MÉDIA')),
        "comentarios": limpar(data.get('comentarios', ''))
    }
    processos.append(atualizado)
    with open(path_triagem, 'w', encoding='utf-8') as f:
        f.write("# Tabela de Processos\n\n")
        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Prioridade | Comentários |\n")
        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|------------|-------------|\n")
        for p in processos:
            f.write(f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} | {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('prioridade', 'MÉDIA')} | {p.get('comentarios', '')} |\n")

def deletar_processo_por_numero(numero):
    path_triagem = current_app.config['PATH_TRIAGEM']
    pasta_dest = current_app.config['PASTA_DESTINO']
    pasta_dat = current_app.config['PASTA_DAT']
    processos = extrair_tabela_md(path_triagem)
    processos = [p for p in processos if p['numeroProcesso'] != numero]
    with open(path_triagem, 'w', encoding='utf-8') as f:
        f.write("# Tabela de Processos\n\n")
        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Prioridade | Comentários |\n")
        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|------------|-------------|\n")
        for p in processos:
            f.write(f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} | {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('prioridade', 'MÉDIA')} | {p.get('comentarios', '')} |\n")
    caminho_md = os.path.join(pasta_dest, f"{numero.replace('/', '-')}.md")
    if os.path.exists(caminho_md):
        os.remove(caminho_md)
    caminho_dat = os.path.join(pasta_dat, f"{numero.replace('/', '-')}.dat")
    if os.path.exists(caminho_dat):
        os.remove(caminho_dat)
    caminho_md_anon = os.path.join(pasta_dest, "anonimizados", f"{numero.replace('/', '-')}_anon.md")
    if os.path.exists(caminho_md_anon):
        os.remove(caminho_md_anon)
    caminho_mapa = os.path.join(pasta_dest, "mapas", f"{numero.replace('/', '-')}_mapa.md")
    if os.path.exists(caminho_mapa):
        os.remove(caminho_mapa)
    nome_pasta = re.sub(r'[^\w\-.]', '_', numero)
    pasta_processo = os.path.join(pasta_dest, nome_pasta)
    if os.path.isdir(pasta_processo):
        import shutil
        shutil.rmtree(pasta_processo, ignore_errors=True)

def obter_dat_por_numero(numero):
    caminho = os.path.join(current_app.config['PASTA_DAT'], f"{numero.replace('/', '-')}.dat")
    if not os.path.exists(caminho):
        raise FileNotFoundError("Arquivo .dat não encontrado")
    with open(caminho, 'r', encoding='utf-8') as f:
        return f.read()

def get_processos():
    path_triagem = current_app.config['PATH_TRIAGEM']
    if not os.path.exists(path_triagem):
        logger.warning(f"Arquivo de triagem não existe: {path_triagem}")
        return []
    try:
        processos = extrair_tabela_md(path_triagem)
        logger.info(f"Extraídos {len(processos)} processos da tabela")
        return processos
    except Exception as e:
        logger.error(f"Erro ao extrair processos: {e}")
        return []