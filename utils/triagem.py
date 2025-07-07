from flask import current_app
import os
import re
import time
import logging
import threading
import socket
from datetime import datetime
from utils.auxiliar import limpar, extrair_tabela_md
from utils.suspeicao import encontrar_suspeitos
from utils.progress_step import send_progress_ws
from threading import RLock

logger = logging.getLogger(__name__)
file_write_lock = RLock()

def processar_com_progresso(data, operation_id, operation_sockets):
    logger.info(f"Iniciando processamento para operation_id: {operation_id}")
    send_progress_ws(operation_id, 1, 'Iniciando processamento...', 5)
    try:
        send_progress_ws(operation_id, 1, 'Validando dados do processo...', 10)
        numero = limpar(data.get('numeroProcesso'))
        markdown = limpar(data.get('markdown'))
        if not numero or not markdown:
            logger.error(f"Campos obrigatórios ausentes para {operation_id}")
            send_progress_ws(operation_id, 0, 'Erro: Campos obrigatórios ausentes', 0)
            return
        tema = limpar(data.get('tema'))
        data_dist = limpar(data.get('dataDistribuicao'))
        responsavel = limpar(data.get('responsavel'))
        status = limpar(data.get('status'))
        comentarios = limpar(data.get('comentarios'))
        prioridade = limpar(data.get('prioridade', 'MÉDIA'))
        dat_base64 = data.get('dat')
        ultima_att = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Processando: {numero}")
        logger.info(f"Dados extraídos - Tema: {tema}, Responsável: {responsavel}, Prioridade: {prioridade}")
        send_progress_ws(operation_id, 2, 'Analisando suspeição...', 25)
        suspeitos = []
        if markdown:
            try:
                suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                logger.info(f"Suspeitos encontrados: {suspeitos}")
            except Exception as e:
                logger.error(f"Erro na análise de suspeitos: {e}")
                suspeitos = []
        send_progress_ws(operation_id, 3, 'Preparando arquivos...', 40)
        nome_base = numero.replace('/', '-')
        pasta_dest = current_app.config['PASTA_DESTINO']
        pasta_dat = current_app.config['PASTA_DAT']
        path_triagem = current_app.config['PATH_TRIAGEM']
        os.makedirs(pasta_dest, exist_ok=True)
        os.makedirs(pasta_dat, exist_ok=True)
        send_progress_ws(operation_id, 4, 'Processando anonimização...', 60)
        logger.info(f"Anonimização processada para {numero}")
        send_progress_ws(operation_id, 5, 'Salvando arquivos...', 70)
        try:
            caminho_md = os.path.join(pasta_dest, f"{nome_base}.md")
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            logger.info(f"Markdown salvo: {caminho_md}")
        except Exception as e:
            logger.error(f"Erro ao salvar markdown: {e}")
        if dat_base64:
            try:
                caminho_dat = os.path.join(pasta_dat, f"{nome_base}.dat")
                with open(caminho_dat, 'w', encoding='utf-8') as f:
                    f.write(dat_base64)
                logger.info(f"DAT salvo: {caminho_dat}")
            except Exception as e:
                logger.error(f"Erro ao salvar DAT: {e}")
        send_progress_ws(operation_id, 6, 'Atualizando tabela de triagem...', 80)
        atualizar_tabela_triagem(numero, tema, data_dist, responsavel, status, 
                                ultima_att, suspeitos, prioridade, comentarios, path_triagem)
        send_progress_ws(operation_id, 7, 'Tabela atualizada com sucesso!', 85)
        logger.info(f"Processo {numero} salvo na tabela: {path_triagem}")
        send_progress_ws(operation_id, 8, 'Iniciando notificação por email (background)...', 90)
        dados_notificacao = {
            'numero': numero,
            'tema': tema,
            'data_dist': data_dist,
            'responsavel': responsavel,
            'status': status,
            'prioridade': prioridade,
            'comentarios': comentarios,
            'suspeitos': suspeitos
        }
        try:
            email_thread = threading.Thread(
                target=enviar_email_seguro,
                args=(dados_notificacao, operation_id),
                daemon=True
            )
            email_thread.start()
            logger.info(f"Thread de email iniciada para {responsavel}")
        except Exception as e:
            logger.warning(f"Erro ao iniciar thread de email: {e}")
        send_progress_ws(operation_id, 9, 'Processo concluído com sucesso!', 100)
        time.sleep(0.5)
        logger.info(f"Processo {numero} processado com sucesso")
        operation_sockets.pop(operation_id, None)
    except Exception as e:
        send_progress_ws(operation_id, 0, f'Erro: {str(e)}', 0)
        import traceback
        traceback.print_exc()
        operation_sockets.pop(operation_id, None)

def enviar_email_seguro(dados_notificacao, operation_id):
    try:
        from utils.email_notification import enviar_notificacao_processo
        numero = dados_notificacao.get('numero', 'N/A')
        responsavel = dados_notificacao.get('responsavel', 'N/A')
        logger.info(f"Iniciando envio para {responsavel} (processo {numero})")
        socket.setdefaulttimeout(30)
        try:
            success = enviar_notificacao_processo(dados_notificacao)
            if success:
                logger.info(f"Email enviado com sucesso para {responsavel}")
            else:
                logger.warning(f"Falha no envio de email para {responsavel}")
        except socket.timeout:
            logger.warning(f"Email timeout para {responsavel}")
        except Exception as e:
            logger.warning(f"Erro no envio de email para {responsavel}: {e}")
    except ImportError:
        logger.warning(f"Módulo email_notification não disponível")
    except Exception as e:
        logger.warning(f"Erro geral no envio de email: {e}")
    finally:
        logger.info(f"Finalizada para processo {dados_notificacao.get('numero')}")

def atualizar_tabela_triagem(numero, tema, data_dist, responsavel, status, ultima_att, suspeitos, prioridade, comentarios, path_triagem):
    try:
        logger.info(f"Iniciando atualização da tabela para {numero}")
        logger.info(f"Path da tabela: {path_triagem}")
        suspeitos_str = ', '.join(suspeitos) if suspeitos else ''
        nova_linha = (
            f"| {numero} "
            f"| {tema} "
            f"| {data_dist} "
            f"| {responsavel} "
            f"| {status} "
            f"| {ultima_att} "
            f"| {suspeitos_str} "
            f"| {prioridade} "
            f"| {comentarios} |\n"
        )
        logger.info(f"Nova linha formatada: {nova_linha.strip()}")
        dir_tabela = os.path.dirname(path_triagem)
        if dir_tabela:
            os.makedirs(dir_tabela, exist_ok=True)
            logger.info(f"Diretório garantido: {dir_tabela}")
        with file_write_lock:
            if not os.path.exists(path_triagem):
                logger.info(f"Arquivo não existe, criando: {path_triagem}")
                criar_cabecalho_tabela(path_triagem)
            else:
                verificar_cabecalho_tabela(path_triagem)
            try:
                with open(path_triagem, 'r', encoding='utf-8') as f:
                    conteudo_atual = f.read()
                logger.info(f"Arquivo lido, tamanho: {len(conteudo_atual)} chars")
            except Exception as e:
                logger.error(f"Erro ao ler arquivo: {e}")
                conteudo_atual = ""
            if not conteudo_atual.strip() or "| Nº Processo |" not in conteudo_atual:
                logger.warning(f"Arquivo vazio ou corrompido, recriando")
                criar_cabecalho_tabela(path_triagem)
                with open(path_triagem, 'r', encoding='utf-8') as f:
                    conteudo_atual = f.read()
            if numero in conteudo_atual:
                logger.warning(f"Processo {numero} já existe na tabela, sobrescrevendo")
                linhas = conteudo_atual.split('\n')
                linhas_filtradas = [linha for linha in linhas if numero not in linha or not linha.startswith('|')]
                conteudo_atual = '\n'.join(linhas_filtradas)
            conteudo_novo = conteudo_atual.rstrip() + '\n' + nova_linha
            with open(path_triagem, 'w', encoding='utf-8') as f:
                f.write(conteudo_novo)
            logger.info(f"Arquivo escrito com sucesso")
            verificar_escrita_sucesso(path_triagem, numero)
    except Exception as e:
        logger.error(f"Erro ao atualizar tabela para {numero}: {e}")
        logger.error(f"Path: {path_triagem}")
        try:
            logger.info(f"Tentando método fallback...")
            with open(path_triagem, 'a', encoding='utf-8') as f:
                f.write(nova_linha)
            logger.info(f"Método fallback funcionou!")
            verificar_escrita_sucesso(path_triagem, numero)
        except Exception as e2:
            logger.error(f"Método fallback também falhou: {e2}")
            raise e2

def criar_cabecalho_tabela(path_triagem):
    logger.info(f"Criando cabeçalho da tabela: {path_triagem}")
    cabecalho = """# Tabela de Processos

| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Prioridade | Comentários |
|-------------|------|-----------------------|-------------|--------|----------------------|-----------|------------|-------------|
"""
    with open(path_triagem, 'w', encoding='utf-8') as f:
        f.write(cabecalho)
    logger.info(f"Cabeçalho criado com sucesso")

def verificar_cabecalho_tabela(path_triagem):
    try:
        with open(path_triagem, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        if "| Prioridade |" not in conteudo:
            logger.warning(f"Cabeçalho sem coluna Prioridade, corrigindo...")
            linhas = conteudo.split('\n')
            dados_existentes = []
            for linha in linhas:
                if linha.strip().startswith('|') and not linha.strip().startswith('|--'):
                    if 'Nº Processo' not in linha:
                        dados_existentes.append(linha.strip())
            criar_cabecalho_tabela(path_triagem)
            if dados_existentes:
                with open(path_triagem, 'a', encoding='utf-8') as f:
                    for linha in dados_existentes:
                        f.write(linha + '\n')
                logger.info(f"{len(dados_existentes)} linhas de dados preservadas")
            logger.info(f"Cabeçalho corrigido com sucesso")
    except Exception as e:
        logger.error(f"Erro ao verificar cabeçalho: {e}")

def verificar_escrita_sucesso(path_triagem, numero):
    try:
        with open(path_triagem, 'r', encoding='utf-8') as f:
            conteudo = f.read()
        if numero in conteudo:
            logger.info(f"Processo {numero} confirmado na tabela")
            linhas = conteudo.split('\n')
            linhas_dados = [l for l in linhas if l.strip().startswith('|') and 'Nº Processo' not in l and not l.startswith('|--')]
            logger.info(f"Estatísticas da tabela:")
            logger.info(f"- Tamanho do arquivo: {len(conteudo)} chars")
            logger.info(f"- Total de linhas: {len(linhas)}")
            logger.info(f"- Linhas de dados: {len(linhas_dados)}")
            logger.info(f"- Última linha: {linhas[-2] if len(linhas) > 1 else 'N/A'}")
        else:
            logger.error(f"Processo {numero} NÃO encontrado na tabela!")
            logger.error(f"Conteúdo atual da tabela:")
            for i, linha in enumerate(conteudo.split('\n')[-5:], 1):
                logger.error(f"Linha -{5-i}: {linha}")
    except Exception as e:
        logger.error(f"Erro ao verificar escrita: {e}")

def processar_sem_progresso(data, operation_id):
    numero = limpar(data.get('numeroProcesso'))
    markdown = limpar(data.get('markdown'))
    dat_base64 = data.get('dat')
    if not numero or not markdown:
        logger.error(f"Campos obrigatórios ausentes para processar_sem_progresso")
        return
    path_triagem = current_app.config['PATH_TRIAGEM']
    pasta_dest = current_app.config['PASTA_DESTINO']
    pasta_dat = current_app.config['PASTA_DAT']
    tema = limpar(data.get('tema'))
    data_dist = limpar(data.get('dataDistribuicao'))
    responsavel = limpar(data.get('responsavel'))
    status = limpar(data.get('status'))
    prioridade = limpar(data.get('prioridade', 'MÉDIA'))
    comentarios = limpar(data.get('comentarios'))
    ultima_att = datetime.now().strftime('%Y-%m-%d')
    suspeitos = []
    try:
        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
    except Exception as e:
        logger.error(f"Erro na análise de suspeitos: {e}")
    nome_base = numero.replace('/', '-')
    os.makedirs(pasta_dest, exist_ok=True)
    os.makedirs(pasta_dat, exist_ok=True)
    with open(os.path.join(pasta_dest, f"{nome_base}.md"), 'w', encoding='utf-8') as f:
        f.write(markdown)
    if dat_base64:
        with open(os.path.join(pasta_dat, f"{nome_base}.dat"), 'w', encoding='utf-8') as f:
            f.write(dat_base64)
    atualizar_tabela_triagem(numero, tema, data_dist, responsavel, status, 
                           ultima_att, suspeitos, prioridade, comentarios, path_triagem)

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