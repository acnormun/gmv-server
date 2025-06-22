from flask import current_app
import os
import time
from datetime import datetime
from utils.auxiliar import limpar, extrair_tabela_md, get_anonimizador
from utils.suspeicao import encontrar_suspeitos
from utils.progress_step import send_progress_ws

def processar_com_progresso(data, operation_id, operation_sockets):
    """Versão melhorada com mais logs e fallback inteligente"""
    
    logger.info(f" Iniciando processamento para operation_id: {operation_id}")
    logger.info(f"operation_sockets atuais: {list(operation_sockets.keys())}")
    
    # Envia primeiro progresso SEMPRE (mesmo sem WebSocket)
    send_progress_ws(operation_id, 1, 'Iniciando processamento...', 5)
    
    # Aguarda WebSocket connection com timeout mais inteligente
    logger.info(f"⏳ Aguardando WebSocket connection para {operation_id}...")
    
    connection_found = False
    for attempt in range(10):  # 10 tentativas de 0.5s = 5s total
        if operation_id in operation_sockets:
            connection_found = True
            logger.info(f" WebSocket connection encontrada na tentativa {attempt + 1}")
            break
        time.sleep(0.5)
    
    if not connection_found:
        logger.warning(f" WebSocket não conectado para {operation_id}, mas continuando com progresso via log")
        # NÃO chama processar_sem_progresso! Continua normalmente
    
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
        dat_base64 = data.get('dat')
        ultima_att = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"📄 Processando: {numero}")

        send_progress_ws(operation_id, 2, 'Analisando suspeição...', 25)
        suspeitos = []
        if markdown:
            try:
                suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                logger.info(f" Suspeitos encontrados: {suspeitos}")
            except Exception as e:
                logger.error(f"Erro na análise de suspeitos: {e}")
                suspeitos = []

        send_progress_ws(operation_id, 3, 'Preparando arquivos...', 40)
        nome_base = numero.replace('/', '-')
        pasta_dest = current_app.config['PASTA_DESTINO']
        pasta_dat = current_app.config['PASTA_DAT']
        os.makedirs(pasta_dest, exist_ok=True)
        os.makedirs(pasta_dat, exist_ok=True)

        caminho_md = os.path.join(pasta_dest, f"{nome_base}.md")
        caminho_dat = os.path.join(pasta_dat, f"{nome_base}.dat")

        send_progress_ws(operation_id, 4, 'Salvando documentos...', 55)
        if markdown:
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            logger.info(f"💾 Markdown salvo: {caminho_md}")
                
        if dat_base64:
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            logger.info(f"💾 DAT salvo: {caminho_dat}")

        send_progress_ws(operation_id, 6, 'Anonimizando documento...', 75)
        try:
            if markdown:
                anonimizador = get_anonimizador()
                mapa = anonimizador.carregar_suspeitos_mapeados("utils/suspeitos.txt")
                texto_anon, mapa_reverso = anonimizador.anonimizar_com_identificadores(markdown, mapa)

                pasta_anon = os.path.join(pasta_dest, "anonimizados")
                pasta_mapas = os.path.join(pasta_dest, "mapas")
                os.makedirs(pasta_anon, exist_ok=True)
                os.makedirs(pasta_mapas, exist_ok=True)

                caminho_md_anon = os.path.join(pasta_anon, f"{nome_base}_anon.md")
                with open(caminho_md_anon, "w", encoding="utf-8") as f:
                    f.write(texto_anon)

                if mapa_reverso:
                    caminho_mapa = os.path.join(pasta_mapas, f"{nome_base}_mapa.md")
                    with open(caminho_mapa, "w", encoding="utf-8") as f:
                        f.write("| Identificador | Nome Original |\n")
                        f.write("|---------------|----------------|\n")
                        for ident, nome in mapa_reverso.items():
                            f.write(f"| {ident} | {nome} |\n")
                
                logger.info(f"🔒 Anonimização concluída")
        except Exception as e:
            logger.error(f"Erro na anonimização: {e}")
            # Continua mesmo se anonimização falhar

        send_progress_ws(operation_id, 7, 'Atualizando tabela...', 90)
        path_triagem = current_app.config['PATH_TRIAGEM']
        suspeitos_str = ', '.join(suspeitos)
        nova_linha = f"| {numero} | {tema} | {data_dist} | {responsavel} | {status} | {ultima_att} | {suspeitos_str} | {comentarios} |\n"

        if not os.path.exists(path_triagem):
            with open(path_triagem, 'w', encoding='utf-8') as f:
                f.write("# Tabela de Processos\n\n")
                f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
        
        with open(path_triagem, 'a', encoding='utf-8') as f:
            f.write(nova_linha)

        logger.info(f" Processo {numero} salvo na tabela")
        send_progress_ws(operation_id, 8, 'Finalizado com sucesso!', 100)

    except Exception as e:
        logger.error(f"Erro no processamento de {operation_id}: {str(e)}")
        send_progress_ws(operation_id, 0, f"Erro: {str(e)}", 0)

def processar_sem_progresso(data, operation_id):
    numero = limpar(data.get('numeroProcesso'))
    markdown = limpar(data.get('markdown'))
    dat_base64 = data.get('dat')

    if not numero or not markdown:
        return

    path_triagem = current_app.config['PATH_TRIAGEM']
    pasta_dest = current_app.config['PASTA_DESTINO']
    pasta_dat = current_app.config['PASTA_DAT']
    tema = limpar(data.get('tema'))
    data_dist = limpar(data.get('dataDistribuicao'))
    responsavel = limpar(data.get('responsavel'))
    status = limpar(data.get('status'))
    comentarios = limpar(data.get('comentarios'))
    ultima_att = datetime.now().strftime('%Y-%m-%d')

    suspeitos = []
    try:
        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
    except:
        pass

    nome_base = numero.replace('/', '-')
    os.makedirs(pasta_dest, exist_ok=True)
    os.makedirs(pasta_dat, exist_ok=True)
    with open(os.path.join(pasta_dest, f"{nome_base}.md"), 'w', encoding='utf-8') as f:
        f.write(markdown)
    if dat_base64:
        with open(os.path.join(pasta_dat, f"{nome_base}.dat"), 'w', encoding='utf-8') as f:
            f.write(dat_base64)

    suspeitos_str = ', '.join(suspeitos)
    nova_linha = f"| {numero} | {tema} | {data_dist} | {responsavel} | {status} | {ultima_att} | {suspeitos_str} | {comentarios} |\n"
    if not os.path.exists(path_triagem):
        with open(path_triagem, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
    with open(path_triagem, 'a', encoding='utf-8') as f:
        f.write(nova_linha)

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
        "comentarios": limpar(data.get('comentarios', ''))
    }
    processos.append(atualizado)

    with open(path_triagem, 'w', encoding='utf-8') as f:
        f.write("# Tabela de Processos\n\n")
        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
        for p in processos:
            f.write(f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} | {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('comentarios', '')} |\n")

def deletar_processo_por_numero(numero):
    path_triagem = current_app.config['PATH_TRIAGEM']
    pasta_dest = current_app.config['PASTA_DESTINO']
    processos = extrair_tabela_md(path_triagem)
    processos = [p for p in processos if p['numeroProcesso'] != numero]
    with open(path_triagem, 'w', encoding='utf-8') as f:
        f.write("# Tabela de Processos\n\n")
        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
        for p in processos:
            f.write(f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} | {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('comentarios', '')} |\n")
    caminho_md = os.path.join(pasta_dest, f"{numero.replace('/', '-')}.md")
    if os.path.exists(caminho_md):
        os.remove(caminho_md)

def obter_dat_por_numero(numero):
    caminho = os.path.join(current_app.config['PASTA_DAT'], f"{numero.replace('/', '-')}.dat")
    if not os.path.exists(caminho):
        raise FileNotFoundError("Arquivo .dat não encontrado")
    with open(caminho, 'r', encoding='utf-8') as f:
        return f.read()

def get_processos():
    path_triagem = current_app.config['PATH_TRIAGEM']
    if not os.path.exists(path_triagem):
        return []
    return extrair_tabela_md(path_triagem)
