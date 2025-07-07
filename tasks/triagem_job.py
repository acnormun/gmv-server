from redis import Redis
import time
import os
import logging
from datetime import datetime
from utils.auxiliar import limpar, get_anonimizador
from utils.progress_step import send_progress_ws
from utils.extrair_metadados_processo import extrair_e_formatar_metadados
from utils.suspeicao import encontrar_suspeitos
from utils.auto_setup import setup_environment
from adaptive_rag import rag_system

logger = logging.getLogger(__name__)

redis = Redis()
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()
ANONIMIZACAO_ATIVA = True
PROCESSADOR_PJE_ATIVO = True


def processar_processo_em_background(data, operation_id):
    numero = limpar(data.get('numeroProcesso'))
    tema = limpar(data.get('tema'))
    data_dist = limpar(data.get('dataDistribuicao'))
    responsavel = limpar(data.get('responsavel'))
    status = limpar(data.get('status'))
    markdown = limpar(data.get('markdown'))
    comentarios = limpar(data.get('comentarios'))
    dat_base64 = data.get('dat')

    lock_key = f"lock:processo:{numero}"
    lock = redis.lock(lock_key, timeout=600)

    if not lock.acquire(blocking=True, blocking_timeout=5):
        print(f"⚠️ Processo {numero} já está em execução. Pulando.")
        send_progress_ws(operation_id, 0, f"Erro: Processo {numero} já está sendo processado", 0)
        return

    try:
        print(f"🚀 [JOB] Iniciando processamento do processo {numero}")
        send_progress_ws(operation_id, 1, 'Validando dados do processo...', 5)
        time.sleep(0.3)

        ultima_att = datetime.now().strftime('%Y-%m-%d')

        if not numero:
            send_progress_ws(operation_id, 0, 'Erro: Número do processo é obrigatório', 0)
            return

        if not markdown and not dat_base64:
            send_progress_ws(operation_id, 0, 'Erro: É necessário fornecer markdown OU arquivo PDF', 0)
            return

        resultado_pje = None
        primeiro_md_pje = None
        caminho_md = os.path.join(PASTA_DESTINO, f"{numero.replace('/', '-')}.md")
        caminho_dat = os.path.join(PASTA_DAT, f"{numero.replace('/', '-')}.dat")

        if dat_base64 and dat_base64.strip():
            send_progress_ws(operation_id, 2, 'Processando PDF do PJe...', 15)
            try:
                from utils.processador_pje_integrado import processar_pje_com_progresso
                resultado_pje = processar_pje_com_progresso(
                    dat_base64, numero, PASTA_DESTINO, operation_id
                )
                if resultado_pje and resultado_pje['sucesso']:
                    markdowns = resultado_pje['arquivos_gerados']['markdowns']
                    if markdowns:
                        with open(markdowns[0], 'r', encoding='utf-8') as f:
                            markdown_pje = f.read()
                        if not markdown:
                            markdown = markdown_pje
            except Exception as e:
                send_progress_ws(operation_id, 2, f'Aviso: Processamento PJe falhou, continuando...', 20)
                if not markdown:
                    primeiro_md_pje = markdowns[0]
                    with open(primeiro_md_pje, 'r', encoding='utf-8') as f:
                        f.write(dat_base64)
        elif not markdown:
            send_progress_ws(operation_id, 0, 'Erro: É necessário markdown OU arquivo PDF válido', 0)
            return

        send_progress_ws(operation_id, 3, 'Analisando suspeição e impedimento...', 25)
        suspeitos = []
        if markdown:
            try:
                suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
            except Exception:
                suspeitos = []

        send_progress_ws(operation_id, 4, 'Preparando estrutura de arquivos...', 35)
        os.makedirs(PASTA_DESTINO, exist_ok=True)
        os.makedirs(PASTA_DAT, exist_ok=True)
        if primeiro_md_pje:
            caminho_md = primeiro_md_pje

        send_progress_ws(operation_id, 5, 'Salvando arquivos processados...', 45)
        campos_extraidos = 0
        markdown_com_metadados = markdown

        try:
            metadados_dict, front_matter = extrair_e_formatar_metadados(markdown)
            campos_extraidos = len([v for v in metadados_dict.values() if v])
            if campos_extraidos:
                if primeiro_md_pje:
                    with open(caminho_md, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    meta_lines = front_matter.splitlines()[1:-1]
                    if original_content.startswith('---'):
                        end_idx = original_content.find('\n---', 3)
                        if end_idx != -1:
                            before = original_content[:end_idx]
                            after = original_content[end_idx:]
                            markdown_com_metadados = before + '\n' + '\n'.join(meta_lines) + after
                        else:
                            markdown_com_metadados = front_matter + "\n\n" + original_content
                    else:
                        markdown_com_metadados = front_matter + "\n\n" + original_content
                else:
                    markdown_com_metadados = front_matter + "\n\n" + markdown
        except Exception:
            pass

        with open(caminho_md, 'w', encoding='utf-8') as f:
            f.write(markdown_com_metadados)

        if RAG_DISPONIVEL and rag_system.is_initialized:
            try:
                from langchain.schema import Document
                doc = Document(
                    page_content=markdown_com_metadados,
                    metadata={
                        "filename": os.path.basename(caminho_md),
                        "source": caminho_md,
                        "numero_processo": numero,
                        "tem_metadados": campos_extraidos > 0,
                        "processado_pje": resultado_pje is not None
                    }
                )
                rag_system.vector_store.add_documents([doc])
                rag_system.documents.append(doc)
            except Exception as e:
                logger.warning(f"Erro ao atualizar RAG: {e}")

        if dat_base64 and not PROCESSADOR_PJE_ATIVO:
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)

        send_progress_ws(operation_id, 7, 'Executando anonimização automática...', 65)
        arquivos_anonimizados = {}
        total_substituicoes = 0
        tempo_anonimizacao = 0

        if ANONIMIZACAO_ATIVA and markdown:
            try:
                inicio = time.time()
                anonimizador = get_anonimizador()
                mapa_suspeitos = anonimizador.carregar_suspeitos_mapeados("./utils/suspeitos.txt")
                texto_anonimizado, mapa_reverso = anonimizador.anonimizar_com_identificadores(markdown, mapa_suspeitos)

                pasta_anon = os.path.join(PASTA_DESTINO, "anonimizados")
                pasta_mapas = os.path.join(PASTA_DESTINO, "mapas")
                os.makedirs(pasta_anon, exist_ok=True)
                os.makedirs(pasta_mapas, exist_ok=True)

                caminho_md_anon = os.path.join(pasta_anon, f"{numero.replace('/', '-')}_anon.md")
                with open(caminho_md_anon, "w", encoding="utf-8") as f:
                    f.write(texto_anonimizado)
                arquivos_anonimizados["md"] = caminho_md_anon

                if mapa_reverso:
                    caminho_mapa = os.path.join(pasta_mapas, f"{numero.replace('/', '-')}_mapa.md")
                    with open(caminho_mapa, "w", encoding="utf-8") as f:
                        f.write("| Identificador | Nome Original |\n")
                        f.write("|---------------|----------------|\n")
                        for ident, nome in sorted(mapa_reverso.items()):
                            f.write(f"| {ident} | {nome} |\n")
                    arquivos_anonimizados["mapa"] = caminho_mapa

                total_substituicoes = len(mapa_reverso)
                tempo_anonimizacao = round(time.time() - inicio, 2)

            except Exception as e:
                logger.error(f"Erro na anonimização: {e}")

        send_progress_ws(operation_id, 8, 'Atualizando tabela de triagem...', 85)
        suspeitos_str = ', '.join(suspeitos) if suspeitos else ''
        nova_linha = (
            f"| {numero} | {tema} | {data_dist} | {responsavel} | {status} | {ultima_att} | {suspeitos_str} | {comentarios} |\n"
        )

        if not os.path.exists(PATH_TRIAGEM):
            with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                f.write("# Tabela de Processos\n\n")
                f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")

        with open(PATH_TRIAGEM, 'a', encoding='utf-8') as f:
            f.write(nova_linha)

        send_progress_ws(operation_id, 9, 'Enviando notificação por email...', 95)
        try:
            from utils.email_notification import enviar_notificacao_processo
            dados_notificacao = {
                'numero': numero,
                'tema': tema,
                'data_dist': data_dist,
                'responsavel': responsavel,
                'status': status,
                'comentarios': comentarios,
                'suspeitos': suspeitos,
                'processamento_pje': resultado_pje is not None
            }
            enviar_notificacao_processo(dados_notificacao)
        except Exception as e:
            logger.warning(f"Erro ao enviar email: {e}")

        send_progress_ws(operation_id, 10, 'Processo concluído com sucesso!', 100)

        print(f"✅ Processo {numero} finalizado com sucesso")

    except Exception as e:
        send_progress_ws(operation_id, 0, f"Erro: {str(e)}", 0)
        logger.error(f"Erro geral no processamento: {e}")
    finally:
        lock.release()
