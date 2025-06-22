# routes/triagem.py
from flask import Blueprint, request, jsonify, current_app
from utils.triagem import (
    get_processos,
    processar_com_progresso,
    atualizar_processo,
    deletar_processo_por_numero,
    obter_dat_por_numero
)
from utils.rag import handle_rag_analyze_processo
from utils.auto_setup import setup_environment
from utils.auxiliar import limpar, get_anonimizador
from utils.suspeicao import encontrar_suspeitos
from utils.progress_step import send_progress_ws
import uuid
import threading
import time
import os
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

triagem_bp = Blueprint('triagem', __name__)
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()

triagem_bp.operation_sockets = {}

ANONIMIZACAO_ATIVA = True 

@triagem_bp.route('/triagem', methods=['GET'])
def listar_processos():
    processos = get_processos()
    return jsonify(processos), 200

@triagem_bp.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    print("📝 Solicitação POST /triagem/form recebida")
    try:
        data = request.get_json()
        print(f"📄 Dados recebidos")
    
        operation_id = str(uuid.uuid4())
        print(f"🆔 Operation ID gerado: {operation_id}")
        
        operation_sockets = getattr(triagem_bp, 'operation_sockets', {})
        
        def processar_em_background():
            """Processa o documento em background"""
            print(f"🔄 Iniciando processamento em background para {operation_id}")
            
            time.sleep(0.5)
            
            # Verifica se o frontend se registrou
            if operation_id not in operation_sockets:
                print(f" Frontend ainda não registrado para {operation_id}, aguardando...")
                for i in range(10):
                    time.sleep(0.5)
                    if operation_id in operation_sockets:
                        break
                    print(f"⏳ Aguardando registro... {i+1}/10")
            
            # MUDANÇA: sempre continua, não chama processar_sem_progresso
            if operation_id not in operation_sockets:
                print(f" Frontend não registrado, mas continuando com progresso")
            else:
                print(f" Frontend registrado! Iniciando processamento com progresso...")
            
            try:
                # ETAPA 1: VALIDAÇÃO INICIAL
                send_progress_ws(operation_id, 1, 'Validando dados do processo...', 10)
                time.sleep(0.3)
                
                numero = limpar(data.get('numeroProcesso'))
                tema = limpar(data.get('tema'))
                data_dist = limpar(data.get('dataDistribuicao'))
                responsavel = limpar(data.get('responsavel'))
                status = limpar(data.get('status'))
                markdown = limpar(data.get('markdown'))
                comentarios = limpar(data.get('comentarios'))
                dat_base64 = data.get('dat')

                ultima_att = datetime.now().strftime('%Y-%m-%d')
                
                if not numero:
                    send_progress_ws(operation_id, 0, 'Erro: Número do processo é obrigatório', 0)
                    print("Número do processo obrigatório")
                    return
                
                if not markdown or not numero:
                    send_progress_ws(operation_id, 0, 'Erro: Campos obrigatórios ausentes', 0)
                    logger.warning(" Campos obrigatórios ausentes")
                    return
                
                logger.info(f"📄 Processando processo: {numero}")
                
                # === PASSO 2: BUSCA SUSPEITOS ===
                send_progress_ws(operation_id, 2, 'Analisando suspeição e impedimento no documento...', 25)
                time.sleep(0.5)
                
                suspeitos = []
                if markdown and markdown.strip():
                    try:
                        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                        if suspeitos:
                            for i, suspeito in enumerate(suspeitos, 1):
                                print(f"   {i}. {suspeito}")
                        else:
                            print("   Nenhum suspeito detectado")
                    except Exception as e:
                        print(f"Erro na busca de suspeitos: {e}")
                        suspeitos = []
                
                logger.info(f"🔍 Suspeitos encontrados: {suspeitos}")

                # === PASSO 3: PREPARANDO ESTRUTURA ===
                send_progress_ws(operation_id, 3, 'Preparando estrutura de arquivos...', 40)
                time.sleep(0.3)
                
                nome_arquivo_base = numero.replace('/', '-')
                os.makedirs(PASTA_DESTINO, exist_ok=True)
                os.makedirs(PASTA_DAT, exist_ok=True)

                caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
                caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")

                # === PASSO 4: SALVA ARQUIVOS ORIGINAIS ===
                print("📁 [PASSO 4] Salvando arquivos originais...")
                
                if markdown and markdown.strip():
                    send_progress_ws(operation_id, 4, 'Salvando documento processado...', 55)
                    time.sleep(0.4)
                    with open(caminho_md, 'w', encoding='utf-8') as f:
                        f.write(markdown)
                    print(f"💾 Markdown salvo: {caminho_md}")
                    logger.info(f"Markdown salvo: {caminho_md}")

                print("💾 [PASSO 4.1] Processando arquivo DAT...")
                if dat_base64 and dat_base64.strip():
                    send_progress_ws(operation_id, 5, 'Salvando arquivo original (DAT)...', 60)
                    time.sleep(0.3)
                    
                    try:
                        # Salva o arquivo DAT
                        with open(caminho_dat, 'w', encoding='utf-8') as f:
                            f.write(dat_base64)
                        print(f" Arquivo DAT salvo com sucesso: {caminho_dat}")
                        logger.info(f"Arquivo DAT salvo: {caminho_dat}")
                        
                        # Verifica se foi salvo corretamente
                        if os.path.exists(caminho_dat):
                            tamanho_arquivo = os.path.getsize(caminho_dat)
                            print(f"📊 Tamanho do arquivo DAT: {tamanho_arquivo} bytes")
                        else:
                            print(f"ERRO: Arquivo DAT não foi criado!")
                            
                    except Exception as e:
                        print(f"Erro ao salvar arquivo DAT: {e}")
                        logger.error(f"Erro ao salvar arquivo DAT: {e}")
                else:
                    print("ℹ️ Nenhum arquivo DAT fornecido para salvar")

                # === PASSO 5: ANONIMIZAÇÃO ===
                print("🔒 [PASSO 5] Iniciando anonimização automática otimizada...")
                send_progress_ws(operation_id, 6, 'Executando anonimização automática...', 75)
                time.sleep(0.4)
                
                arquivos_anonimizados = {}
                total_substituicoes = 0
                tempo_anonimizacao = 0
                
                if ANONIMIZACAO_ATIVA and markdown and markdown.strip():
                    try:
                        inicio = time.time()
                        print(f"🔄 Executando anonimização otimizada para processo {numero}")
                        
                        anonimizador = get_anonimizador()
                        if anonimizador is None:
                            print("ERRO: Anonimizador não pôde ser inicializado!")
                            raise Exception("Anonimizador não disponível")
                        
                        print(" Anonimizador carregado com sucesso")
                        print("📋 Carregando mapeamento de suspeitos...")
                        mapa_suspeitos = anonimizador.carregar_suspeitos_mapeados("./utils/suspeitos.txt")
                        print(f"📊 Suspeitos carregados: {len(mapa_suspeitos)} mapeamentos")
                        print("🔒 Iniciando processo de anonimização...")
                        
                        texto_anonimizado, mapa_reverso = anonimizador.anonimizar_com_identificadores(
                            markdown, mapa_suspeitos, debug=False  # Mude para True se quiser ver debug
                        )
                        
                        pasta_anon = os.path.join(PASTA_DESTINO, "anonimizados")
                        pasta_mapas = os.path.join(PASTA_DESTINO, "mapas")
                        os.makedirs(pasta_anon, exist_ok=True)
                        os.makedirs(pasta_mapas, exist_ok=True)
                        send_progress_ws(operation_id, 8, 'Salvando versão anonimizada...', 80)
                        caminho_md_anon = os.path.join(pasta_anon, f"{nome_arquivo_base}_anon.md")
                        try:
                            with open(caminho_md_anon, "w", encoding="utf-8") as f:
                                f.write(texto_anonimizado)
                            print(f" Markdown anonimizado salvo: {caminho_md_anon}")
                            
                            # Verifica se foi salvo
                            if os.path.exists(caminho_md_anon):
                                tamanho = os.path.getsize(caminho_md_anon)
                                print(f"📊 Tamanho do arquivo anonimizado: {tamanho} bytes")
                                arquivos_anonimizados["md"] = caminho_md_anon
                            else:
                                print(f"ERRO: Arquivo anonimizado não foi criado!")
                                
                        except Exception as e:
                            print(f"Erro ao salvar markdown anonimizado: {e}")
                            raise
                        
                        caminho_mapa = None
                        if mapa_reverso and len(mapa_reverso) > 0:
                            caminho_mapa = os.path.join(pasta_mapas, f"{nome_arquivo_base}_mapa.md")
                            with open(caminho_mapa, "w", encoding="utf-8") as f:
                                f.write("| Identificador | Nome Original |\n")
                                f.write("|---------------|----------------|\n")
                                for ident, nome in sorted(mapa_reverso.items()):
                                    f.write(f"| {ident} | {nome} |\n")
                        
                        total_substituicoes = len(mapa_reverso)
                        tempo_anonimizacao = round(time.time() - inicio, 2)
                        
                        arquivos_anonimizados = {
                            "md": caminho_md_anon,
                            "mapa": caminho_mapa if caminho_mapa else None
                        }
                        
                        print(f" Anonimização concluída em {tempo_anonimizacao}s")
                        print(f"📊 Total de substituições: {total_substituicoes}")
                        logger.info(f" Anonimização concluída: {total_substituicoes} substituições em {tempo_anonimizacao}s")
                        
                    except Exception as e:
                        print(f"Erro durante anonimização otimizada: {e}")
                        logger.error(f"Erro durante anonimização otimizada: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    if not ANONIMIZACAO_ATIVA:
                        print("ℹ️ Anonimização desativada")
                    elif not markdown or not markdown.strip():
                        print("ℹ️ Sem conteúdo markdown para anonimizar")

                send_progress_ws(operation_id, 7, 'Atualizando tabela de triagem...', 90)
                print("📋 [PASSO 6] Atualizando tabela de triagem...")
                time.sleep(0.3)  # MUDANÇA: delay menor
                
                suspeitos_str = ', '.join(suspeitos) if suspeitos else ''

                nova_linha = (
                    f"| {numero} "
                    f"| {tema} "
                    f"| {data_dist} "
                    f"| {responsavel} "
                    f"| {status} "
                    f"| {ultima_att} "
                    f"| {suspeitos_str} "
                    f"| {comentarios} |\n"
                )

                if not os.path.exists(PATH_TRIAGEM):
                    print(f"📄 Criando novo arquivo de triagem: {PATH_TRIAGEM}")
                    logger.info(f"📄 Criando novo arquivo de triagem: {PATH_TRIAGEM}")
                    with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                        f.write("# Tabela de Processos\n\n")
                        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")

                with open(PATH_TRIAGEM, 'r', encoding='utf-8') as f:
                    linhas = f.readlines()

                indice_separador = next(
                    (i for i, linha in enumerate(linhas) if re.match(r'^\|\s*-+\s*\|', linha.strip())),
                    None
                )

                if indice_separador is not None:
                    while indice_separador + 1 < len(linhas) and not linhas[indice_separador + 1].strip().startswith('|'):
                        del linhas[indice_separador + 1]
                    linhas.insert(indice_separador + 1, nova_linha)
                else:
                    linhas += [
                        "| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n",
                        "|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n",
                        nova_linha
                    ]

                with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                    f.writelines(linhas)
                
                send_progress_ws(operation_id, 9, 'Processo adicionado com sucesso!', 100)
                time.sleep(0.5)
                
                print(f"🎉 Processo {numero} salvo com sucesso")
                print(f"    📊 Suspeitos detectados: {len(suspeitos)}")
                print(f"    🔄 Substituições anonimização: {total_substituicoes}")
                print(f"    ⏱️ Tempo de anonimização: {tempo_anonimizacao}s")
                print(f"    📁 Arquivos anonimizados: {len([a for a in arquivos_anonimizados.values() if a])}")
                
                logger.info(f" Processo {numero} salvo com sucesso")
                
                operation_sockets.pop(operation_id, None)
                
            except Exception as e:
                send_progress_ws(operation_id, 0, f'Erro: {str(e)}', 0)
                print(f"Erro no processamento: {e}")
                import traceback
                traceback.print_exc()
                operation_sockets.pop(operation_id, None)
        
        thread = threading.Thread(target=processar_em_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "success": True,
            "message": "Processamento iniciado em background",
            "operation_id": operation_id
        }), 200
        
    except Exception as e:
        print(f"Erro em POST /triagem/form: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@triagem_bp.route('/triagem/<numero>', methods=['PUT'])
def editar_processo(numero):
    data = request.get_json()
    atualizar_processo(numero, data)
    return jsonify({'message': 'Processo atualizado com sucesso'}), 200

@triagem_bp.route('/triagem/<numero>', methods=['DELETE'])
def deletar_processo(numero):
    deletar_processo_por_numero(numero)
    return jsonify({'message': 'Processo deletado com sucesso'}), 200

@triagem_bp.route('/triagem/<numero>/dat', methods=['GET'])
def obter_dat(numero):
    try:
        conteudo = obter_dat_por_numero(numero)
        return jsonify({'dat': conteudo}), 200
    except FileNotFoundError:
        return jsonify({'error': 'Arquivo não encontrado'}), 404

@triagem_bp.route('/triagem/<numero>/analyze', methods=['POST'])
def analisar_processo(numero):
    return handle_rag_analyze_processo(numero)