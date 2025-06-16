from flask import Flask, jsonify, request
import pandas as pd
import re
import os
import uuid
import time
import signal
import sys
import logging
from datetime import datetime
from pathlib import Path
from utils.anonimizacao import AnonimizadorOtimizado
from utils.auxiliar import extrair_tabela_md, get_anonimizador, limpar
from utils.auto_setup import setup_environment
from utils.progress_step import send_progress_ws
from utils import progress_step
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sua_chave_secreta'

socketio = SocketIO(app, 
                   cors_allowed_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   logger=True,
                   engineio_logger=False)
operation_sockets = {}
progress_step.init_progress(socketio, operation_sockets)

try:
    from utils.anonimizacao import AnonimizadorOtimizado
    ANONIMIZACAO_ATIVA = True
    print("Módulo de anonimização carregado")
except ImportError as e:
    ANONIMIZACAO_ATIVA = False
    print(f"Anonimização desabilitada: {e}")

# ==========================================
#  INICIALIZAÇÃO DO SETUP
# ==========================================

print(" GMV SISTEMA - INICIALIZAÇÃO COM LIMPEZA DE CACHE")
print("=" * 60)
print(f" Diretório de trabalho: {os.getcwd()}")
print(f" Procurando .env em: {os.path.abspath('.env')}")

# Executa setup automático COM LIMPEZA FORÇADA
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()

# Verificação final
if not PATH_TRIAGEM or not PASTA_DESTINO:
    print("\n ERRO CRÍTICO: Não foi possível configurar variáveis de ambiente!")
    print(" DEPURAÇÃO:")
    print("   1. Verifique se você tem permissão para criar arquivos neste diretório")
    print("   2. Verifique se o arquivo .env foi criado corretamente")
    print("   3. Tente executar como administrador")
    print(f"   4. Arquivo .env deveria estar em: {os.path.abspath('.env')}")
    sys.exit(1)

# TESTE FINAL - Confirma que variáveis corretas estão sendo usadas
print(" TESTE FINAL DE VERIFICAÇÃO:")
print("=" * 40)
final_vars = {
    'PATH_TRIAGEM': os.getenv('PATH_TRIAGEM'),
    'PASTA_DESTINO': os.getenv('PASTA_DESTINO'), 
    'PASTA_DAT': os.getenv('PASTA_DAT')
}

for var_name, var_value in final_vars.items():
    print(f"{var_name} = {var_value}")
    
    # Verifica se o caminho é absoluto ou relativo
    if var_value:
        abs_path = os.path.abspath(var_value)
        print(f"   Caminho absoluto: {abs_path}")

print("\nCONFIRMAÇÃO:")
print(f"Cache de variáveis antigas foi limpo")
print(f"Arquivo .env atual foi carregado")
print(f"{len([v for v in final_vars.values() if v])} variáveis essenciais definidas")
print("=" * 60)

# ==========================================
# 🌐 CONFIGURAÇÃO DO FLASK
# ==========================================

# Importa utils só depois do setup
try:
    from utils.suspeicao import encontrar_suspeitos
    print("Módulo de suspeição carregado")
except ImportError as e:
    print(f"Aviso: Módulo de suspeição não encontrado: {e}")
    def encontrar_suspeitos(texto, arquivo):
        return []

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuração CORS
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
    print("CORS configurado")
except ImportError:
    print("flask-cors não instalado - CORS pode não funcionar")

# Log de inicialização
logger.info(f"Servidor Flask iniciando com PID: {os.getpid()}")
logger.info(f"PATH_TRIAGEM: {PATH_TRIAGEM}")
logger.info(f" PASTA_DESTINO: {PASTA_DESTINO}")
logger.info(f" PASTA_DAT: {PASTA_DAT}")

# ==========================================
# 🌐 ROTAS DA API
# ==========================================

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "pid": os.getpid(),
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "env_vars": {
            "PATH_TRIAGEM": PATH_TRIAGEM,
            "PASTA_DESTINO": PASTA_DESTINO,
            "PASTA_DAT": PASTA_DAT
        }
    }), 200

@app.route('/process-info', methods=['GET'])
def process_info():
    return jsonify({
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, 'getppid') else None,
        "cwd": os.getcwd(),
        "env_vars": {
            "PATH_TRIAGEM": PATH_TRIAGEM,
            "PASTA_DESTINO": PASTA_DESTINO,
            "PASTA_DAT": PASTA_DAT
        }
    }), 200

@app.route('/triagem', methods=['GET'])
def get_processos():
    logger.info(" Solicitação GET /triagem recebida")
    try:
        dados = extrair_tabela_md(PATH_TRIAGEM)
        logger.info(f" Retornando {len(dados)} processos")
        return jsonify(dados)
    except Exception as e:
        logger.error(f" Erro em GET /triagem: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    print("📝 Solicitação POST /triagem/form recebida")
    try:
        data = request.get_json()
        print(f"📄 Dados recebidos")
        
        # CRIA O OPERATION_ID PRIMEIRO!
        operation_id = str(uuid.uuid4())
        print(f"🆔 Operation ID gerado: {operation_id}")
        
        def processar_em_background():
            """Processa o documento em background"""
            print(f"🔄 Iniciando processamento em background para {operation_id}")
            
            # Pequeno delay para dar tempo do frontend se registrar
            time.sleep(1)
            
            # Verifica se o frontend se registrou
            if operation_id not in operation_sockets:
                print(f"⚠️ Frontend ainda não registrado para {operation_id}, aguardando...")
                # Aguarda mais um pouco
                for i in range(5):  # Aguarda até 5 segundos
                    time.sleep(1)
                    if operation_id in operation_sockets:
                        break
                    print(f"⏳ Aguardando registro... {i+1}/5")
            
            if operation_id not in operation_sockets:
                print(f"❌ Frontend não se registrou para {operation_id} após 6 segundos")
                # Continua processamento sem progresso
                processar_sem_progresso(data, operation_id)
                return
            
            print(f"✅ Frontend registrado! Iniciando processamento com progresso...")
            
            try:
                # ETAPA 1: VALIDAÇÃO INICIAL
                send_progress_ws(operation_id, 1, 'Validando dados do processo...', 10)
                time.sleep(0.5) 
                
                numero = limpar(data.get('numeroProcesso'))
                tema = limpar(data.get('tema'))
                data_dist = limpar(data.get('dataDistribuicao'))
                responsavel = limpar(data.get('responsavel'))
                status = limpar(data.get('status'))
                markdown = limpar(data.get('markdown'))
                comentarios = limpar(data.get('comentarios'))
                dat_base64 = data.get('dat')
                
                # DATA ATUAL AUTOMÁTICA - sempre seta data de hoje
                from datetime import datetime
                ultima_att = datetime.now().strftime('%Y-%m-%d')
                
                # Validação básica
                if not numero:
                    send_progress_ws(operation_id, 0, 'Erro: Número do processo é obrigatório', 0)
                    print("❌ Número do processo obrigatório")
                    return
                
                if not markdown or not numero:
                    send_progress_ws(operation_id, 0, 'Erro: Campos obrigatórios ausentes', 0)
                    logger.warning("⚠️ Campos obrigatórios ausentes")
                    return
                
                logger.info(f"📄 Processando processo: {numero}")
                
                # === PASSO 2: BUSCA SUSPEITOS NO TEXTO ORIGINAL ===
                send_progress_ws(operation_id, 2, 'Analisando suspeição e impedimento no documento...', 25)
                time.sleep(1)
                
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
                        print(f"❌ Erro na busca de suspeitos: {e}")
                        suspeitos = []
                
                logger.info(f"🔍 Suspeitos encontrados: {suspeitos}")

                # === PASSO 3: PREPARANDO ESTRUTURA ===
                send_progress_ws(operation_id, 3, 'Preparando estrutura de arquivos...', 40)
                time.sleep(0.5)
                
                nome_arquivo_base = numero.replace('/', '-')
                os.makedirs(PASTA_DESTINO, exist_ok=True)
                os.makedirs(PASTA_DAT, exist_ok=True)

                caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
                caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")

                # === PASSO 4: SALVA ARQUIVOS ORIGINAIS ===
                print("📁 [PASSO 4] Salvando arquivos originais...")
                
                # Salva markdown se fornecido
                if markdown and markdown.strip():
                    send_progress_ws(operation_id, 4, 'Salvando documento processado...', 55)
                    time.sleep(0.8)
                    with open(caminho_md, 'w', encoding='utf-8') as f:
                        f.write(markdown)
                    print(f"💾 Markdown salvo: {caminho_md}")
                    logger.info(f"Markdown salvo: {caminho_md}")

                # Salva .dat como base64 se enviado
                if dat_base64 and dat_base64.strip():
                    send_progress_ws(operation_id, 5, 'Salvando arquivo original...', 65)
                    time.sleep(0.5)
                    with open(caminho_dat, 'w', encoding='utf-8') as f:
                        f.write(dat_base64)
                    print(f"💾 Arquivo DAT salvo: {caminho_dat}")

                # === PASSO 5: ANONIMIZAÇÃO AUTOMÁTICA OTIMIZADA ===
                print("🔒 [PASSO 5] Iniciando anonimização automática otimizada...")
                send_progress_ws(operation_id, 6, 'Executando anonimização automática...', 75)
                time.sleep(0.7)
                
                arquivos_anonimizados = {}
                total_substituicoes = 0
                tempo_anonimizacao = 0
                
                if ANONIMIZACAO_ATIVA and markdown and markdown.strip():
                    try:
                        inicio = time.time()
                        print(f"🔄 Executando anonimização otimizada para processo {numero}")
                        
                        # Usa a instância otimizada do anonimizador
                        anonimizador = get_anonimizador()
                        
                        # Carrega mapeamento de suspeitos (com cache)
                        mapa_suspeitos = anonimizador.carregar_suspeitos_mapeados("utils/suspeitos.txt")
                        
                        # Executa anonimização otimizada
                        texto_anonimizado, mapa_reverso = anonimizador.anonimizar_com_identificadores(
                            markdown, mapa_suspeitos
                        )
                        
                        # Salva arquivos anonimizados
                        pasta_anon = os.path.join(PASTA_DESTINO, "anonimizados")
                        pasta_mapas = os.path.join(PASTA_DESTINO, "mapas")
                        os.makedirs(pasta_anon, exist_ok=True)
                        os.makedirs(pasta_mapas, exist_ok=True)
                        
                        # Salva texto anonimizado
                        caminho_md_anon = os.path.join(pasta_anon, f"{nome_arquivo_base}_anon.md")
                        with open(caminho_md_anon, "w", encoding="utf-8") as f:
                            f.write(texto_anonimizado)
                        
                        # Salva mapa de substituições se houver
                        caminho_mapa = None
                        if mapa_reverso:
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
                        
                        print(f"✅ Anonimização concluída em {tempo_anonimizacao}s")
                        print(f"📊 Total de substituições: {total_substituicoes}")
                        logger.info(f"✅ Anonimização concluída: {total_substituicoes} substituições em {tempo_anonimizacao}s")
                        
                    except Exception as e:
                        print(f"❌ Erro durante anonimização otimizada: {e}")
                        logger.error(f"❌ Erro durante anonimização otimizada: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    if not ANONIMIZACAO_ATIVA:
                        print("ℹ️ Anonimização desativada")
                    elif not markdown or not markdown.strip():
                        print("ℹ️ Sem conteúdo markdown para anonimizar")

                # === PASSO 6: ATUALIZA TABELA DE TRIAGEM ===
                send_progress_ws(operation_id, 7, 'Atualizando tabela de triagem...', 90)
                print("📋 [PASSO 6] Atualizando tabela de triagem...")
                time.sleep(0.5)
                
                # Converte lista de suspeitos para string
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
                
                # === FINALIZAÇÃO ===
                send_progress_ws(operation_id, 8, 'Processo adicionado com sucesso!', 100)
                time.sleep(0.5)
                
                print(f"🎉 Processo {numero} salvo com sucesso")
                print(f"    📊 Suspeitos detectados: {len(suspeitos)}")
                print(f"    🔄 Substituições anonimização: {total_substituicoes}")
                print(f"    ⏱️ Tempo de anonimização: {tempo_anonimizacao}s")
                print(f"    📁 Arquivos anonimizados: {len([a for a in arquivos_anonimizados.values() if a])}")
                
                logger.info(f"✅ Processo {numero} salvo com sucesso")
                
                # Remove da lista de sockets após 5 segundos
                import threading
                threading.Timer(5.0, lambda: operation_sockets.pop(operation_id, None)).start()
                
            except Exception as e:
                send_progress_ws(operation_id, 0, f'Erro: {str(e)}', 0)
                print(f"❌ Erro no processamento: {e}")
                import traceback
                traceback.print_exc()
        
        def processar_sem_progresso(data, operation_id):
            """Fallback: processa sem enviar progresso (cópia da lógica original)"""
            print(f"🔄 Processando sem progresso para {operation_id}")
            try:
                numero = limpar(data.get('numeroProcesso'))
                tema = limpar(data.get('tema'))
                data_dist = limpar(data.get('dataDistribuicao'))
                responsavel = limpar(data.get('responsavel'))
                status = limpar(data.get('status'))
                markdown = limpar(data.get('markdown'))
                comentarios = limpar(data.get('comentarios'))
                dat_base64 = data.get('dat')
                
                from datetime import datetime
                ultima_att = datetime.now().strftime('%Y-%m-%d')
                
                if not numero or not markdown:
                    print("❌ Campos obrigatórios ausentes")
                    return
                
                # Busca suspeitos
                suspeitos = []
                if markdown and markdown.strip():
                    try:
                        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                    except Exception as e:
                        print(f"❌ Erro na busca de suspeitos: {e}")
                
                # Salva arquivos
                nome_arquivo_base = numero.replace('/', '-')
                os.makedirs(PASTA_DESTINO, exist_ok=True)
                os.makedirs(PASTA_DAT, exist_ok=True)
                
                if markdown and markdown.strip():
                    caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
                    with open(caminho_md, 'w', encoding='utf-8') as f:
                        f.write(markdown)
                
                if dat_base64 and dat_base64.strip():
                    caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")
                    with open(caminho_dat, 'w', encoding='utf-8') as f:
                        f.write(dat_base64)
                
                # Atualiza tabela (versão simplificada)
                suspeitos_str = ', '.join(suspeitos) if suspeitos else ''
                nova_linha = f"| {numero} | {tema} | {data_dist} | {responsavel} | {status} | {ultima_att} | {suspeitos_str} | {comentarios} |\n"
                
                # Salva na tabela (lógica simplificada)
                if not os.path.exists(PATH_TRIAGEM):
                    with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                        f.write("# Tabela de Processos\n\n")
                        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
                        f.write(nova_linha)
                else:
                    with open(PATH_TRIAGEM, 'a', encoding='utf-8') as f:
                        f.write(nova_linha)
                
                print(f"✅ Processo {numero} salvo (sem progresso)")
                
            except Exception as e:
                print(f"❌ Erro no processamento sem progresso: {e}")
        
        # Inicia processamento em background
        import threading
        thread = threading.Thread(target=processar_em_background)
        thread.daemon = True
        thread.start()
        
        # Retorna imediatamente para o frontend
        resultado_inicial = {
            "message": "Processamento iniciado",
            "operation_id": operation_id
        }
        
        return jsonify(resultado_inicial), 202  # 202 = Accepted
        
    except Exception as e:
        print(f"❌ Erro em POST /triagem/form: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>', methods=['PUT'])
def editar_processo(numero):
    print(f" Solicitação PUT /triagem/{numero} recebida")
    logger.info(f" Solicitação PUT /triagem/{numero} recebida")
    try:
        data = request.get_json()
        print(f" Dados recebidos: {data}")
        
        # Extrai processos existentes
        processos = extrair_tabela_md(PATH_TRIAGEM)
        
        # Encontra o processo existente para preservar suspeitos se necessário
        processo_existente = next((p for p in processos if p['numeroProcesso'] == numero), None)
        suspeitos_existentes = processo_existente.get('suspeitos', '') if processo_existente else ''
        
        # Remove o processo antigo da lista
        processos = [p for p in processos if p['numeroProcesso'] != numero]
        
        # DATA ATUAL AUTOMÁTICA - sempre seta data de hoje para última atualização
        from datetime import datetime
        ultima_att = datetime.now().strftime('%Y-%m-%d')
        print(f" Última atualização automática: {ultima_att}")
        
        # Determina como lidar com suspeitos
        markdown = data.get('markdown', '')
        suspeitos_calculados = ''
        
        if markdown and markdown.strip():
            # Se há markdown novo, recalcula suspeitos
            try:
                suspeitos_lista = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                suspeitos_calculados = ', '.join(suspeitos_lista) if suspeitos_lista else ''
                print(f" Suspeitos recalculados: {suspeitos_calculados}")
            except Exception as e:
                print(f" Erro ao calcular suspeitos: {e}")
                suspeitos_calculados = suspeitos_existentes
        else:
            # Se não há markdown, mantém suspeitos existentes
            suspeitos_calculados = suspeitos_existentes
            print(f" Mantendo suspeitos existentes: {suspeitos_calculados}")
        
        # Salva markdown atualizado se fornecido
        if markdown and markdown.strip():
            nome_arquivo_base = numero.replace('/', '-')
            os.makedirs(PASTA_DESTINO, exist_ok=True)
            caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
            
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f" Markdown atualizado: {caminho_md}")
        
        # Salva arquivo DAT se fornecido
        dat_base64 = data.get('dat')
        if dat_base64 and dat_base64.strip():
            nome_arquivo_base = numero.replace('/', '-')
            os.makedirs(PASTA_DAT, exist_ok=True)
            caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")
            
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            print(f"Arquivo DAT atualizado: {caminho_dat}")
        
        # Cria o processo atualizado (SEMPRE usa data atual para última atualização)
        processo_atualizado = {
            "numeroProcesso": limpar(data['numeroProcesso']),
            "tema": limpar(data['tema']),
            "dataDistribuicao": limpar(data['dataDistribuicao']),  # Mantém a data original
            "responsavel": limpar(data['responsavel']),
            "status": limpar(data['status']),
            "ultimaAtualizacao": ultima_att,  # SEMPRE data atual
            "suspeitos": suspeitos_calculados,
            "comentarios": limpar(data.get('comentarios', ''))
        }
        
        # Adiciona o processo atualizado à lista
        processos.append(processo_atualizado)
        
        # Reescreve o arquivo de triagem
        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
            for p in processos:
                f.write(
                    f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} "
                    f"| {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('comentarios', '')} |\n"
                )

        print(f" Processo {numero} atualizado com sucesso")
        logger.info(f" Processo {numero} atualizado com sucesso")
        return jsonify({"message": "Processo atualizado com sucesso"}), 200

    except KeyError as e:
        print(f" Campo obrigatório ausente em PUT /triagem/{numero}: {str(e)}")
        return jsonify({'error': f'Campo obrigatório ausente: {str(e)}'}), 400
    except Exception as e:
        print(f" Erro em PUT /triagem/{numero}: {str(e)}")
        logger.error(f" Erro em PUT /triagem/{numero}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>', methods=['DELETE'])
def deletar_processo(numero):
    logger.info(f"🗑️ Solicitação DELETE /triagem/{numero} recebida")
    try:
        processos = extrair_tabela_md(PATH_TRIAGEM)
        processos = [p for p in processos if p['numeroProcesso'] != numero]

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
            for p in processos:
                f.write(
                    f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} "
                    f"| {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('comentarios', '')} |\n"
                )

        caminho_md = os.path.join(PASTA_DESTINO, f"{numero.replace('/', '-')}.md")
        if os.path.exists(caminho_md):
            os.remove(caminho_md)
            logger.info(f" Arquivo markdown removido: {caminho_md}")

        logger.info(f" Processo {numero} excluído com sucesso")
        return jsonify({"message": "Processo excluído com sucesso"}), 200

    except Exception as e:
        logger.error(f" Erro em DELETE /triagem/{numero}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>/dat', methods=['GET'])
def obter_dat(numero):
    logger.info(f" Solicitação GET /triagem/{numero}/dat recebida")
    try:
        nome_arquivo = f"{numero.replace('/', '-')}.dat"
        caminho = os.path.join(PASTA_DAT, nome_arquivo)

        if not os.path.exists(caminho):
            logger.warning(f" Arquivo DAT não encontrado: {caminho}")
            return jsonify({'error': 'Arquivo .dat não encontrado'}), 404

        with open(caminho, 'r', encoding='utf-8') as f:
            dat_base64 = f.read()

        logger.info(f"Arquivo DAT retornado: {caminho}")
        return jsonify({'base64': dat_base64}), 200

    except Exception as e:
        logger.error(f" Erro em GET /triagem/{numero}/dat: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/debug/sockets')
def debug_sockets():
    """Debug dos sockets ativos"""
    return jsonify({
        'active_sockets': operation_sockets,
        'socket_count': len(operation_sockets),
        'keys': list(operation_sockets.keys())
    })

@app.route('/debug/test-progress/<operation_id>')
def test_progress(operation_id):
    """Testa envio de progresso manualmente"""
    from utils.progress_step import send_progress_ws
    
    send_progress_ws(operation_id, 2, 'Teste manual de progresso', 50)
    
    return jsonify({
        'message': f'Progresso teste enviado para {operation_id}',
        'socket_registered': operation_id in operation_sockets,
        'socket_id': operation_sockets.get(operation_id, 'Não encontrado')
    })

# ==========================================
# 🔄 FINALIZAÇÃO E EXECUÇÃO
# ==========================================

def signal_handler(sig, frame):
    logger.info(f" Sinal {sig} recebido. Finalizando servidor graciosamente...")
    logger.info(f" Servidor com PID {os.getpid()} finalizado")
    sys.exit(0)

# Registra os handlers de sinal
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@socketio.on('connect')
def handle_connect():
    print(f'✅ Cliente WebSocket conectado: {request.sid}')
    
@socketio.on('start_listening')
def handle_start_listening(data):
    print(f"📥 Evento start_listening recebido")
    print(f"📊 Dados: {data}")
    print(f"📍 Socket ID: {request.sid}")
    
    operation_id = data.get('operation_id')
    
    if operation_id:
        operation_sockets[operation_id] = request.sid
        print(f"✅ Registrado: {operation_id} -> {request.sid}")
        print(f"📊 Sockets ativos agora: {operation_sockets}")
        
        # Confirma registro
        socketio.emit('progress_update', {
            'step': 1,
            'message': 'Canal de progresso ativo! Aguardando processamento...',
            'percentage': 5,
            'operation_id': operation_id,
            'error': False
        }, to=request.sid)
        
        print(f"📤 Confirmação enviada para {request.sid}")
        
    else:
        print("❌ operation_id ausente!")
        socketio.emit('error', {'message': 'operation_id é obrigatório'}, to=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print(f'❌ Cliente WebSocket desconectado: {request.sid}')
    # Remove das operações ativas
    to_remove = [op_id for op_id, sid in operation_sockets.items() if sid == request.sid]
    for op_id in to_remove:
        operation_sockets.pop(op_id, None)
    print(f"🧹 Removidas {len(to_remove)} operações do cliente")

if __name__ == '__main__':
    try:
        print(f"\n SERVIDOR GMV SISTEMA PRONTO!")
        print("=" * 40)
        print(f" URL: http://127.0.0.1:5000")
        print(f" Health: http://127.0.0.1:5000/health")
        print(f" Info: http://127.0.0.1:5000/process-info")
        print(f" Dados: {os.path.abspath(os.path.dirname(PATH_TRIAGEM))}")
        print("=" * 40)
        print(f" USANDO AS SEGUINTES CONFIGURAÇÕES:")
        print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
        print(f"   PASTA_DESTINO: {PASTA_DESTINO}")
        print(f"   PASTA_DAT: {PASTA_DAT}")
        print("=" * 40)

        logger.info(f"Iniciando servidor Flask na porta 5000 com PID: {os.getpid()}")
        
        
        # CORRETO PARA SUPORTE A SOCKET.IO
        socketio.run(app, debug=True, port=5000, host='127.0.0.1')

    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {e}")
        print(f"\n ERRO CRÍTICO: {str(e)}")
        sys.exit(1)
