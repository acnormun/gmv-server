from flask import Blueprint, request, jsonify
from utils.triagem import (
    get_processos,
    atualizar_processo,
    deletar_processo_por_numero,
    obter_dat_por_numero
)
from utils.auto_setup import setup_environment
from utils.auxiliar import limpar, get_anonimizador
from utils.suspeicao import encontrar_suspeitos
from utils.progress_step import send_progress_ws
import uuid
import threading
import time
import os
import logging
from datetime import datetime
from utils.extrair_metadados_processo import extrair_e_formatar_metadados
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import copy

try:
    from utils.email_notification import email_service
except ImportError:
    pass
except Exception:
    pass

logger = logging.getLogger(__name__)

triagem_bp = Blueprint('triagem', __name__)
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()

MAX_CONCURRENT_OPERATIONS = 15
ENABLE_WEBSOCKET_WAIT = True

class OperationTracker:
    def __init__(self):
        self._lock = RLock()
        self._operations = {}
        self._sockets = {}
    
    def register(self, operation_id, details=None):
        with self._lock:
            self._operations[operation_id] = {
                'start_time': datetime.now(),
                'details': details or {},
                'status': 'queued',
                'progress': 0
            }
    
    def set_socket(self, operation_id, socket_id):
        with self._lock:
            self._sockets[operation_id] = socket_id
    
    def get_socket(self, operation_id):
        with self._lock:
            return self._sockets.get(operation_id)
    
    def update_status(self, operation_id, status, progress=None):
        with self._lock:
            if operation_id in self._operations:
                self._operations[operation_id]['status'] = status
                if progress is not None:
                    self._operations[operation_id]['progress'] = progress
    
    def remove(self, operation_id):
        with self._lock:
            self._operations.pop(operation_id, None)
            self._sockets.pop(operation_id, None)
    
    def get_all_operations(self):
        with self._lock:
            return copy.deepcopy(self._operations)
    
    def cleanup_old(self, max_age_seconds=1800):
        with self._lock:
            current_time = datetime.now()
            expired = []
            for op_id, op_data in self._operations.items():
                age = (current_time - op_data['start_time']).total_seconds()
                if age > max_age_seconds:
                    expired.append(op_id)
            for op_id in expired:
                self.remove(op_id)
                logger.info(f"Operação expirada removida: {op_id}")

operation_tracker = OperationTracker()

_thread_pool = None
_pool_lock = threading.Lock()

def get_thread_pool():
    global _thread_pool
    if _thread_pool is None:
        with _pool_lock:
            if _thread_pool is None:
                _thread_pool = ThreadPoolExecutor(
                    max_workers=MAX_CONCURRENT_OPERATIONS,
                    thread_name_prefix="TriagemWorker"
                )
                logger.info(f"Pool de threads criado com {MAX_CONCURRENT_OPERATIONS} workers")
    return _thread_pool

ANONIMIZACAO_ATIVA = True

try:
    from adaptive_rag import rag_system
    RAG_DISPONIVEL = True
except ImportError:
    RAG_DISPONIVEL = False

file_write_lock = RLock()
anonimization_lock = RLock()

def safe_send_progress(operation_id, step, message, percentage):
    try:
        operation_tracker.update_status(operation_id, 'running', percentage)
        send_progress_ws(operation_id, step, message, percentage)
        logger.info(f"[{operation_id[:8]}] {percentage}% - {message}")
        return True
    except Exception as e:
        logger.error(f"Erro ao enviar progresso [{operation_id[:8]}]: {e}")
        return False

def processar_processo_isolado(data, operation_id):
    try:
        logger.info(f"Iniciando processamento isolado: {operation_id[:8]}")
        operation_tracker.update_status(operation_id, 'running')
        if ENABLE_WEBSOCKET_WAIT:
            safe_send_progress(operation_id, 1, 'Conectando...', 5)
            for _ in range(6):
                if operation_tracker.get_socket(operation_id):
                    logger.info(f"WebSocket conectado para {operation_id[:8]}")
                    break
                time.sleep(0.5)
            if not operation_tracker.get_socket(operation_id):
                logger.info(f"WebSocket não conectado para {operation_id[:8]}, continuando...")
        safe_send_progress(operation_id, 2, 'Validando dados...', 10)
        numero = limpar(data.get('numeroProcesso'))
        tema = limpar(data.get('tema', ''))
        data_dist = limpar(data.get('dataDistribuicao', ''))
        responsavel = limpar(data.get('responsavel', ''))
        status = limpar(data.get('status', ''))
        markdown = limpar(data.get('markdown', ''))
        comentarios = limpar(data.get('comentarios', ''))
        dat_base64 = data.get('dat')
        ultima_att = datetime.now().strftime('%Y-%m-%d')
        if not numero:
            safe_send_progress(operation_id, 0, 'Erro: Número do processo obrigatório', 0)
            return
        if not markdown and not dat_base64:
            safe_send_progress(operation_id, 0, 'Erro: Markdown ou PDF obrigatório', 0)
            return
        logger.info(f"Processando: {numero} [{operation_id[:8]}]")
        resultado_pje = None
        if dat_base64 and dat_base64.strip():
            safe_send_progress(operation_id, 3, 'Processando PDF do PJe...', 20)
            try:
                from utils.processador_pje_integrado import processar_pje_com_progresso
                nome_arquivo_base = numero.replace('/', '-')
                pasta_processo = os.path.join(PASTA_DESTINO, nome_arquivo_base)
                os.makedirs(pasta_processo, exist_ok=True)
                resultado_pje = processar_pje_com_progresso(
                    dat_base64,
                    numero,
                    pasta_processo,
                    operation_id
                )

                if resultado_pje and resultado_pje['sucesso']:
                    if resultado_pje['arquivos_gerados']['markdowns']:
                        primeiro_md = resultado_pje['arquivos_gerados']['markdowns'][0]
                        try:
                            with open(primeiro_md, 'r', encoding='utf-8') as f:
                                markdown_pje = f.read()
                            if not markdown or not markdown.strip():
                                markdown = markdown_pje
                        except Exception as e:
                            logger.warning(f"Erro ao ler markdown PJe: {e}")
            except Exception as e:
                logger.warning(f"ProcessadorPJe falhou [{operation_id[:8]}]: {e}")
                safe_send_progress(operation_id, 3, 'Aviso: ProcessadorPJe falhou, continuando...', 25)
        safe_send_progress(operation_id, 4, 'Analisando suspeição...', 35)
        suspeitos = []
        if markdown and markdown.strip():
            try:
                suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                logger.info(f"Suspeitos encontrados [{numero}]: {suspeitos}")
            except Exception as e:
                logger.error(f"Erro na análise de suspeitos [{numero}]: {e}")
                suspeitos = []
        safe_send_progress(operation_id, 5, 'Preparando arquivos...', 45)
        nome_arquivo_base = numero.replace('/', '-')
        os.makedirs(PASTA_DESTINO, exist_ok=True)
        os.makedirs(PASTA_DAT, exist_ok=True)
        caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
        safe_send_progress(operation_id, 6, 'Salvando markdown...', 55)
        if markdown and markdown.strip():
            try:
                metadados_dict, front_matter = extrair_e_formatar_metadados(markdown)
                campos_extraidos = len([v for v in metadados_dict.values() if v])
                if campos_extraidos > 0:
                    markdown_com_metadados = front_matter + "\n\n" + markdown
                else:
                    markdown_com_metadados = markdown
                with open(caminho_md, 'w', encoding='utf-8') as f:
                    f.write(markdown_com_metadados)
                logger.info(f"Markdown salvo: {caminho_md}")
            except Exception as e:
                logger.error(f"Erro ao salvar markdown [{numero}]: {e}")
        if RAG_DISPONIVEL:
            try:
                safe_send_progress(operation_id, 7, 'Atualizando busca...', 65)
            except Exception as e:
                logger.warning(f"Erro RAG [{numero}]: {e}")
        safe_send_progress(operation_id, 8, 'Executando anonimização...', 75)
        total_substituicoes = 0
        if ANONIMIZACAO_ATIVA and markdown and markdown.strip():
            try:
                anonimizador = get_anonimizador()
                if anonimizador:
                    mapa_suspeitos = anonimizador.carregar_suspeitos_mapeados("./utils/suspeitos.txt")
                    texto_anonimizado, mapa_reverso = anonimizador.anonimizar_com_identificadores(
                        markdown, mapa_suspeitos, debug=False
                    )
                    pasta_anon = os.path.join(PASTA_DESTINO, "anonimizados")
                    os.makedirs(pasta_anon, exist_ok=True)
                    caminho_md_anon = os.path.join(pasta_anon, f"{nome_arquivo_base}_anon.md")
                    with open(caminho_md_anon, "w", encoding="utf-8") as f:
                        f.write(texto_anonimizado)
                    total_substituicoes = len(mapa_reverso)
                    logger.info(f"Anonimização [{numero}]: {total_substituicoes} substituições")
            except Exception as e:
                logger.error(f"Erro na anonimização [{numero}]: {e}")
        safe_send_progress(operation_id, 9, 'Atualizando tabela...', 90)
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
        try:
            with file_write_lock:
                # ✅ SEM FCNTL - funciona no Windows e Linux
                with open(PATH_TRIAGEM, 'a+', encoding='utf-8') as f:
                    f.seek(0)
                    linhas = f.readlines()
                    if not linhas:
                        f.write("# Tabela de Processos\n\n")
                        f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                        f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
                    f.seek(0, 2)  # Vai para o final do arquivo
                    f.write(nova_linha)
            logger.info(f"Tabela atualizada para {numero}")
        except Exception as e:
            logger.error(f"Erro ao atualizar tabela [{numero}]: {e}")

        try:
            from utils.email_notification import enviar_notificacao_processo
            dados_notificacao = {
                'numero': numero,
                'tema': tema,
                'responsavel': responsavel,
                'suspeitos': suspeitos,
                'processamento_pje': resultado_pje is not None,
                'total_substituicoes': total_substituicoes
            }
            email_thread = threading.Thread(
                target=enviar_notificacao_processo,
                args=(dados_notificacao,),
                daemon=True
            )
            email_thread.start()
        except Exception as e:
            logger.warning(f"Erro no envio de email [{numero}]: {e}")
        safe_send_progress(operation_id, 10, 'Processo concluído!', 100)
        logger.info(f"Processo concluído: {numero} [{operation_id[:8]}]")
        logger.info(f"Suspeitos: {len(suspeitos)}")
        logger.info(f"Substituições: {total_substituicoes}")
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Erro no processamento [{operation_id[:8]}]: {e}")
        import traceback
        traceback.print_exc()
        safe_send_progress(operation_id, 0, f'Erro: {str(e)}', 0)
    finally:
        operation_tracker.remove(operation_id)

def cleanup_expired_operations():
    operation_tracker.cleanup_old()

@triagem_bp.route('/triagem', methods=['GET'])
def listar_processos():
    processos = get_processos()
    return jsonify(processos), 200

@triagem_bp.route('/triagem/status', methods=['GET'])
def status_operacoes():
    cleanup_expired_operations()
    operations = operation_tracker.get_all_operations()
    active_ops = {
        'total_active': len(operations),
        'max_concurrent': MAX_CONCURRENT_OPERATIONS,
        'operations': {
            op_id: {
                'start_time': op_data['start_time'].isoformat(),
                'duration_seconds': (datetime.now() - op_data['start_time']).total_seconds(),
                'status': op_data['status'],
                'progress': op_data.get('progress', 0)
            }
            for op_id, op_data in operations.items()
        }
    }
    return jsonify(active_ops), 200

@triagem_bp.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    try:
        cleanup_expired_operations()
        current_operations = len(operation_tracker.get_all_operations())
        if current_operations >= MAX_CONCURRENT_OPERATIONS:
            return jsonify({
                'success': False,
                'error': 'Limite de operações simultâneas atingido',
                'max_concurrent': MAX_CONCURRENT_OPERATIONS,
                'active_operations': current_operations
            }), 429
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Dados não fornecidos'}), 400
        operation_id = str(uuid.uuid4())
        operation_tracker.register(operation_id, {
            'numero_processo': data.get('numeroProcesso', 'N/A'),
            'responsavel': data.get('responsavel', 'N/A'),
            'endpoint': 'POST /triagem/form'
        })
        thread_pool = get_thread_pool()
        thread_pool.submit(processar_processo_isolado, data, operation_id)
        logger.info(f"Operação submetida: {operation_id[:8]}")
        logger.info(f"Operações ativas: {current_operations + 1}/{MAX_CONCURRENT_OPERATIONS}")
        return jsonify({
            "success": True,
            "message": "Processamento iniciado",
            "operation_id": operation_id,
            "queue_position": current_operations + 1
        }), 200
    except Exception as e:
        logger.error(f"Erro em POST /triagem/form: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def register_websocket(operation_id, socket_id):
    operation_tracker.set_socket(operation_id, socket_id)
    logger.info(f"WebSocket registrado: {operation_id[:8]} -> {socket_id}")

triagem_bp.register_websocket = register_websocket

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