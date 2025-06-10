from flask import Flask, jsonify, request
import pandas as pd
import re
import os
import io
import signal
import sys
import logging
from datetime import datetime
from pathlib import Path

# ==========================================
# 🔧 SETUP AUTOMÁTICO DE VARIÁVEIS DE AMBIENTE
# ==========================================

def setup_environment():
    """Setup automático das variáveis de ambiente"""
    print("🔧 SETUP AUTOMÁTICO DO GMV SISTEMA")
    print("=" * 40)
    
    # Remove variáveis antigas do cache (se existirem)
    old_vars = ['PATH_TRIAGEM', 'PASTA_DESTINO', 'PASTA_DAT', 'GITHUB_TOKEN']
    for var in old_vars:
        if var in os.environ:
            print(f"🗑️ Removendo cache: {var}")
            del os.environ[var]
    
    # Verifica se python-dotenv está disponível
    try:
        from dotenv import load_dotenv
        dotenv_available = True
        print("✅ python-dotenv disponível")
    except ImportError:
        dotenv_available = False
        print("⚠️ python-dotenv não instalado - usando método manual")
    
    # Procura arquivo .env
    env_file = '.env'
    if not os.path.exists(env_file):
        print(f"📝 Arquivo .env não encontrado, criando...")
        create_default_env()
    
    # Carrega variáveis
    if dotenv_available:
        print(f"📄 Carregando .env: {os.path.abspath(env_file)}")
        load_dotenv(env_file, override=True)
    else:
        print(f"📄 Carregando .env manualmente...")
        load_env_manual(env_file)
    
    # Verifica se carregou
    PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
    PASTA_DESTINO = os.getenv("PASTA_DESTINO")
    PASTA_DAT = os.getenv("PASTA_DAT")
    
    print(f"\n📋 Variáveis carregadas:")
    print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
    print(f"   PASTA_DESTINO: {PASTA_DESTINO}")
    print(f"   PASTA_DAT: {PASTA_DAT}")
    
    if not PATH_TRIAGEM or not PASTA_DESTINO:
        print("\n❌ Variáveis essenciais não definidas!")
        create_default_env()
        if dotenv_available:
            load_dotenv(env_file, override=True)
        else:
            load_env_manual(env_file)
    
    # Cria estrutura de pastas
    setup_directories()
    
    print("✅ Setup concluído!\n")
    return os.getenv("PATH_TRIAGEM"), os.getenv("PASTA_DESTINO"), os.getenv("PASTA_DAT")

def create_default_env():
    """Cria arquivo .env padrão"""
    env_content = """# Configurações do GMV Sistema
PATH_TRIAGEM=./data/triagem.md
PASTA_DESTINO=./data/processos
PASTA_DAT=./data/dat
GITHUB_TOKEN=seu_token_aqui
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"✅ Arquivo .env criado: {os.path.abspath('.env')}")
    except Exception as e:
        print(f"❌ Erro ao criar .env: {e}")

def load_env_manual(env_file):
    """Carrega .env manualmente (fallback se dotenv não disponível)"""
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Variáveis carregadas manualmente")
    except Exception as e:
        print(f"❌ Erro ao carregar .env: {e}")

def setup_directories():
    """Cria estrutura de diretórios"""
    print("📁 Verificando/criando diretórios...")
    
    PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
    PASTA_DESTINO = os.getenv("PASTA_DESTINO") 
    PASTA_DAT = os.getenv("PASTA_DAT")
    
    # Cria diretórios
    for pasta, nome in [(PASTA_DESTINO, "PASTA_DESTINO"), (PASTA_DAT, "PASTA_DAT")]:
        if pasta:
            try:
                os.makedirs(pasta, exist_ok=True)
                print(f"   ✅ {nome}: {pasta}")
            except Exception as e:
                print(f"   ❌ Erro ao criar {nome}: {e}")
    
    # Cria arquivo de triagem se não existir
    if PATH_TRIAGEM:
        try:
            # Cria diretório pai se necessário
            triagem_dir = os.path.dirname(PATH_TRIAGEM)
            if triagem_dir:
                os.makedirs(triagem_dir, exist_ok=True)
            
            if not os.path.exists(PATH_TRIAGEM):
                triagem_content = """# Tabela de Processos

| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |
|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|

"""
                with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                    f.write(triagem_content)
                print(f"   ✅ Arquivo de triagem criado: {PATH_TRIAGEM}")
            else:
                print(f"   ✅ PATH_TRIAGEM: {PATH_TRIAGEM}")
        except Exception as e:
            print(f"   ❌ Erro ao criar arquivo de triagem: {e}")

# ==========================================
# 🚀 INICIALIZAÇÃO DO SETUP
# ==========================================

# Executa setup automático
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()

# Verificação final
if not PATH_TRIAGEM or not PASTA_DESTINO:
    print("\n❌ ERRO CRÍTICO: Não foi possível configurar variáveis de ambiente!")
    print("Verifique se:")
    print("1. Você tem permissão para criar arquivos neste diretório")
    print("2. O arquivo .env foi criado corretamente")
    print("3. As variáveis estão definidas no .env")
    sys.exit(1)

# ==========================================
# 🌐 CONFIGURAÇÃO DO FLASK
# ==========================================

# Importa utils só depois do setup
try:
    from utils.suspeicao import encontrar_suspeitos
    print("✅ Módulo de suspeição carregado")
except ImportError as e:
    print(f"⚠️ Aviso: Módulo de suspeição não encontrado: {e}")
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

app = Flask(__name__)

# Configuração CORS
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})
    print("✅ CORS configurado")
except ImportError:
    print("⚠️ flask-cors não instalado - CORS pode não funcionar")

# Log de inicialização
logger.info(f"🚀 Servidor Flask iniciando com PID: {os.getpid()}")
logger.info(f"📁 PATH_TRIAGEM: {PATH_TRIAGEM}")
logger.info(f"📁 PASTA_DESTINO: {PASTA_DESTINO}")
logger.info(f"📁 PASTA_DAT: {PASTA_DAT}")

# ==========================================
# 🛠️ FUNÇÕES AUXILIARES
# ==========================================

def limpar(valor):
    return str(valor).strip() if valor is not None else ""

def extrair_tabela_md(arquivo_md):
    try:
        with open(arquivo_md, 'r', encoding='utf-8') as f:
            linhas = f.readlines()

        inicio = next((i for i, l in enumerate(linhas) if re.match(r'^\|.+\|$', l)), None)
        if inicio is None:
            logger.warning(f"⚠️ Nenhuma tabela encontrada em {arquivo_md}")
            return []

        tabela_linhas = []
        for linha in linhas[inicio:]:
            if not linha.strip().startswith('|'):
                break
            if re.match(r'^\|\s*-+\s*\|', linha):
                continue
            tabela_linhas.append(linha.strip())

        if not tabela_linhas:
            logger.warning(f"⚠️ Tabela vazia em {arquivo_md}")
            return []

        tabela_str = '\n'.join(tabela_linhas)
        
        try:
            df = pd.read_csv(io.StringIO(tabela_str), sep='|', engine='python', skipinitialspace=True)
            df = df.dropna(axis=1, how='all')
            df.columns = [col.strip() for col in df.columns]
        except Exception as e:
            logger.error(f"❌ Erro ao processar CSV: {e}")
            return []

        processos = []
        for _, row in df.iterrows():
            processos.append({
                "numeroProcesso": limpar(row.get("Nº Processo")),
                "tema": limpar(row.get("Tema")),
                "dataDistribuicao": limpar(row.get("Data da Distribuição")),
                "responsavel": limpar(row.get("Responsável")),
                "status": limpar(row.get("Status")),
                "ultimaAtualizacao": limpar(row.get("Última Atualização")),
                "suspeitos": limpar(row.get('Suspeitos')),
                "comentarios": limpar(row.get("Comentários")) if "Comentários" in row else ""
            })
        
        logger.info(f"📋 {len(processos)} processos extraídos de {arquivo_md}")
        return processos
        
    except Exception as e:
        logger.error(f"❌ Erro ao extrair tabela de {arquivo_md}: {str(e)}")
        return []

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
    logger.info("📖 Solicitação GET /triagem recebida")
    try:
        dados = extrair_tabela_md(PATH_TRIAGEM)
        logger.info(f"✅ Retornando {len(dados)} processos")
        return jsonify(dados)
    except Exception as e:
        logger.error(f"❌ Erro em GET /triagem: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    logger.info("📝 Solicitação POST /triagem/form recebida")
    try:
        data = request.get_json()
        numero = limpar(data.get('numeroProcesso'))
        tema = limpar(data.get('tema'))
        data_dist = limpar(data.get('dataDistribuicao'))
        responsavel = limpar(data.get('responsavel'))
        status = limpar(data.get('status'))
        ultima_att = limpar(data.get('ultimaAtualizacao'))
        markdown = limpar(data.get('markdown'))
        comentarios = limpar(data.get('comentarios'))
        dat_base64 = data.get('dat')
        
        logger.info(f"📄 Processando processo: {numero}")
        
        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
        logger.info(f"🔍 Suspeitos encontrados: {suspeitos}")

        if not markdown or not numero:
            logger.warning("⚠️ Campos obrigatórios ausentes")
            return jsonify({'error': 'Campos obrigatórios ausentes'}), 400

        nome_arquivo_base = numero.replace('/', '-')
        os.makedirs(PASTA_DESTINO, exist_ok=True)
        os.makedirs(PASTA_DAT, exist_ok=True)

        caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
        caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")

        # Salva markdown
        with open(caminho_md, 'w', encoding='utf-8') as f:
            f.write(markdown)
        logger.info(f"💾 Markdown salvo: {caminho_md}")

        # Salva .dat como base64 se enviado
        if dat_base64:
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            logger.info(f"💾 Arquivo DAT salvo: {caminho_dat}")

        nova_linha = (
            f"| {numero} "
            f"| {tema} "
            f"| {data_dist} "
            f"| {responsavel} "
            f"| {status} "
            f"| {ultima_att} "
            f"| {suspeitos} "
            f"| {comentarios} |\n"
        )

        if not os.path.exists(PATH_TRIAGEM):
            logger.info(f"📝 Criando novo arquivo de triagem: {PATH_TRIAGEM}")
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
        
        logger.info(f"✅ Processo {numero} salvo com sucesso")
        return jsonify({"message": "Processo e arquivos salvos com sucesso"}), 201

    except Exception as e:
        logger.error(f"❌ Erro em POST /triagem/form: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>', methods=['PUT'])
def editar_processo(numero):
    logger.info(f"✏️ Solicitação PUT /triagem/{numero} recebida")
    try:
        data = request.get_json()
        processos = extrair_tabela_md(PATH_TRIAGEM)
        processos = [p for p in processos if p['numeroProcesso'] != numero]

        processos.append({
            "numeroProcesso": limpar(data['numeroProcesso']),
            "tema": limpar(data['tema']),
            "dataDistribuicao": limpar(data['dataDistribuicao']),
            "responsavel": limpar(data['responsavel']),
            "status": limpar(data['status']),
            "ultimaAtualizacao": limpar(data['ultimaAtualizacao']),
            "suspeitos": limpar(data['suspeitos']),
            "comentarios": limpar(data.get('comentarios', ''))
        })

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
            for p in processos:
                f.write(
                    f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} "
                    f"| {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('comentarios', '')} |\n"
                )

        logger.info(f"✅ Processo {numero} atualizado com sucesso")
        return jsonify({"message": "Processo atualizado com sucesso"}), 200

    except Exception as e:
        logger.error(f"❌ Erro em PUT /triagem/{numero}: {str(e)}")
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
            logger.info(f"🗑️ Arquivo markdown removido: {caminho_md}")

        logger.info(f"✅ Processo {numero} excluído com sucesso")
        return jsonify({"message": "Processo excluído com sucesso"}), 200

    except Exception as e:
        logger.error(f"❌ Erro em DELETE /triagem/{numero}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>/dat', methods=['GET'])
def obter_dat(numero):
    logger.info(f"📁 Solicitação GET /triagem/{numero}/dat recebida")
    try:
        nome_arquivo = f"{numero.replace('/', '-')}.dat"
        caminho = os.path.join(PASTA_DAT, nome_arquivo)

        if not os.path.exists(caminho):
            logger.warning(f"⚠️ Arquivo DAT não encontrado: {caminho}")
            return jsonify({'error': 'Arquivo .dat não encontrado'}), 404

        with open(caminho, 'r', encoding='utf-8') as f:
            dat_base64 = f.read()

        logger.info(f"✅ Arquivo DAT retornado: {caminho}")
        return jsonify({'base64': dat_base64}), 200

    except Exception as e:
        logger.error(f"❌ Erro em GET /triagem/{numero}/dat: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==========================================
# 🔄 FINALIZAÇÃO E EXECUÇÃO
# ==========================================

def signal_handler(sig, frame):
    logger.info(f"🛑 Sinal {sig} recebido. Finalizando servidor graciosamente...")
    logger.info(f"🏁 Servidor com PID {os.getpid()} finalizado")
    sys.exit(0)

# Registra os handlers de sinal
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    try:
        print(f"\n🌟 SERVIDOR GMV SISTEMA PRONTO!")
        print("=" * 40)
        print(f"🔗 URL: http://127.0.0.1:5000")
        print(f"🩺 Health: http://127.0.0.1:5000/health")
        print(f"📊 Info: http://127.0.0.1:5000/process-info")
        print(f"📁 Dados: {os.path.abspath('./data')}")
        print("=" * 40)
        
        logger.info(f"🌟 Iniciando servidor Flask na porta 5000 com PID: {os.getpid()}")
        app.run(debug=True, port=5000, host='127.0.0.1')
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar servidor: {str(e)}")
        print(f"\n❌ ERRO CRÍTICO: {str(e)}")
        print("\n🔧 POSSÍVEIS SOLUÇÕES:")
        print("1. Verifique se a porta 5000 está livre")
        print("2. Execute como administrador")
        print("3. Verifique permissões de arquivo")
        sys.exit(1)
    finally:
        logger.info(f"🏁 Servidor Flask com PID {os.getpid()} finalizado")