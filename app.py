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
    """Setup automático das variáveis de ambiente com LIMPEZA FORÇADA DE CACHE"""
    print("🔧 SETUP AUTOMÁTICO DO GMV SISTEMA")
    print("=" * 40)
    
    # LIMPEZA FORÇADA - Remove TODAS as variáveis relacionadas ao GMV
    old_vars = ['PATH_TRIAGEM', 'PASTA_DESTINO', 'PASTA_DAT', 'GITHUB_TOKEN']
    removed_count = 0
    
    print("🗑️ LIMPEZA FORÇADA DE CACHE:")
    for var in old_vars:
        if var in os.environ:
            old_value = os.environ[var]
            del os.environ[var]
            print(f"   ❌ Removido: {var} = {old_value}")
            removed_count += 1
    
    if removed_count == 0:
        print("   ✅ Nenhuma variável em cache (primeira execução)")
    else:
        print(f"   🗑️ {removed_count} variáveis antigas removidas do cache")
    
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
    env_path = os.path.abspath(env_file)
    
    print(f"\n📄 VERIFICAÇÃO DO ARQUIVO .ENV:")
    print(f"   Local: {env_path}")
    print(f"   Existe: {os.path.exists(env_file)}")
    
    if os.path.exists(env_file):
        # Mostra conteúdo do .env para debug
        print("   Conteúdo atual:")
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if 'TOKEN' in line.upper():
                            key = line.split('=')[0] if '=' in line else line
                            print(f"      {i:2d}: {key}=***HIDDEN***")
                        else:
                            print(f"      {i:2d}: {line}")
        except Exception as e:
            print(f"   ❌ Erro ao ler .env: {e}")
    else:
        print(f"   📝 Arquivo .env não encontrado, criando...")
        create_default_env()
    
    print(f"\n🔄 CARREGAMENTO FORÇADO:")
    # Carrega variáveis com OVERRIDE forçado
    if dotenv_available:
        print(f"   📥 Usando python-dotenv com override=True")
        from dotenv import load_dotenv
        result = load_dotenv(env_file, override=True, verbose=True)
        print(f"   📊 Resultado: {result}")
    else:
        print(f"   📥 Carregando manualmente...")
        load_env_manual(env_file)
    
    # Verifica se carregou corretamente
    PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
    PASTA_DESTINO = os.getenv("PASTA_DESTINO")
    PASTA_DAT = os.getenv("PASTA_DAT")
    
    print(f"\n📋 VARIÁVEIS CARREGADAS (ATUAL):")
    print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
    print(f"   PASTA_DESTINO: {PASTA_DESTINO}")
    print(f"   PASTA_DAT: {PASTA_DAT}")
    
    # Se ainda não carregou, tenta novamente
    if not PATH_TRIAGEM or not PASTA_DESTINO:
        print("\n⚠️ Variáveis essenciais não definidas! Tentando corrigir...")
        
        # Recria .env
        create_default_env()
        
        # Tenta carregar novamente
        if dotenv_available:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
        else:
            load_env_manual(env_file)
        
        # Verifica novamente
        PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
        PASTA_DESTINO = os.getenv("PASTA_DESTINO")
        PASTA_DAT = os.getenv("PASTA_DAT")
        
        print(f"   📋 APÓS CORREÇÃO:")
        print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
        print(f"   PASTA_DESTINO: {PASTA_DESTINO}")
        print(f"   PASTA_DAT: {PASTA_DAT}")
    
    # Cria estrutura de pastas
    setup_directories()
    
    print("✅ Setup concluído!\n")
    return PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT

def create_default_env():
    """Cria arquivo .env padrão"""
    env_content = """# Configurações do GMV Sistema
# Edite os caminhos conforme necessário para este PC

# Arquivo principal de triagem (será criado se não existir)
PATH_TRIAGEM=./data/triagem.md

# Pasta onde ficam os arquivos markdown dos processos
PASTA_DESTINO=./data/processos

# Pasta onde ficam os arquivos .dat
PASTA_DAT=./data/dat

# Token do GitHub (opcional, para atualizações)
GITHUB_TOKEN=seu_token_aqui

# ===========================================
# EXEMPLOS para diferentes PCs:
# ===========================================
# 
# Para usar caminhos absolutos Windows:
# PATH_TRIAGEM=C:/GMV_Data/triagem.md
# PASTA_DESTINO=C:/GMV_Data/processos
# PASTA_DAT=C:/GMV_Data/dat
#
# Para usar pasta do usuário:
# PATH_TRIAGEM=%USERPROFILE%/Documents/GMV/triagem.md
# PASTA_DESTINO=%USERPROFILE%/Documents/GMV/processos
# PASTA_DAT=%USERPROFILE%/Documents/GMV/dat
#
# Para usar pasta específica:
# PATH_TRIAGEM=D:/Trabalho/GMV/triagem.md
# PASTA_DESTINO=D:/Trabalho/GMV/processos  
# PASTA_DAT=D:/Trabalho/GMV/dat
"""
    
    try:
        with open('.env', 'w', encoding='utf-8') as f:
            f.write(env_content)
        print(f"   ✅ Arquivo .env criado: {os.path.abspath('.env')}")
        print(f"   📝 Edite o arquivo se quiser usar outros caminhos")
        return True
    except Exception as e:
        print(f"   ❌ Erro ao criar .env: {e}")
        return False

def load_env_manual(env_file):
    """Carrega .env manualmente (fallback se dotenv não disponível)"""
    loaded_vars = {}
    try:
        with open(env_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                original_line = line
                line = line.strip()
                
                # Pula linhas vazias e comentários
                if not line or line.startswith('#'):
                    continue
                
                # Processa linha com =
                if '=' in line:
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove aspas se existirem
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        # Define no ambiente
                        os.environ[key] = value
                        loaded_vars[key] = value
                        
                        # Log (esconde tokens)
                        if 'TOKEN' in key.upper() or 'PASSWORD' in key.upper():
                            print(f"      ✅ {key} = ***HIDDEN***")
                        else:
                            print(f"      ✅ {key} = {value}")
                            
                    except Exception as e:
                        print(f"      ⚠️ Erro na linha {line_num}: {original_line.strip()} - {e}")
                else:
                    print(f"      ⚠️ Linha {line_num} inválida (sem =): {original_line.strip()}")
        
        print(f"   📊 Total: {len(loaded_vars)} variáveis carregadas manualmente")
        return True
        
    except Exception as e:
        print(f"   ❌ Erro ao carregar .env manualmente: {e}")
        return False

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

print("🚀 GMV SISTEMA - INICIALIZAÇÃO COM LIMPEZA DE CACHE")
print("=" * 60)
print(f"📁 Diretório de trabalho: {os.getcwd()}")
print(f"📄 Procurando .env em: {os.path.abspath('.env')}")

# Executa setup automático COM LIMPEZA FORÇADA
PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()

# Verificação final
if not PATH_TRIAGEM or not PASTA_DESTINO:
    print("\n❌ ERRO CRÍTICO: Não foi possível configurar variáveis de ambiente!")
    print("🔧 DEPURAÇÃO:")
    print("   1. Verifique se você tem permissão para criar arquivos neste diretório")
    print("   2. Verifique se o arquivo .env foi criado corretamente")
    print("   3. Tente executar como administrador")
    print(f"   4. Arquivo .env deveria estar em: {os.path.abspath('.env')}")
    sys.exit(1)

# TESTE FINAL - Confirma que variáveis corretas estão sendo usadas
print("🧪 TESTE FINAL DE VERIFICAÇÃO:")
print("=" * 40)
final_vars = {
    'PATH_TRIAGEM': os.getenv('PATH_TRIAGEM'),
    'PASTA_DESTINO': os.getenv('PASTA_DESTINO'), 
    'PASTA_DAT': os.getenv('PASTA_DAT')
}

for var_name, var_value in final_vars.items():
    print(f"✅ {var_name} = {var_value}")
    
    # Verifica se o caminho é absoluto ou relativo
    if var_value:
        abs_path = os.path.abspath(var_value)
        print(f"   📍 Caminho absoluto: {abs_path}")

print("\n🎯 CONFIRMAÇÃO:")
print(f"✅ Cache de variáveis antigas foi limpo")
print(f"✅ Arquivo .env atual foi carregado")
print(f"✅ {len([v for v in final_vars.values() if v])} variáveis essenciais definidas")
print("=" * 60)

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

# Correção para POST /triagem/form
@app.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    print("📝 Solicitação POST /triagem/form recebida")
    try:
        data = request.get_json()
        print(f"📄 Dados recebidos: {data}")
        
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
        print(f"📅 Data de distribuição informada: {data_dist}")
        print(f"📅 Última atualização automática: {ultima_att}")
        
        print(f"📄 Processando processo: {numero}")
        
        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
        print(f"🔍 Suspeitos encontrados: {suspeitos}")

        if not numero:
            print("⚠️ Número do processo obrigatório")
            return jsonify({'error': 'Número do processo é obrigatório'}), 400
        
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

        # Salva markdown se fornecido
        if markdown and markdown.strip():
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"💾 Markdown salvo: {caminho_md}")
        # Salva markdown
        with open(caminho_md, 'w', encoding='utf-8') as f:
            f.write(markdown)
        logger.info(f"💾 Markdown salvo: {caminho_md}")

        # Salva .dat como base64 se enviado
        if dat_base64 and dat_base64.strip():
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            print(f"💾 Arquivo DAT salvo: {caminho_dat}")

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
            print(f"📝 Criando novo arquivo de triagem: {PATH_TRIAGEM}")
            logger.info(f"📝 Criando novo arquivo de triagem: {PATH_TRIAGEM}")
            with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                f.write("# Tabela de Processos\n\n")
                f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
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
                "|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n",
                nova_linha
            ]

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.writelines(linhas)
        
        print(f"✅ Processo {numero} salvo com sucesso")
        
        logger.info(f"✅ Processo {numero} salvo com sucesso")
        return jsonify({"message": "Processo e arquivos salvos com sucesso"}), 201

    except Exception as e:
        print(f"❌ Erro em POST /triagem/form: {str(e)}")
        logger.error(f"❌ Erro em POST /triagem/form: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Correção para PUT /triagem/<numero>
@app.route('/triagem/<numero>', methods=['PUT'])
def editar_processo(numero):
    print(f"✏️ Solicitação PUT /triagem/{numero} recebida")
    logger.info(f"✏️ Solicitação PUT /triagem/{numero} recebida")
    try:
        data = request.get_json()
        print(f"📝 Dados recebidos: {data}")
        
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
        print(f"📅 Última atualização automática: {ultima_att}")
        
        # Determina como lidar com suspeitos
        markdown = data.get('markdown', '')
        suspeitos_calculados = ''
        
        if markdown and markdown.strip():
            # Se há markdown novo, recalcula suspeitos
            try:
                suspeitos_lista = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                suspeitos_calculados = ', '.join(suspeitos_lista) if suspeitos_lista else ''
                print(f"🔍 Suspeitos recalculados: {suspeitos_calculados}")
            except Exception as e:
                print(f"⚠️ Erro ao calcular suspeitos: {e}")
                suspeitos_calculados = suspeitos_existentes
        else:
            # Se não há markdown, mantém suspeitos existentes
            suspeitos_calculados = suspeitos_existentes
            print(f"🔄 Mantendo suspeitos existentes: {suspeitos_calculados}")
        
        # Salva markdown atualizado se fornecido
        if markdown and markdown.strip():
            nome_arquivo_base = numero.replace('/', '-')
            os.makedirs(PASTA_DESTINO, exist_ok=True)
            caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
            
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"💾 Markdown atualizado: {caminho_md}")
        
        # Salva arquivo DAT se fornecido
        dat_base64 = data.get('dat')
        if dat_base64 and dat_base64.strip():
            nome_arquivo_base = numero.replace('/', '-')
            os.makedirs(PASTA_DAT, exist_ok=True)
            caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")
            
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            print(f"💾 Arquivo DAT atualizado: {caminho_dat}")
        
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

        print(f"✅ Processo {numero} atualizado com sucesso")
        logger.info(f"✅ Processo {numero} atualizado com sucesso")
        return jsonify({"message": "Processo atualizado com sucesso"}), 200

    except KeyError as e:
        print(f"❌ Campo obrigatório ausente em PUT /triagem/{numero}: {str(e)}")
        return jsonify({'error': f'Campo obrigatório ausente: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Erro em PUT /triagem/{numero}: {str(e)}")
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
        print(f"📁 Dados: {os.path.abspath(os.path.dirname(PATH_TRIAGEM))}")
        print("=" * 40)
        print(f"📋 USANDO AS SEGUINTES CONFIGURAÇÕES:")
        print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
        print(f"   PASTA_DESTINO: {PASTA_DESTINO}")
        print(f"   PASTA_DAT: {PASTA_DAT}")
        print("=" * 40)
        
        logger.info(f"Iniciando servidor Flask na porta 5000 com PID: {os.getpid()}")
        app.run(debug=True, port=5000, host='127.0.0.1')
        
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {e}")
        print(f"\n❌ ERRO CRÍTICO: {str(e)}")
        print("\n🔧 POSSÍVEIS SOLUÇÕES:")
        print("1. Verifique se a porta 5000 está livre")
        print("2. Execute como administrador")
        print("3. Verifique permissões de arquivo")
        print("4. Verifique se as variáveis de ambiente estão corretas")
        sys.exit(1)
    finally:
        logger.info(f"Servidor Flask com PID {os.getpid()} finalizado")