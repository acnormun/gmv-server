import os

def setup_environment():
    """Setup automático das variáveis de ambiente com LIMPEZA FORÇADA DE CACHE"""
    print("SETUP AUTOMÁTICO DO GMV SISTEMA")
    print("=" * 40)
    
    # LIMPEZA FORÇADA - Remove TODAS as variáveis relacionadas ao GMV
    old_vars = ['PATH_TRIAGEM', 'PASTA_DESTINO', 'PASTA_DAT', 'GITHUB_TOKEN']
    removed_count = 0
    
    print("LIMPEZA FORÇADA DE CACHE:")
    for var in old_vars:
        if var in os.environ:
            old_value = os.environ[var]
            del os.environ[var]
            print(f"  Removido: {var} = {old_value}")
            removed_count += 1
    
    if removed_count == 0:
        print("   Nenhuma variável em cache (primeira execução)")
    else:
        print(f"   {removed_count} variáveis antigas removidas do cache")
    
    # Verifica se python-dotenv está disponível
    try:
        from dotenv import load_dotenv
        dotenv_available = True
        print("python-dotenv disponível")
    except ImportError:
        dotenv_available = False
        print("python-dotenv não instalado - usando método manual")
    
    # Procura arquivo .env
    env_file = '.env'
    env_path = os.path.abspath(env_file)
    
    print(f"\nVERIFICAÇÃO DO ARQUIVO .ENV:")
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
            print(f"    Erro ao ler .env: {e}")
    else:
        print(f"   Arquivo .env não encontrado, criando...")
        create_default_env()
    
    print(f"\n CARREGAMENTO FORÇADO:")
    # Carrega variáveis com OVERRIDE forçado
    if dotenv_available:
        print(f"    Usando python-dotenv com override=True")
        from dotenv import load_dotenv
        result = load_dotenv(env_file, override=True, verbose=True)
        print(f"    Resultado: {result}")
    else:
        print(f"    Carregando manualmente...")
        load_env_manual(env_file)
    
    # Verifica se carregou corretamente
    PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
    PASTA_DESTINO = os.getenv("PASTA_DESTINO")
    PASTA_DAT = os.getenv("PASTA_DAT")
    
    # Se ainda não carregou, tenta novamente
    if not PATH_TRIAGEM or not PASTA_DESTINO:
        print("\n Variáveis essenciais não definidas! Tentando corrigir...")
        
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
        
        print(f"    APÓS CORREÇÃO:")
        print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
        print(f"   PASTA_DESTINO: {PASTA_DESTINO}")
        print(f"   PASTA_DAT: {PASTA_DAT}")
    
    # Cria estrutura de pastas
    setup_directories()
    
    print("Setup concluído!\n")
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
        print(f"    Arquivo .env criado: {os.path.abspath('.env')}")
        print(f"    Edite o arquivo se quiser usar outros caminhos")
        return True
    except Exception as e:
        print(f"    Erro ao criar .env: {e}")
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
                            print(f"      {key} = ***HIDDEN***")
                        else:
                            print(f"      {key} = {value}")
                            
                    except Exception as e:
                        print(f"       Erro na linha {line_num}: {original_line.strip()} - {e}")
                else:
                    print(f"       Linha {line_num} inválida (sem =): {original_line.strip()}")
        
        print(f"    Total: {len(loaded_vars)} variáveis carregadas manualmente")
        return True
        
    except Exception as e:
        print(f"    Erro ao carregar .env manualmente: {e}")
        return False

def setup_directories():
    """Cria estrutura de diretórios"""
    print(" Verificando/criando diretórios...")
    
    PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
    PASTA_DESTINO = os.getenv("PASTA_DESTINO") 
    PASTA_DAT = os.getenv("PASTA_DAT")
    
    # Cria diretórios
    for pasta, nome in [(PASTA_DESTINO, "PASTA_DESTINO"), (PASTA_DAT, "PASTA_DAT")]:
        if pasta:
            try:
                os.makedirs(pasta, exist_ok=True)
                print(f"   {nome}: {pasta}")
            except Exception as e:
                print(f"   Erro ao criar {nome}: {e}")
    
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
                print(f"   Arquivo de triagem criado: {PATH_TRIAGEM}")
            else:
                print(f"   PATH_TRIAGEM: {PATH_TRIAGEM}")
        except Exception as e:
            print(f"   Erro ao criar arquivo de triagem: {e}")

if __name__ == "__main__":
    print(" GMV SISTEMA - INICIALIZAÇÃO COM LIMPEZA DE CACHE")
    print("=" * 60)
    print(f" Diretório de trabalho: {os.getcwd()}")
    print(f" Procurando .env em: {os.path.abspath('.env')}")

    PATH_TRIAGEM, PASTA_DESTINO, PASTA_DAT = setup_environment()

    if not PATH_TRIAGEM or not PASTA_DESTINO:
        print("\n ERRO CRÍTICO: Não foi possível configurar variáveis de ambiente!")
        print(" DEPURAÇÃO:")
        print("   1. Verifique se você tem permissão para criar arquivos neste diretório")
        print("   2. Verifique se o arquivo .env foi criado corretamente")
        print("   3. Tente executar como administrador")
        print(f"   4. Arquivo .env deveria estar em: {os.path.abspath('.env')}")
        sys.exit(1)

    print(" TESTE FINAL DE VERIFICAÇÃO:")
    print("=" * 40)
    final_vars = {
        'PATH_TRIAGEM': os.getenv('PATH_TRIAGEM'),
        'PASTA_DESTINO': os.getenv('PASTA_DESTINO'), 
        'PASTA_DAT': os.getenv('PASTA_DAT')
    }

    for var_name, var_value in final_vars.items():
        print(f"{var_name} = {var_value}")
        if var_value:
            abs_path = os.path.abspath(var_value)
            print(f"   Caminho absoluto: {abs_path}")

    print("\nCONFIRMAÇÃO:")
    print(f"Cache de variáveis antigas foi limpo")
    print(f"Arquivo .env atual foi carregado")
    print(f"{len([v for v in final_vars.values() if v])} variáveis essenciais definidas")
    print("=" * 60)
