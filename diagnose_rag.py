#!/usr/bin/env python3
"""
Diagnóstico Completo RAG - GMV Sistema
Identifica problemas de configuração e dependências
"""

import requests
import subprocess
import sys
import os
import json
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_check(item, status, details=""):
    emoji = "" if status else ""
    print(f"{emoji} {item}")
    if details:
        print(f"   └─ {details}")

def check_python_deps():
    """Verifica dependências Python"""
    print_header("DEPENDÊNCIAS PYTHON")
    
    required_deps = [
        'flask',
        'langchain', 
        'langchain_community',
        'ollama',
        'faiss',
        'sentence_transformers',
        'numpy',
        'pandas'
    ]
    
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep)
            print_check(f"{dep}", True)
        except ImportError as e:
            print_check(f"{dep}", False, f"Erro: {e}")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n DEPENDÊNCIAS AUSENTES:")
        print("Execute os seguintes comandos:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True

def check_ollama():
    """Verifica Ollama"""
    print_header("OLLAMA")
    
    # 1. Verifica se Ollama está instalado
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_check("Ollama instalado", True, version)
        else:
            print_check("Ollama instalado", False, "Comando ollama não encontrado")
            print("    Instale em: https://ollama.com/download")
            return False
    except FileNotFoundError:
        print_check("Ollama instalado", False, "Comando ollama não encontrado")
        print("    Instale em: https://ollama.com/download")
        return False
    except Exception as e:
        print_check("Ollama instalado", False, f"Erro: {e}")
        return False
    
    # 2. Verifica se está rodando
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print_check("Ollama rodando", True, "http://localhost:11434")
            
            # 3. Verifica modelos
            data = response.json()
            models = [m.get('name', '') for m in data.get('models', [])]
            
            required_models = ['llama3.1:8b', 'nomic-embed-text']
            
            for model in required_models:
                found = any(model in m for m in models)
                if found:
                    print_check(f"Modelo {model}", True)
                else:
                    print_check(f"Modelo {model}", False, f"Execute: ollama pull {model}")
            
            return len([m for m in required_models if any(m in model for model in models)]) == len(required_models)
            
        else:
            print_check("Ollama rodando", False, f"HTTP {response.status_code}")
            print("    Execute: ollama serve")
            return False
            
    except requests.exceptions.ConnectionError:
        print_check("Ollama rodando", False, "Conexão recusada")
        print("    Execute: ollama serve")
        return False
    except Exception as e:
        print_check("Ollama rodando", False, f"Erro: {e}")
        return False

def check_backend():
    """Verifica backend Flask"""
    print_header("BACKEND FLASK")
    
    try:
        # Health check
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print_check("Backend rodando", True, "http://localhost:5000")
            
            # Verifica rotas RAG
            rag_routes = [
                '/api/rag/health',
                '/api/rag/stats',
                '/api/rag/init'
            ]
            
            for route in rag_routes:
                try:
                    r = requests.get(f"http://localhost:5000{route}", timeout=3)
                    print_check(f"Rota {route}", r.status_code < 500, f"Status: {r.status_code}")
                except:
                    print_check(f"Rota {route}", False, "Não responde")
            
            return True
        else:
            print_check("Backend rodando", False, f"HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_check("Backend rodando", False, "Conexão recusada")
        print("    Execute: python app.py")
        return False
    except Exception as e:
        print_check("Backend rodando", False, f"Erro: {e}")
        return False

def check_environment():
    """Verifica configurações do ambiente"""
    print_header("CONFIGURAÇÕES AMBIENTE")
    
    # Verifica arquivo .env
    env_file = Path('.env')
    if env_file.exists():
        print_check(".env existe", True)
        
        # Lê variáveis importantes
        env_vars = {}
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        except Exception as e:
            print_check("Leitura .env", False, f"Erro: {e}")
        
        # Verifica variáveis importantes
        important_vars = [
            'PATH_TRIAGEM',
            'PASTA_DESTINO', 
            'PASTA_DAT',
            'DADOS_ANONIMOS'
        ]
        
        for var in important_vars:
            value = env_vars.get(var) or os.getenv(var)
            if value:
                print_check(f"Variável {var}", True, value)
                
                # Verifica se pasta existe
                if var in ['PASTA_DESTINO', 'PASTA_DAT', 'DADOS_ANONIMOS']:
                    if os.path.exists(value):
                        files = len(list(Path(value).rglob('*.*'))) if os.path.isdir(value) else 1
                        print_check(f"  Pasta {var} existe", True, f"{files} arquivos")
                    else:
                        print_check(f"  Pasta {var} existe", False, "Pasta não encontrada")
            else:
                print_check(f"Variável {var}", False, "Não definida")
                
    else:
        print_check(".env existe", False, "Arquivo não encontrado")
        print("    Execute o setup para criar: python setup_rag.py")

def check_rag_detailed():
    """Verifica RAG em detalhes"""
    print_header("DIAGNÓSTICO RAG DETALHADO")
    
    try:
        # Tenta inicialização com detalhes
        print(" Tentando inicializar RAG...")
        
        response = requests.post(
            "http://localhost:5000/api/rag/init",
            timeout=30
        )
        
        print(f"Status HTTP: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Resposta: {json.dumps(data, indent=2)}")
            
            if data.get("success"):
                print_check("Inicialização RAG", True, f"{data.get('documents_loaded', 0)} docs carregados")
                return True
            else:
                print_check("Inicialização RAG", False, data.get('message', 'Erro desconhecido'))
                return False
        else:
            error_text = response.text
            print_check("Inicialização RAG", False, f"HTTP {response.status_code}")
            print(f"Erro detalhado: {error_text}")
            return False
            
    except Exception as e:
        print_check("Inicialização RAG", False, f"Exceção: {e}")
        return False

def get_detailed_logs():
    """Tenta obter logs detalhados"""
    print_header("LOGS E DIAGNÓSTICOS")
    
    # Verifica logs do servidor
    log_files = ['server.log', 'app.log', 'gmv.log']
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print_check(f"Log {log_file}", True)
            try:
                # Mostra últimas linhas
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print("   Últimas linhas:")
                        for line in lines[-5:]:
                            print(f"   {line.strip()}")
            except Exception as e:
                print(f"   Erro ao ler log: {e}")
        else:
            print_check(f"Log {log_file}", False, "Arquivo não encontrado")

def provide_solutions():
    """Fornece soluções baseadas nos problemas encontrados"""
    print_header("SOLUÇÕES RECOMENDADAS")
    
    print("🔧 CHECKLIST DE CORREÇÃO:")
    
    print("\n1. OLLAMA:")
    print("   - Instalar: https://ollama.com/download")
    print("   - Iniciar: ollama serve")
    print("   - Modelos: ollama pull llama3.1:8b && ollama pull nomic-embed-text")
    
    print("\n2. DEPENDÊNCIAS PYTHON:")
    print("   - pip install langchain langchain-community ollama faiss-cpu sentence-transformers")
    
    print("\n3. BACKEND:")
    print("   - Verificar se app.py está no diretório correto")
    print("   - Executar: python app.py")
    print("   - Verificar logs: tail -f server.log")
    
    print("\n4. CONFIGURAÇÃO:")
    print("   - Criar/verificar arquivo .env")
    print("   - Definir DADOS_ANONIMOS=./data")
    print("   - Criar pastas necessárias: mkdir -p data/processos data/dat")
    
    print("\n5. TESTE PASSO-A-PASSO:")
    print("   - ollama serve &")
    print("   - python app.py &")
    print("   - python quick_test.py")

def main():
    """Função principal de diagnóstico"""
    print("🏥 DIAGNÓSTICO COMPLETO RAG - GMV SISTEMA")
    print("Este script vai identificar todos os problemas de configuração")
    
    checks = [
        ("Dependências Python", check_python_deps),
        ("Ollama", check_ollama),
        ("Backend Flask", check_backend),
        ("Configurações", check_environment),
        ("RAG Detalhado", check_rag_detailed)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f" Erro no diagnóstico {name}: {e}")
            results.append((name, False))
    
    # Logs
    get_detailed_logs()
    
    # Resumo
    print_header("RESUMO DO DIAGNÓSTICO")
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = " OK" if result else " PROBLEMA"
        print(f"{status:12} {name}")
        if result:
            passed += 1
    
    print(f"\nStatus geral: {passed}/{total} verificações passaram")
    
    if passed < total:
        provide_solutions()
    else:
        print("\n TUDO OK! Seu sistema deveria estar funcionando.")
        print("Se ainda há problemas, verifique os logs detalhadamente.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Diagnóstico interrompido")
    except Exception as e:
        print(f"\n Erro no diagnóstico: {e}")