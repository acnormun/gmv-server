#!/usr/bin/env python3
"""
Fix Timeout RAG - GMV Sistema
Diagnóstica e corrige problemas de timeout
"""

import requests
import subprocess
import time
import sys
import os
import signal
from pathlib import Path

def print_step(step, desc):
    print(f"\n🔧 {step}. {desc}")
    print("-" * 50)

def run_command(cmd, timeout=10):
    """Executa comando com timeout"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)

def check_ollama_process():
    """Verifica se Ollama está rodando"""
    print_step(1, "VERIFICANDO PROCESSO OLLAMA")
    
    # Verifica processo
    success, stdout, stderr = run_command("pgrep -f ollama", 5)
    
    if success and stdout.strip():
        pids = stdout.strip().split('\n')
        print(f" Ollama está rodando (PIDs: {', '.join(pids)})")
        return True
    else:
        print(" Ollama NÃO está rodando")
        return False

def kill_ollama():
    """Mata todos os processos Ollama"""
    print(" Matando processos Ollama existentes...")
    
    commands = [
        "pkill -f ollama",
        "killall ollama 2>/dev/null || true",
        "ps aux | grep ollama | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true"
    ]
    
    for cmd in commands:
        run_command(cmd, 5)
    
    time.sleep(2)
    print(" Processos Ollama finalizados")

def start_ollama():
    """Inicia Ollama"""
    print_step(2, "INICIANDO OLLAMA")
    
    # Mata processos existentes primeiro
    kill_ollama()
    
    print(" Iniciando Ollama...")
    
    # Inicia Ollama em background
    try:
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        print(f" Ollama iniciado (PID: {process.pid})")
        
        # Aguarda inicialização
        print("⏳ Aguardando Ollama inicializar...")
        
        for i in range(15):  # 15 segundos
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print(f" Ollama respondendo após {i+1}s")
                    return True
            except:
                pass
            
            time.sleep(1)
            print(f"   {i+1}/15 segundos...")
        
        print(" Ollama não respondeu após 15 segundos")
        return False
        
    except FileNotFoundError:
        print(" Comando 'ollama' não encontrado!")
        print("    Instale Ollama: https://ollama.com/download")
        return False
    except Exception as e:
        print(f" Erro ao iniciar Ollama: {e}")
        return False

def test_ollama_connection():
    """Testa conexão com Ollama"""
    print_step(3, "TESTANDO CONEXÃO OLLAMA")
    
    # Teste básico
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print(" Ollama respondendo")
            
            # Lista modelos
            data = response.json()
            models = [m.get('name', '') for m in data.get('models', [])]
            
            print(f"📦 Modelos disponíveis: {len(models)}")
            for model in models:
                print(f"   - {model}")
            
            return True, models
        else:
            print(f" Ollama retornou status {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print(" Não conseguiu conectar ao Ollama")
        return False, []
    except requests.exceptions.Timeout:
        print(" Timeout ao conectar ao Ollama")
        return False, []
    except Exception as e:
        print(f" Erro: {e}")
        return False, []

def download_models():
    """Baixa modelos necessários"""
    print_step(4, "VERIFICANDO/BAIXANDO MODELOS")
    
    required_models = ["llama3.1:8b", "nomic-embed-text"]
    
    # Verifica modelos existentes
    success, models = test_ollama_connection()
    if not success:
        print(" Não foi possível verificar modelos (Ollama não responde)")
        return False
    
    for model in required_models:
        model_exists = any(model in m for m in models)
        
        if model_exists:
            print(f" {model} já está baixado")
        else:
            print(f"📥 Baixando {model}...")
            
            # Baixa modelo
            cmd = f"ollama pull {model}"
            print(f"   Executando: {cmd}")
            
            try:
                process = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Mostra progresso
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(f"   {output.strip()}")
                
                if process.returncode == 0:
                    print(f" {model} baixado com sucesso!")
                else:
                    print(f" Erro ao baixar {model}")
                    return False
                    
            except Exception as e:
                print(f" Erro ao baixar {model}: {e}")
                return False
    
    return True

def test_ollama_model():
    """Testa modelo Ollama"""
    print_step(5, "TESTANDO MODELO OLLAMA")
    
    test_prompt = "Responda apenas 'OK' para confirmar que está funcionando."
    
    try:
        print("🧪 Testando llama3.1...")
        
        payload = {
            "model": "llama3.1:8b",
            "prompt": test_prompt,
            "stream": False
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data.get('response', '')
            print(f" Modelo respondeu: '{result[:50]}...'")
            return True
        else:
            print(f" Erro HTTP {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(" Timeout no teste do modelo (>30s)")
        return False
    except Exception as e:
        print(f" Erro no teste: {e}")
        return False

def test_rag_init():
    """Testa inicialização RAG"""
    print_step(6, "TESTANDO INICIALIZAÇÃO RAG")
    
    try:
        print(" Tentando inicializar RAG...")
        
        response = requests.post(
            "http://localhost:5000/api/rag/init",
            timeout=60  # 1 minuto
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                docs = data.get("documents_loaded", 0)
                print(f" RAG inicializado! {docs} documentos carregados")
                return True
            else:
                print(f" Falha na inicialização: {data.get('message')}")
                return False
        else:
            print(f" Erro HTTP {response.status_code}")
            print(f"Resposta: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(" Timeout na inicialização RAG (>60s)")
        print("   Isso pode indicar que o Ollama está muito lento")
        return False
    except Exception as e:
        print(f" Erro: {e}")
        return False

def create_data_directory():
    """Cria diretório de dados se não existir"""
    print_step(7, "VERIFICANDO DADOS")
    
    data_dir = Path("./data")
    
    if not data_dir.exists():
        print(" Criando diretório ./data/")
        data_dir.mkdir(exist_ok=True)
        
        # Cria subdiretórios
        (data_dir / "processos").mkdir(exist_ok=True)
        (data_dir / "dat").mkdir(exist_ok=True)
        
        print(" Diretórios criados")
    else:
        print(" Diretório ./data/ já existe")
    
    # Verifica arquivos
    files = list(data_dir.rglob("*.*"))
    print(f"📄 Encontrados {len(files)} arquivos para processar")
    
    return True

def main():
    """Função principal de correção"""
    print("🛠️ CORREÇÃO DE TIMEOUT RAG - GMV SISTEMA")
    print("Este script vai corrigir problemas de timeout passo-a-passo")
    print("="*60)
    
    # Verifica se é problema de Ollama
    if not check_ollama_process():
        # Ollama não está rodando - inicia
        if not start_ollama():
            print("\n FALHA CRÍTICA: Não foi possível iniciar Ollama")
            print(" Soluções:")
            print("   1. Instale Ollama: https://ollama.com/download")
            print("   2. Reinicie o sistema")
            print("   3. Execute manualmente: ollama serve")
            sys.exit(1)
    else:
        # Ollama está rodando - testa conexão
        success, models = test_ollama_connection()
        if not success:
            print(" Ollama está rodando mas não responde - reiniciando...")
            if not start_ollama():
                print(" Falha ao reiniciar Ollama")
                sys.exit(1)
    
    # Baixa modelos se necessário
    if not download_models():
        print(" Falha ao configurar modelos")
        sys.exit(1)
    
    # Testa modelo
    if not test_ollama_model():
        print(" Modelo não está funcionando corretamente")
        sys.exit(1)
    
    # Cria dados
    create_data_directory()
    
    # Testa RAG
    if not test_rag_init():
        print(" RAG ainda não funciona após correções")
        print("\n Diagnóstico adicional necessário:")
        print("   1. Verifique logs: tail -f server.log")
        print("   2. Teste manual: python diagnose_rag.py")
        print("   3. Verifique dependências: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n CORREÇÃO CONCLUÍDA COM SUCESSO!")
    print(" Ollama funcionando")
    print(" Modelos carregados")
    print(" RAG inicializado")
    
    print("\n🧪 Testando consulta rápida...")
    
    # Teste final
    try:
        response = requests.post(
            "http://localhost:5000/api/rag/query",
            json={"question": "Teste básico - responda OK"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(" Sistema RAG funcionando perfeitamente!")
            else:
                print(f" RAG responde mas com erro: {data.get('message')}")
        else:
            print(f" Problema na consulta: {response.status_code}")
    except Exception as e:
        print(f" Erro no teste final: {e}")
    
    print("\n Agora você pode usar:")
    print("   python quick_test.py")
    print("   python test_rag_simple.py 'sua pergunta'")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Correção interrompida")
    except Exception as e:
        print(f"\n Erro inesperado: {e}")