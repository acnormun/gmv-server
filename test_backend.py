# test_backend.py - Teste rápido do backend
import os
import sys
import subprocess
import time

def test_python():
    """Testa Python e dependências"""
    print("🐍 TESTE PYTHON:")
    print(f"   Versão: {sys.version}")
    print(f"   Executável: {sys.executable}")
    
    # Testa dependências essenciais
    deps = ['flask', 'pandas']
    for dep in deps:
        try:
            __import__(dep)
            print(f"    {dep}")
        except ImportError:
            print(f"   ❌ {dep} - Execute: pip install {dep}")
            return False
    return True

def test_files():
    """Testa estrutura de arquivos"""
    print("\n📁 TESTE ARQUIVOS:")
    
    files_to_check = [
        ('app.py', 'Arquivo principal'),
        ('.env', 'Configuração (pode não existir)'),
        ('data/', 'Pasta de dados (será criada)'),
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            print(f"    {description}: {file_path}")
        else:
            print(f"   ⚠️ {description}: {file_path} (não existe)")
    
    return True

def test_backend_start():
    """Testa inicialização do backend"""
    print("\n🚀 TESTE BACKEND:")
    print("   Iniciando servidor...")
    
    try:
        # Inicia o backend em processo separado
        process = subprocess.Popen(
            [sys.executable, 'app.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Aguarda alguns segundos
        time.sleep(3)
        
        # Verifica se ainda está rodando
        if process.poll() is None:
            print("    Servidor iniciou com sucesso!")
            
            # Para o servidor
            process.terminate()
            process.wait(timeout=5)
            print("    Servidor finalizado")
            return True
        else:
            # Processo morreu, pega output
            stdout, stderr = process.communicate()
            print("   ❌ Servidor falhou!")
            if stderr:
                print(f"   Erro: {stderr[:500]}")
            if stdout:
                print(f"   Output: {stdout[:500]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao testar: {e}")
        return False

def test_imports():
    """Testa imports específicos do app"""
    print("\n📦 TESTE IMPORTS:")
    
    imports_to_test = [
        ('flask', 'Flask, jsonify, request'),
        ('pandas', None),  # Corrigido: não tenta importar 'pd' 
        ('re', None),
        ('os', None),
        ('io', None),
        ('sys', None),
        ('logging', None),
        ('datetime', None),
    ]
    
    all_good = True
    for module, submodules in imports_to_test:
        try:
            if submodules:
                exec(f"from {module} import {submodules}")
            else:
                __import__(module)
            print(f"    {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            all_good = False
    
    # Teste específico do pandas
    try:
        import pandas as pd
        print(f"    pandas as pd")
    except ImportError as e:
        print(f"   ❌ pandas as pd: {e}")
        all_good = False
    
    return all_good

def main():
    print("🧪 TESTE COMPLETO DO BACKEND GMV")
    print("=" * 40)
    
    tests = [
        ("Python e dependências", test_python),
        ("Imports", test_imports), 
        ("Estrutura de arquivos", test_files),
        ("Inicialização do backend", test_backend_start),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Erro no teste {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo
    print(f"\n📋 RESUMO DOS TESTES:")
    print("=" * 30)
    all_passed = True
    for test_name, passed in results:
        status = " PASSOU" if passed else "❌ FALHOU"
        print(f"   {status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print(" Backend está pronto para usar")
        print("💡 Execute: python app.py")
    else:
        print(f"\n⚠️ ALGUNS TESTES FALHARAM")
        print("🔧 Resolva os problemas acima antes de executar o backend")
        print("\n💡 SOLUÇÕES COMUNS:")
        print("1. pip install flask pandas")
        print("2. Verifique se app.py existe")
        print("3. Execute como administrador")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Teste interrompido")
    except Exception as e:
        print(f"\n❌ Erro no teste: {e}")
    finally:
        input("\nPressione Enter para sair...")