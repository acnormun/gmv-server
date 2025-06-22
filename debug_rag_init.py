#!/usr/bin/env python3
"""
Debug Inicialização RAG - GMV Sistema
Diagnostica problemas específicos na inicialização do RAG
"""

import requests
import sys
import time
import os
from pathlib import Path
import tempfile

def print_step(num, desc):
    print(f"\n {num}. {desc}")
    print("-" * 50)

def test_dependencies():
    """Testa dependências específicas do RAG"""
    print_step(1, "TESTANDO DEPENDÊNCIAS RAG")
    
    deps = [
        'langchain',
        'langchain_community', 
        'langchain_core',
        'ollama',
        'faiss',
        'sentence_transformers',
        'numpy',
        'sklearn'
    ]
    
    missing = []
    
    for dep in deps:
        try:
            if dep == 'faiss':
                import faiss
                print(f" {dep} - versão disponível")
            elif dep == 'sentence_transformers':
                import sentence_transformers
                print(f" {dep} - versão {sentence_transformers.__version__}")
            elif dep == 'sklearn':
                import sklearn
                print(f" {dep} - versão {sklearn.__version__}")
            else:
                __import__(dep)
                print(f" {dep}")
        except ImportError as e:
            print(f" {dep} - {e}")
            missing.append(dep)
        except Exception as e:
            print(f" {dep} - {e}")
    
    if missing:
        print(f"\n🚨 Dependências ausentes: {missing}")
        print("Execute: pip install " + " ".join(missing))
        return False
    
    return True

def create_test_documents():
    """Cria documentos de teste"""
    print_step(2, "CRIANDO DOCUMENTOS DE TESTE")
    
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Cria alguns arquivos de teste
    test_docs = [
        {
            "name": "processo_001.md",
            "content": """# Processo GMV-001

**Data:** 2024-01-15
**Responsável:** João Silva
**Status:** Em andamento
**Tema:** CÍVEL

## Descrição
Processo relacionado a irregularidades contratuais da empresa TESTE LTDA.

## Suspeitos
- Carlos Alberto da Silva
- Maria Santos

## Irregularidades
- Documentos fraudulentos
- Contratos irregulares
"""
        },
        {
            "name": "processo_002.md", 
            "content": """# Processo GMV-002

**Data:** 2024-01-20
**Responsável:** Ana Costa
**Status:** Para revisão
**Tema:** TRABALHISTA

## Descrição
Investigação de práticas trabalhistas irregulares.

## Suspeitos
- LOCK ADVOGADOS
- Roberto Mendes

## Irregularidades
- Não pagamento de direitos
- Documentação inadequada
"""
        }
    ]
    
    created = 0
    
    for doc in test_docs:
        file_path = data_dir / doc["name"]
        
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            print(f" Criado: {doc['name']}")
            created += 1
        else:
            print(f"📄 Já existe: {doc['name']}")
    
    # Verifica total de arquivos
    all_files = list(data_dir.rglob("*.md"))
    print(f"\nTotal de arquivos .md: {len(all_files)}")
    
    return len(all_files) > 0

def test_rag_components():
    """Testa componentes do RAG individualmente"""
    print_step(3, "TESTANDO COMPONENTES RAG")
    
    try:
        # Testa imports básicos
        print("📦 Testando imports...")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        from langchain.llms import Ollama
        from langchain.embeddings import OllamaEmbeddings
        print(" Imports LangChain OK")
        
        # Testa conexão Ollama via LangChain
        print(" Testando conexão LangChain -> Ollama...")
        llm = Ollama(
            model="llama3.1",
            temperature=0.1,
            num_predict=50  # Resposta curta
        )
        
        response = llm("Responda apenas 'OK'")
        print(f" LLM resposta: '{response[:50]}...'")
        
        # Testa embeddings
        print("🔤 Testando embeddings...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_embedding = embeddings.embed_query("teste")
        print(f" Embedding gerado (dim: {len(test_embedding)})")
        
        # Testa text splitter
        print("✂️ Testando text splitter...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=10
        )
        chunks = splitter.split_text("Este é um teste de texto para dividir em chunks menores.")
        print(f" Text splitter OK ({len(chunks)} chunks)")
        
        return True
        
    except Exception as e:
        print(f" Erro nos componentes: {e}")
        return False

def test_rag_init_step_by_step():
    """Testa inicialização RAG passo a passo"""
    print_step(4, "TESTE INICIALIZAÇÃO PASSO-A-PASSO")
    
    # Primeiro, verifica se o endpoint responde rápido
    print("🏥 Testando health check...")
    try:
        response = requests.get("http://localhost:5000/api/rag/health", timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Resposta: {data}")
        else:
            print(f"Erro: {response.text}")
    except Exception as e:
        print(f" Health check falhou: {e}")
        return False
    
    # Testa stats (não deveria travar)
    print("\nTestando stats...")
    try:
        response = requests.get("http://localhost:5000/api/rag/stats", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Stats: {data}")
        else:
            print(f"Erro: {response.text}")
    except Exception as e:
        print(f" Stats falharam: {e}")
    
    # Agora tenta init com timeout menor
    print(f"\n Testando inicialização (timeout 30s)...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:5000/api/rag/init",
            timeout=30
        )
        end_time = time.time()
        
        print(f"⏱️ Tempo decorrido: {end_time - start_time:.2f}s")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f" Sucesso: {data}")
            return True
        else:
            print(f" Erro: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(" Timeout após 30s - RAG está travando na inicialização")
        return False
    except Exception as e:
        print(f" Erro: {e}")
        return False

def check_backend_logs():
    """Verifica logs do backend"""
    print_step(5, "VERIFICANDO LOGS")
    
    log_files = ['server.log', 'app.log', 'gmv.log']
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"📜 Log encontrado: {log_file}")
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        print("Últimas 10 linhas:")
                        for i, line in enumerate(lines[-10:], 1):
                            print(f"  {i:2d}: {line.strip()}")
                    else:
                        print("  Arquivo vazio")
            except Exception as e:
                print(f"  Erro ao ler: {e}")
        else:
            print(f"📜 Log não encontrado: {log_file}")

def test_adaptive_rag_direct():
    """Testa adaptive_rag.py diretamente"""
    print_step(6, "TESTE DIRETO ADAPTIVE_RAG")
    
    try:
        print("📥 Importando adaptive_rag...")
        
        # Tenta importar
        sys.path.append('.')
        from adaptive_rag import AdaptiveRAG, RAGConfig
        
        print(" Import bem-sucedido")
        
        # Testa criação da instância
        print("🔧 Criando instância RAG...")
        config = RAGConfig(
            chunk_size=200,  # Menor para teste
            temperature=0.1
        )
        rag = AdaptiveRAG(config)
        print(" Instância criada")
        
        # Testa inicialização
        print(" Testando inicialização direta...")
        success = rag.initialize()
        
        if success:
            print(" Inicialização direta bem-sucedida!")
            
            # Testa carregamento de documentos
            print("📄 Testando carregamento de documentos...")
            docs_count = rag.load_documents_from_directory("./data")
            print(f" {docs_count} documentos carregados")
            
            return True
        else:
            print(" Inicialização direta falhou")
            return False
            
    except ImportError as e:
        print(f" Erro de import: {e}")
        print("   Verifique se adaptive_rag.py existe no diretório")
        return False
    except Exception as e:
        print(f" Erro na inicialização: {e}")
        import traceback
        print("Traceback completo:")
        traceback.print_exc()
        return False

def provide_solutions():
    """Fornece soluções específicas"""
    print_step(7, "SOLUÇÕES RECOMENDADAS")
    
    print("🔧 PROBLEMAS IDENTIFICADOS E SOLUÇÕES:")
    
    print("\n1. RAG TRAVANDO NA INICIALIZAÇÃO:")
    print("   Possíveis causas:")
    print("   • Problema com FAISS (vector store)")
    print("   • Problema com embeddings")
    print("   • Loop infinito no código")
    print("   • Dependência ausente")
    
    print("\n2. SOLUÇÕES IMEDIATAS:")
    print("   a) Reinstalar dependências:")
    print("      pip install --force-reinstall langchain langchain-community faiss-cpu")
    
    print("\n   b) Testar com dados mínimos:")
    print("      • Use apenas 1-2 documentos pequenos")
    print("      • Reduza chunk_size para 100")
    
    print("\n   c) Verificar logs detalhados:")
    print("      • Execute: python app.py > debug.log 2>&1")
    print("      • Em outro terminal: tail -f debug.log")
    print("      • Tente inicializar RAG")
    
    print("\n   d) Teste alternativo:")
    print("      • Modifique adaptive_rag.py para adicionar prints/logs")
    print("      • Identifique onde está travando")
    
    print("\n3. CONFIGURAÇÃO DE EMERGÊNCIA:")
    print("   Se nada funcionar, crie um RAG simplificado:")
    print("   • Desabilite vector store temporariamente")
    print("   • Use apenas texto simples")
    print("   • Adicione timeouts internos")

def main():
    """Função principal de debug"""
    print("🐛 DEBUG INICIALIZAÇÃO RAG - GMV SISTEMA")
    print("Este script vai diagnosticar por que o RAG está travando")
    print("="*65)
    
    # Executa testes
    tests = [
        ("Dependências RAG", test_dependencies),
        ("Documentos de teste", create_test_documents),
        ("Componentes RAG", test_rag_components),
        ("Inicialização step-by-step", test_rag_init_step_by_step),
        ("Adaptive RAG direto", test_adaptive_rag_direct)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = test_func()
            results.append((name, result))
            
            if not result:
                print(f" Teste '{name}' falhou - investigando...")
            
        except Exception as e:
            print(f" Erro no teste '{name}': {e}")
            results.append((name, False))
    
    # Verifica logs sempre
    check_backend_logs()
    
    # Resumo
    print("\n" + "="*65)
    print("RESUMO DO DIAGNÓSTICO")
    print("="*65)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = " PASSOU" if result else " FALHOU"
        print(f"{status:12} {name}")
    
    print(f"\n📈 Score: {passed}/{total}")
    
    if passed < total:
        provide_solutions()
    else:
        print("\n TODOS OS TESTES PASSARAM!")
        print("O problema pode estar em configurações específicas ou timing.")
        print("Tente:")
        print("1. Reiniciar o backend: python app.py")
        print("2. Testar novamente: python quick_test.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Debug interrompido")
    except Exception as e:
        print(f"\n Erro no diagnóstico: {e}")
        import traceback
        traceback.print_exc()