#!/usr/bin/env python3
import requests
import sys

def quick_test():
    """Teste rápido de funcionamento"""
    print(" TESTE RÁPIDO RAG")
    print("="*30)
    
    try:
        # 1. Teste backend
        print("1️⃣ Testando backend...")
        r = requests.get("http://localhost:5000/health", timeout=3)
        if r.status_code == 200:
            print("    Backend OK")
        else:
            print("    Backend com problemas")
            return False
        
        # 2. Teste RAG status
        print("2️⃣ Testando RAG...")
        r = requests.get("http://localhost:5000/api/rag/health", timeout=5)
        
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "healthy":
                print("    RAG OK")
            else:
                print("    RAG precisa inicializar")
                # Tenta inicializar
                print("    Inicializando...")
                init_r = requests.post("http://localhost:5000/api/rag/init", timeout=20)
                if init_r.status_code == 200:
                    print("    RAG inicializado!")
                else:
                    print("    Falha na inicialização")
                    return False
        else:
            print("    RAG indisponível")
            return False
        
        # 3. Teste consulta simples
        print("3️⃣ Testando consulta...")
        
        question = "Quantos processos existem no sistema?"
        r = requests.post(
            "http://localhost:5000/api/rag/query",
            json={"question": question},
            timeout=15
        )
        
        if r.status_code == 200:
            data = r.json()
            if data.get("success"):
                result = data.get("data", {})
                answer = result.get("answer", "Sem resposta")
                query_type = result.get("query_type", "unknown")
                sources = len(result.get("source_documents", []))
                
                print("    Consulta OK")
                print(f"   Pergunta: {question}")
                print(f"    Resposta: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                print(f"   🎯 Tipo: {query_type}")
                print(f"   📚 Fontes: {sources}")
                
                # 4. Teste suspeitos (se possível)
                print("4️⃣ Testando detecção de suspeitos...")
                
                suspeito_question = "Quais suspeitos estão relacionados à LOCK ADVOGADOS?"
                r = requests.post(
                    "http://localhost:5000/api/rag/query",
                    json={"question": suspeito_question},
                    timeout=10
                )
                
                if r.status_code == 200:
                    data = r.json()
                    if data.get("success"):
                        result = data.get("data", {})
                        suspeitos = result.get("suspeitos", [])
                        
                        if suspeitos:
                            print(f"    Suspeitos detectados: {len(suspeitos)}")
                            print(f"    Exemplos: {', '.join(suspeitos[:3])}")
                        else:
                            print("    Nenhum suspeito detectado (pode ser normal)")
                    else:
                        print("    Consulta de suspeitos falhou")
                else:
                    print("    Erro na consulta de suspeitos")
                
                print("\n TESTE CONCLUÍDO COM SUCESSO!")
                print(" Seu sistema RAG está funcionando!")
                return True
                
            else:
                print(f"    Erro na consulta: {data.get('message')}")
                return False
        else:
            print(f"    Erro HTTP: {r.status_code}")
            return False
        
    except requests.exceptions.ConnectionError:
        print(" ERRO: Não consegui conectar ao backend")
        print("   Verifique se o servidor está rodando:")
        print("   python app.py")
        return False
    
    except requests.exceptions.Timeout:
        print(" ERRO: Timeout na conexão")
        print("   O servidor pode estar sobrecarregado")
        return False
    
    except Exception as e:
        print(f" ERRO inesperado: {e}")
        return False

def main():
    """Função principal"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(" Teste Ultra-Rápido RAG")
        print("="*30)
        print("Testa o funcionamento básico do sistema RAG em menos de 30s")
        print("\nUso:")
        print("  python quick_test.py")
        print("\nO que é testado:")
        print("   Conectividade com backend")
        print("   Status do sistema RAG")
        print("   Consulta básica")
        print("   Detecção de suspeitos")
        print("\nSe algo falhar, use os outros scripts para diagnóstico:")
        print("  python test_rag_simple.py    # Teste simples")
        print("  python test_rag.py auto      # Teste completo")
        return
    
    success = quick_test()
    
    if not success:
        print("\n PRÓXIMOS PASSOS:")
        print("1. Verifique se Ollama está rodando: ollama serve")
        print("2. Verifique se o backend está ativo: python app.py")
        print("3. Use teste detalhado: python test_rag.py")
        sys.exit(1)

if __name__ == "__main__":
    main()