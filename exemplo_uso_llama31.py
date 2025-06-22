#!/usr/bin/env python3
"""
🦙 Exemplo Simples: RAG Adaptativo + Llama 3.1:8B
Demonstra como usar o sistema completo
"""

from adaptive_rag_llama31 import GMVAdaptiveRAGLlama31, initialize_rag, query_rag, get_rag_statistics

def test_llama_connection():
    """Testa se Llama 3.1:8B está funcionando"""
    print("🔌 TESTE DE CONEXÃO")
    print("-" * 20)
    
    try:
        rag = GMVAdaptiveRAGLlama31()
        results = rag.test_llama_connection()
        
        success_count = 0
        for query, result in results.items():
            if result.get("success"):
                print(f"✅ {result['response']} ({result['time']:.2f}s)")
                success_count += 1
            else:
                print(f"❌ Erro: {result.get('error', 'Falhou')}")
        
        if success_count == len(results):
            print("🎉 Llama 3.1:8B está funcionando perfeitamente!")
            return True
        else:
            print(f"⚠️ {success_count}/{len(results)} testes passaram")
            return False
            
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")
        return False

def demo_without_data():
    """Demo sem carregar dados (teste básico)"""
    print("\n🧪 DEMO SEM DADOS")
    print("-" * 20)
    
    try:
        rag = GMVAdaptiveRAGLlama31()
        
        # Perguntas gerais sobre o domínio
        demo_queries = [
            "O que é lavagem de dinheiro?",
            "Explique fraude financeira",
            "Como funciona um esquema de corrupção?",
            "Quais são os indicadores de suspeita?"
        ]
        
        for query in demo_queries:
            print(f"\n📝 {query}")
            try:
                # Gera resposta simples (sem RAG)
                response = rag.llm_client.generate(
                    prompt=query,
                    system_prompt="Responda de forma clara e educativa em português brasileiro sobre crimes financeiros."
                )
                print(f"💬 {response[:200]}...")
                
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no demo: {e}")
        return False

def demo_with_data():
    """Demo completo com dados"""
    print("\n📊 DEMO COM DADOS")
    print("-" * 20)
    
    # Caminhos dos dados (AJUSTE CONFORME SUA ESTRUTURA)
    data_paths = {
        "triagem": "data/triagem.md",           # Arquivo de triagem
        "processos": "data/processos/",         # Pasta com processos .md
        "dat": "data/dat/"                      # Pasta com arquivos .dat
    }
    
    print("📁 Tentando carregar dados...")
    for name, path in data_paths.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: {path}")
        else:
            print(f"   ❌ {name}: {path} (não encontrado)")
    
    try:
        # Inicializar RAG com dados
        success = initialize_rag(
            triagem_path=data_paths["triagem"],
            pasta_destino=data_paths["processos"],
            pasta_dat=data_paths["dat"]
        )
        
        if not success:
            print("⚠️ Falha ao carregar dados - verifique os caminhos")
            print("💡 Para testar sem dados, use demo_without_data()")
            return False
        
        # Mostrar estatísticas
        stats = get_rag_statistics()
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   📄 Documentos: {stats['total_documents']}")
        print(f"   🧩 Chunks: {stats['total_chunks']}")
        print(f"   🧠 Modelo: {stats['llm_model']}")
        print(f"   🔗 Embeddings: {stats['embedding_model']}")
        
        # Consultas de exemplo
        rag_queries = [
            "Quais processos envolvem lavagem de dinheiro?",
            "Quantos processos estão em investigação?",
            "Compare os tipos de crimes mais frequentes",
            "Explique o contexto dos processos suspeitos",
            "Qual a distribuição por status dos processos?"
        ]
        
        print(f"\n🔍 TESTANDO {len(rag_queries)} CONSULTAS:")
        
        for i, query in enumerate(rag_queries, 1):
            print(f"\n{i}. {query}")
            
            try:
                result = query_rag(query)
                
                if "error" in result:
                    print(f"   ❌ Erro: {result['error']}")
                else:
                    print(f"   🎯 Estratégia: {result['strategy_used']}")
                    print(f"   📊 Confiança: {result['confidence_score']:.3f}")
                    print(f"   ⏱️ Tempo: {result['processing_time']:.2f}s")
                    print(f"   📝 Resposta: {result['response'][:150]}...")
                
            except Exception as e:
                print(f"   ❌ Erro na consulta: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no demo: {e}")
        return False

def interactive_mode():
    """Modo interativo para fazer perguntas"""
    print("\n💬 MODO INTERATIVO")
    print("-" * 20)
    print("Digite suas perguntas (ou 'sair' para finalizar)")
    
    while True:
        try:
            query = input("\n🔍 Sua pergunta: ").strip()
            
            if query.lower() in ['sair', 'exit', 'quit']:
                print("👋 Até logo!")
                break
            
            if not query:
                continue
            
            print("⏳ Processando...")
            result = query_rag(query)
            
            if "error" in result:
                print(f"❌ Erro: {result['error']}")
            else:
                print(f"\n🎯 Estratégia: {result['strategy_used']}")
                print(f"📊 Confiança: {result['confidence_score']:.3f}")
                print(f"⏱️ Tempo: {result['processing_time']:.2f}s")
                print(f"\n💬 Resposta:")
                print(result['response'])
        
        except KeyboardInterrupt:
            print("\n👋 Saindo...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

def main():
    """Função principal"""
    print("🦙 RAG ADAPTATIVO + LLAMA 3.1:8B")
    print("=" * 40)
    
    # 1. Testar conexão
    if not test_llama_connection():
        print("\n❌ Llama 3.1:8B não está funcionando")
        print("💡 Execute primeiro: python setup_llama31_script.py")
        return
    
    # 2. Menu de opções
    while True:
        print("\n📋 OPÇÕES:")
        print("1. Demo básico (sem dados)")
        print("2. Demo completo (com dados)")
        print("3. Modo interativo")
        print("4. Sair")
        
        try:
            choice = input("\n🎯 Escolha uma opção (1-4): ").strip()
            
            if choice == "1":
                demo_without_data()
            
            elif choice == "2":
                demo_with_data()
            
            elif choice == "3":
                # Tenta carregar dados primeiro
                try:
                    initialize_rag("data/triagem.md", "data/processos/", "data/dat/")
                    print("✅ Dados carregados para modo interativo")
                except:
                    print("⚠️ Dados não carregados - respostas serão gerais")
                
                interactive_mode()
            
            elif choice == "4":
                print("👋 Até logo!")
                break
            
            else:
                print("❌ Opção inválida")
        
        except KeyboardInterrupt:
            print("\n👋 Saindo...")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    import os
    main()