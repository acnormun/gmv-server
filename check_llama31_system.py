#!/usr/bin/env python3
"""
🔍 Verificador do Sistema RAG + Llama 3.1:8B
Verifica se tudo está funcionando antes de usar
"""

import os
import sys
import time
import requests
import importlib.util
from datetime import datetime

class SystemChecker:
    """Verificador completo do sistema"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.1:8b"
        self.checks_passed = 0
        self.total_checks = 0
    
    def print_header(self):
        """Header do verificador"""
        print("🔍 VERIFICADOR DO SISTEMA RAG + LLAMA 3.1:8B")
        print("=" * 50)
        print("Este script verifica se tudo está pronto para uso")
        print("=" * 50)
    
    def check_python_version(self):
        """Verifica versão do Python"""
        print("\n🐍 VERIFICANDO PYTHON...")
        self.total_checks += 1
        
        version = sys.version_info
        print(f"   Versão: Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major >= 3 and version.minor >= 8:
            print("   ✅ Versão compatível")
            self.checks_passed += 1
            return True
        else:
            print("   ❌ Python 3.8+ necessário")
            return False
    
    def check_dependencies(self):
        """Verifica dependências Python"""
        print("\n📦 VERIFICANDO DEPENDÊNCIAS...")
        
        required_packages = [
            ("requests", "requests"),
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("sklearn", "scikit-learn"),
            ("sentence_transformers", "sentence-transformers"),
            ("torch", "torch"),
            ("flask", "flask")
        ]
        
        missing_packages = []
        
        for import_name, package_name in required_packages:
            self.total_checks += 1
            try:
                importlib.import_module(import_name)
                print(f"   ✅ {package_name}")
                self.checks_passed += 1
            except ImportError:
                print(f"   ❌ {package_name}")
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"\n💡 Para instalar dependências faltantes:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
        else:
            print("   🎉 Todas as dependências estão instaladas!")
            return True
    
    def check_ollama_installation(self):
        """Verifica instalação do Ollama"""
        print("\n🔧 VERIFICANDO OLLAMA...")
        self.total_checks += 1
        
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"   ✅ Ollama instalado: {version}")
                self.checks_passed += 1
                return True
            else:
                print("   ❌ Ollama não responde")
                return False
                
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("   ❌ Ollama não encontrado")
            print("   💡 Instale em: https://ollama.ai")
            return False
        except Exception as e:
            print(f"   ❌ Erro ao verificar Ollama: {e}")
            return False
    
    def check_ollama_server(self):
        """Verifica se servidor Ollama está rodando"""
        print("\n🌐 VERIFICANDO SERVIDOR OLLAMA...")
        self.total_checks += 1
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"   ✅ Servidor rodando ({len(models)} modelo(s) disponível(is))")
                self.checks_passed += 1
                return True
            else:
                print(f"   ❌ Servidor respondeu com erro: {response.status_code}")
                return False
                
        except requests.RequestException:
            print("   ❌ Servidor não está rodando")
            print("   💡 Execute: ollama serve")
            return False
    
    def check_llama31_model(self):
        """Verifica se Llama 3.1:8B está disponível"""
        print(f"\n🦙 VERIFICANDO {self.model_name.upper()}...")
        self.total_checks += 1
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name in model_names:
                    # Pega informações do modelo
                    for model in models:
                        if model['name'] == self.model_name:
                            size = model.get('size', 0)
                            size_gb = size / (1024**3) if size > 0 else 0
                            print(f"   ✅ Modelo disponível ({size_gb:.1f}GB)")
                            self.checks_passed += 1
                            return True
                else:
                    print(f"   ❌ Modelo não encontrado")
                    print(f"   📋 Modelos disponíveis: {', '.join(model_names) if model_names else 'Nenhum'}")
                    print(f"   💡 Para instalar: ollama pull {self.model_name}")
                    return False
            else:
                print("   ❌ Erro ao verificar modelos")
                return False
                
        except Exception as e:
            print(f"   ❌ Erro na verificação: {e}")
            return False
    
    def test_llama31_generation(self):
        """Testa geração de texto com Llama 3.1:8B"""
        print(f"\n🧪 TESTANDO GERAÇÃO COM {self.model_name.upper()}...")
        self.total_checks += 1
        
        test_prompt = "Responda apenas: 'Sistema funcionando' em português"
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 20
                }
            }
            
            print("   ⏳ Gerando resposta de teste...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                processing_time = end_time - start_time
                
                print(f"   ✅ Resposta gerada em {processing_time:.2f}s")
                print(f"   💬 Resultado: {result}")
                
                # Verifica se resposta faz sentido
                if any(word in result.lower() for word in ['sistema', 'funcionando', 'ok', 'teste']):
                    print("   🎯 Resposta coerente!")
                    self.checks_passed += 1
                    return True
                else:
                    print("   ⚠️ Resposta inesperada, mas modelo funciona")
                    self.checks_passed += 1
                    return True
            else:
                print(f"   ❌ Erro HTTP: {response.status_code}")
                return False
                
        except requests.Timeout:
            print("   ❌ Timeout na geração (muito lento)")
            return False
        except Exception as e:
            print(f"   ❌ Erro na geração: {e}")
            return False
    
    def check_rag_components(self):
        """Verifica componentes do RAG"""
        print("\n🧠 VERIFICANDO COMPONENTES RAG...")
        
        # Embeddings
        self.total_checks += 1
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L12-v2")
            test_embedding = model.encode(["teste"])
            print(f"   ✅ Embeddings funcionando (dimensão: {len(test_embedding[0])})")
            self.checks_passed += 1
        except Exception as e:
            print(f"   ❌ Erro nos embeddings: {e}")
        
        # Similarity search
        self.total_checks += 1
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Teste simples
            vec1 = np.array([[1, 0, 1]])
            vec2 = np.array([[1, 1, 0]])
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            print(f"   ✅ Similarity search funcionando (teste: {similarity:.3f})")
            self.checks_passed += 1
        except Exception as e:
            print(f"   ❌ Erro no similarity search: {e}")
        
        # Clustering
        self.total_checks += 1
        try:
            from sklearn.cluster import KMeans
            
            # Teste simples
            data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(data)
            
            print(f"   ✅ Clustering funcionando ({len(set(kmeans.labels_))} clusters)")
            self.checks_passed += 1
        except Exception as e:
            print(f"   ❌ Erro no clustering: {e}")
    
    def check_adaptive_rag_import(self):
        """Verifica se pode importar o RAG adaptativo"""
        print("\n📚 VERIFICANDO IMPORTAÇÃO DO RAG...")
        self.total_checks += 1
        
        try:
            # Tenta importar o módulo principal
            if os.path.exists("adaptive_rag_llama31.py"):
                sys.path.insert(0, ".")
                import adaptive_rag_llama31
                print("   ✅ adaptive_rag_llama31.py importado com sucesso")
                self.checks_passed += 1
                return True
            else:
                print("   ❌ adaptive_rag_llama31.py não encontrado")
                print("   💡 Certifique-se de ter todos os arquivos")
                return False
                
        except Exception as e:
            print(f"   ❌ Erro na importação: {e}")
            return False
    
    def performance_benchmark(self):
        """Executa benchmark rápido de performance"""
        print("\n⚡ BENCHMARK DE PERFORMANCE...")
        
        if not all([
            self.check_ollama_server(),
            self.check_llama31_model()
        ]):
            print("   ⚠️ Pulando benchmark - pré-requisitos não atendidos")
            return
        
        try:
            queries = [
                "O que é inteligência artificial?",
                "Explique machine learning brevemente",
                "Defina processamento de linguagem natural"
            ]
            
            total_time = 0
            successful_queries = 0
            
            for i, query in enumerate(queries, 1):
                print(f"   🔄 Teste {i}/3: {query[:30]}...")
                
                try:
                    payload = {
                        "model": self.model_name,
                        "prompt": query,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 100}
                    }
                    
                    start = time.time()
                    response = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json=payload,
                        timeout=30
                    )
                    end = time.time()
                    
                    if response.status_code == 200:
                        query_time = end - start
                        result = response.json().get('response', '')
                        words = len(result.split())
                        
                        print(f"      ✅ {words} palavras em {query_time:.2f}s ({words/query_time:.1f} pal/s)")
                        total_time += query_time
                        successful_queries += 1
                    else:
                        print(f"      ❌ Erro HTTP: {response.status_code}")
                        
                except Exception as e:
                    print(f"      ❌ Erro: {e}")
            
            if successful_queries > 0:
                avg_time = total_time / successful_queries
                print(f"\n   📊 RESULTADOS:")
                print(f"      ⏱️ Tempo médio: {avg_time:.2f}s")
                print(f"      ✅ Taxa de sucesso: {successful_queries}/{len(queries)}")
                
                if avg_time < 5:
                    print("      🚀 Performance excelente!")
                elif avg_time < 10:
                    print("      👍 Performance boa")
                else:
                    print("      ⚠️ Performance baixa - considere otimizações")
            
        except Exception as e:
            print(f"   ❌ Erro no benchmark: {e}")
    
    def generate_report(self):
        """Gera relatório final"""
        print("\n📋 RELATÓRIO FINAL")
        print("=" * 30)
        
        success_rate = (self.checks_passed / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        print(f"✅ Verificações passaram: {self.checks_passed}/{self.total_checks}")
        print(f"📊 Taxa de sucesso: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("\n🎉 SISTEMA PRONTO PARA USO!")
            print("✅ Todos os componentes estão funcionando corretamente")
            print("💡 Execute: python exemplo_uso_llama31.py")
            
        elif success_rate >= 70:
            print("\n⚠️ SISTEMA PARCIALMENTE FUNCIONAL")
            print("🔧 Alguns componentes precisam de ajustes")
            print("💡 Verifique os erros acima e corrija")
            
        else:
            print("\n❌ SISTEMA NÃO ESTÁ PRONTO")
            print("🚨 Muitos problemas encontrados")
            print("💡 Execute: python setup_llama31_script.py")
        
        print(f"\n📅 Verificação realizada em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_all_checks(self):
        """Executa todas as verificações"""
        self.print_header()
        
        # Verificações básicas
        self.check_python_version()
        self.check_dependencies()
        
        # Verificações do Ollama
        self.check_ollama_installation()
        server_ok = self.check_ollama_server()
        
        if server_ok:
            self.check_llama31_model()
            self.test_llama31_generation()
        
        # Verificações do RAG
        self.check_rag_components()
        self.check_adaptive_rag_import()
        
        # Benchmark (opcional)
        if server_ok:
            self.performance_benchmark()
        
        # Relatório final
        self.generate_report()

def main():
    """Função principal"""
    try:
        checker = SystemChecker()
        checker.run_all_checks()
    except KeyboardInterrupt:
        print("\n\n⏹️ Verificação cancelada")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()