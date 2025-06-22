#!/usr/bin/env python3
"""
🦙 Setup Rápido para Llama 3.1:8B + RAG Adaptativo
Configura tudo automaticamente para usar Llama 3.1:8B
"""

import os
import sys
import time
import subprocess
import requests
import json
from datetime import datetime

class Llama31Setup:
    """Setup automático para Llama 3.1:8B"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model_name = "llama3.1:8b"
        self.model_size = "4.7GB"
        
    def print_header(self):
        """Mostra header do setup"""
        print("🦙 SETUP LLAMA 3.1:8B + RAG ADAPTATIVO")
        print("=" * 45)
        print(f"Modelo: {self.model_name}")
        print(f"Tamanho: {self.model_size}")
        print(f"Qualidade: ⭐⭐⭐⭐ (Equilibrado)")
        print(f"Velocidade: ⚡⚡⚡ (Boa)")
        print("=" * 45)
    
    def check_python_deps(self) -> bool:
        """Verifica dependências Python"""
        print("\n📦 Verificando dependências Python...")
        
        required = [
            "requests", "numpy", "pandas", "sklearn", 
            "sentence_transformers", "torch"
        ]
        
        missing = []
        for pkg in required:
            try:
                __import__(pkg.replace("-", "_"))
                print(f"   ✅ {pkg}")
            except ImportError:
                missing.append(pkg)
                print(f"   ❌ {pkg}")
        
        if missing:
            print(f"\n📥 Instalando {len(missing)} dependência(s)...")
            try:
                cmd = [sys.executable, "-m", "pip", "install"] + missing
                subprocess.run(cmd, check=True)
                print("✅ Dependências instaladas!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Erro na instalação. Tente manualmente:")
                print(f"   pip install {' '.join(missing)}")
                return False
        
        print("✅ Todas as dependências estão instaladas!")
        return True
    
    def check_ollama_installed(self) -> bool:
        """Verifica se Ollama está instalado"""
        print("\n🔍 Verificando Ollama...")
        
        try:
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Ollama instalado: {version}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        print("❌ Ollama não encontrado")
        return False
    
    def install_ollama_instructions(self):
        """Mostra instruções para instalar Ollama"""
        print("\n📥 COMO INSTALAR OLLAMA:")
        print("-" * 30)
        print("1. Visite: https://ollama.ai")
        print("2. Baixe o instalador para seu sistema")
        print("3. Execute o instalador")
        print("4. Reinicie o terminal")
        print("5. Execute este script novamente")
        print("\n💡 Comandos alternativos:")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   macOS: brew install ollama")
        print("   Windows: Baixe o .exe do site")
    
    def check_ollama_running(self) -> bool:
        """Verifica se Ollama está rodando"""
        print("\n🔌 Verificando servidor Ollama...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama está rodando!")
                return True
        except requests.RequestException:
            pass
        
        print("❌ Servidor Ollama não está rodando")
        return False
    
    def start_ollama(self) -> bool:
        """Tenta iniciar Ollama"""
        print("\n🚀 Iniciando servidor Ollama...")
        
        try:
            # Tenta iniciar em background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Aguarda alguns segundos
            print("⏳ Aguardando servidor inicializar...")
            for i in range(15):
                time.sleep(1)
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("✅ Servidor iniciado com sucesso!")
                        return True
                except:
                    pass
                print(f"   {i+1}/15...")
            
            print("⚠️ Servidor demorou para iniciar")
            print("💡 Tente manualmente: ollama serve")
            return False
            
        except Exception as e:
            print(f"❌ Erro ao iniciar: {e}")
            return False
    
    def check_model_installed(self) -> bool:
        """Verifica se Llama 3.1:8B está instalado"""
        print(f"\n🔍 Verificando {self.model_name}...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name in model_names:
                    print(f"✅ {self.model_name} já instalado!")
                    return True
                
                if model_names:
                    print(f"📋 Modelos disponíveis: {', '.join(model_names)}")
        except Exception as e:
            print(f"❌ Erro ao verificar modelos: {e}")
        
        print(f"❌ {self.model_name} não encontrado")
        return False
    
    def install_llama31(self) -> bool:
        """Instala Llama 3.1:8B"""
        print(f"\n📥 Baixando {self.model_name} ({self.model_size})...")
        print("⏳ Isso pode demorar alguns minutos dependendo da sua conexão...")
        
        try:
            # Usa ollama CLI para mostrar progresso
            process = subprocess.Popen(
                ["ollama", "pull", self.model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            # Mostra progresso
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Filtra linhas de progresso
                    if any(word in output.lower() for word in ['pulling', 'downloading', 'verifying']):
                        print(f"   {output.strip()}")
            
            if process.returncode == 0:
                print(f"✅ {self.model_name} instalado com sucesso!")
                return True
            else:
                print(f"❌ Erro no download")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante instalação: {e}")
            return False
    
    def test_llama31(self) -> bool:
        """Testa Llama 3.1:8B"""
        print(f"\n🧪 Testando {self.model_name}...")
        
        test_prompt = "Responda apenas: 'Llama 3.1 funcionando!' em português"
        
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
            
            print("⏳ Gerando resposta de teste...")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                processing_time = end_time - start_time
                
                print(f"✅ Teste bem-sucedido!")
                print(f"   Resposta: {result}")
                print(f"   Tempo: {processing_time:.2f}s")
                
                if "llama" in result.lower() or "funcionando" in result.lower():
                    return True
                else:
                    print("⚠️ Resposta inesperada, mas modelo funciona")
                    return True
            else:
                print(f"❌ Erro HTTP: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro no teste: {e}")
            return False
    
    def create_example_script(self):
        """Cria script de exemplo"""
        example_code = '''#!/usr/bin/env python3
"""
🦙 Exemplo de uso do RAG Adaptativo com Llama 3.1:8B
"""

from adaptive_rag_llama31 import GMVAdaptiveRAGLlama31

def main():
    print("🦙 EXEMPLO RAG + LLAMA 3.1:8B")
    print("=" * 35)
    
    # Inicializar RAG
    rag = GMVAdaptiveRAGLlama31()
    
    # Testar conexão com Llama
    print("\\n🔌 Testando Llama 3.1:8B...")
    test_results = rag.test_llama_connection()
    
    for query, result in test_results.items():
        if result.get("success"):
            print(f"   ✅ {query}: {result['response']}")
        else:
            print(f"   ❌ {query}: {result.get('error', 'Falhou')}")
    
    # Carregar dados (ajuste os caminhos)
    print("\\n📁 Carregando dados...")
    try:
        success = rag.initialize(
            triagem_path="data/triagem.md",
            pasta_destino="data/processos/",
            pasta_dat="data/dat/"
        )
        
        if success:
            print("✅ Dados carregados!")
            
            # Estatísticas
            stats = rag.get_statistics()
            print(f"📊 {stats['total_documents']} documentos, {stats['total_chunks']} chunks")
            
            # Exemplo de consultas
            test_queries = [
                "Quais processos envolvem lavagem de dinheiro?",
                "Quantos processos estão em investigação?",
                "Compare os tipos de crimes mais frequentes",
                "Explique o contexto dos processos suspeitos"
            ]
            
            print("\\n🔍 Testando consultas...")
            for query in test_queries:
                print(f"\\n📝 {query}")
                try:
                    result = rag.query(query)
                    print(f"   🎯 Estratégia: {result.strategy_used}")
                    print(f"   📊 Confiança: {result.confidence_score:.3f}")
                    print(f"   ⏱️ Tempo: {result.processing_time:.2f}s")
                    print(f"   💬 Resposta: {result.response[:200]}...")
                except Exception as e:
                    print(f"   ❌ Erro: {e}")
        
        else:
            print("⚠️ Falha ao carregar dados - verifique os caminhos")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("💡 Ajuste os caminhos dos arquivos no código")

if __name__ == "__main__":
    main()
'''
        
        try:
            with open("exemplo_llama31.py", "w", encoding="utf-8") as f:
                f.write(example_code)
            print("📄 Exemplo criado: exemplo_llama31.py")
        except Exception as e:
            print(f"⚠️ Erro ao criar exemplo: {e}")
    
    def show_next_steps(self):
        """Mostra próximos passos"""
        print("\n🎯 PRÓXIMOS PASSOS")
        print("=" * 20)
        print("1. Execute o exemplo: python exemplo_llama31.py")
        print("2. Ajuste os caminhos dos seus dados")
        print("3. Faça suas próprias consultas")
        print("4. Monitore performance e ajuste conforme necessário")
        print("\n💡 DICAS:")
        print("- Llama 3.1:8B é equilibrado (qualidade + velocidade)")
        print("- Ideal para produção com hardware moderno")
        print("- Funciona bem em português brasileiro")
        print("- Use queries específicas para melhores resultados")
    
    def run_setup(self):
        """Executa setup completo"""
        self.print_header()
        
        # 1. Verificar Python
        if not self.check_python_deps():
            print("❌ Setup abortado - dependências Python")
            return False
        
        # 2. Verificar Ollama
        if not self.check_ollama_installed():
            self.install_ollama_instructions()
            return False
        
        # 3. Verificar servidor
        if not self.check_ollama_running():
            if not self.start_ollama():
                print("❌ Não foi possível iniciar Ollama")
                print("💡 Tente manualmente: ollama serve")
                return False
        
        # 4. Verificar modelo
        if not self.check_model_installed():
            if not self.install_llama31():
                print("❌ Falha ao instalar Llama 3.1:8B")
                return False
        
        # 5. Testar modelo
        if not self.test_llama31():
            print("❌ Llama 3.1:8B não está funcionando")
            return False
        
        # 6. Criar exemplo
        self.create_example_script()
        
        # 7. Finalizar
        print("\n🎉 SETUP CONCLUÍDO!")
        print("✅ Llama 3.1:8B pronto para uso!")
        
        self.show_next_steps()
        
        return True

def main():
    """Função principal"""
    try:
        setup = Llama31Setup()
        setup.run_setup()
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup cancelado")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()