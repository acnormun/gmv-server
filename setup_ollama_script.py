#!/usr/bin/env python3
"""
🚀 Script de Setup Automático do Ollama + Modelos
Instala e configura tudo automaticamente para usar múltiplas LLMs
"""

import os
import sys
import time
import platform
import subprocess
import requests
import json
from typing import Dict, List

class OllamaSetup:
    """Classe para setup automático do Ollama"""
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.system = platform.system().lower()
        
        # Modelos recomendados por categoria
        self.recommended_models = {
            "essencial": [
                {"name": "llama3.2:3b", "size": "3GB", "desc": "Rápido e eficiente"},
                {"name": "mistral:7b", "size": "4.1GB", "desc": "Qualidade superior"}
            ],
            "desenvolvimento": [
                {"name": "tinyllama:1.1b", "size": "637MB", "desc": "Ultra rápido para testes"},
                {"name": "codellama:7b", "size": "3.8GB", "desc": "Especializado em código"}
            ],
            "producao": [
                {"name": "llama3.1:8b", "size": "4.7GB", "desc": "Equilibrado para produção"},
                {"name": "gemma:7b", "size": "5.2GB", "desc": "Modelo do Google"}
            ]
        }
    
    def check_ollama_installed(self) -> bool:
        """Verifica se Ollama está instalado"""
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Ollama já instalado: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("❌ Ollama não encontrado")
        return False
    
    def install_ollama(self) -> bool:
        """Instala Ollama automaticamente"""
        print(f"🔽 Instalando Ollama para {self.system}...")
        
        try:
            if self.system == "linux":
                # Instalar no Linux
                cmd = "curl -fsSL https://ollama.ai/install.sh | sh"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Ollama instalado no Linux!")
                    return True
                else:
                    print(f"❌ Erro na instalação: {result.stderr}")
                    return False
            
            elif self.system == "darwin":  # macOS
                print("🍎 Para macOS:")
                print("   1. Visite: https://ollama.ai")
                print("   2. Baixe o instalador .pkg")
                print("   3. Execute o instalador")
                print("   4. Execute este script novamente")
                return False
            
            elif self.system == "windows":
                print("🪟 Para Windows:")
                print("   1. Visite: https://ollama.ai")
                print("   2. Baixe o instalador .exe")
                print("   3. Execute como administrador")
                print("   4. Execute este script novamente")
                return False
            
            else:
                print(f"❌ Sistema {self.system} não suportado")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante instalação: {e}")
            return False
    
    def check_ollama_running(self) -> bool:
        """Verifica se Ollama está rodando"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama está rodando!")
                return True
        except:
            pass
        
        print("❌ Ollama não está rodando")
        return False
    
    def start_ollama(self) -> bool:
        """Inicia o servidor Ollama"""
        print("🔄 Iniciando servidor Ollama...")
        
        try:
            if self.system == "windows":
                # No Windows, Ollama pode ser um serviço
                subprocess.Popen(["ollama", "serve"], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # Linux/macOS
                subprocess.Popen(["ollama", "serve"])
            
            # Aguarda alguns segundos para o servidor iniciar
            print("⏳ Aguardando servidor inicializar...")
            for i in range(10):
                time.sleep(1)
                if self.check_ollama_running():
                    return True
                print(f"   {i+1}/10...")
            
            print("⚠️ Servidor demorou para iniciar, mas pode estar funcionando")
            return False
            
        except Exception as e:
            print(f"❌ Erro ao iniciar servidor: {e}")
            return False
    
    def list_installed_models(self) -> List[str]:
        """Lista modelos já instalados"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
        except:
            pass
        return []
    
    def install_model(self, model_name: str) -> bool:
        """Instala um modelo específico"""
        print(f"📥 Baixando modelo: {model_name}")
        
        try:
            # Usa ollama CLI para download com progress
            result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {model_name} instalado com sucesso!")
                return True
            else:
                print(f"❌ Erro ao instalar {model_name}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Erro durante download: {e}")
            return False
    
    def show_model_menu(self) -> List[str]:
        """Mostra menu de seleção de modelos"""
        print("\n📋 SELEÇÃO DE MODELOS")
        print("=" * 40)
        
        all_models = []
        model_index = 1
        
        for category, models in self.recommended_models.items():
            print(f"\n🏷️ {category.upper()}:")
            for model in models:
                print(f"   {model_index}. {model['name']} ({model['size']}) - {model['desc']}")
                all_models.append(model['name'])
                model_index += 1
        
        print(f"\n   0. Todos os modelos essenciais")
        print(f"   99. Personalizado")
        
        while True:
            try:
                choice = input("\n🎯 Escolha os modelos (ex: 1,2,3 ou 0 para essenciais): ")
                
                if choice == "0":
                    return [model['name'] for model in self.recommended_models['essencial']]
                
                elif choice == "99":
                    custom = input("Digite os nomes dos modelos (separados por vírgula): ")
                    return [m.strip() for m in custom.split(',') if m.strip()]
                
                else:
                    indices = [int(x.strip()) for x in choice.split(',')]
                    selected = []
                    for idx in indices:
                        if 1 <= idx <= len(all_models):
                            selected.append(all_models[idx-1])
                    
                    if selected:
                        return selected
                    else:
                        print("❌ Seleção inválida")
                        
            except (ValueError, KeyboardInterrupt):
                print("❌ Entrada inválida")
    
    def install_selected_models(self, models: List[str]):
        """Instala modelos selecionados"""
        installed = self.list_installed_models()
        
        for model in models:
            if model in installed:
                print(f"✅ {model} já instalado")
            else:
                self.install_model(model)
    
    def test_installation(self):
        """Testa a instalação"""
        print("\n🧪 TESTANDO INSTALAÇÃO")
        print("=" * 30)
        
        models = self.list_installed_models()
        
        if not models:
            print("❌ Nenhum modelo encontrado")
            return False
        
        print(f"✅ {len(models)} modelo(s) disponível(is):")
        for model in models:
            print(f"   - {model}")
        
        # Teste simples
        test_model = models[0]
        print(f"\n🔄 Testando {test_model}...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": test_model,
                    "prompt": "Diga olá em português",
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '')
                print(f"✅ Teste bem-sucedido!")
                print(f"   Resposta: {result.strip()}")
                return True
            else:
                print(f"❌ Erro no teste: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erro no teste: {e}")
            return False
    
    def create_usage_guide(self):
        """Cria guia de uso"""
        guide_content = """# 🚀 Guia de Uso - Ollama + RAG

## 📋 Modelos Instalados
"""
        
        models = self.list_installed_models()
        for model in models:
            guide_content += f"- `{model}`\n"
        
        guide_content += """
## 🐍 Exemplo de Uso Python

```python
from adaptive_rag_ollama import GMVAdaptiveRAGOllama

# Inicializar
rag = GMVAdaptiveRAGOllama()
rag.initialize("triagem.md", "pasta_destino")

# Listar modelos
print(rag.list_available_models())

# Definir modelo
rag.set_llm_model("llama3.2:3b")

# Fazer pergunta
result = rag.query("Sua pergunta aqui")
print(result.response)
```

## 🔄 Comandos Úteis

```bash
# Listar modelos
ollama list

# Baixar novo modelo
ollama pull modelo_nome

# Iniciar servidor
ollama serve

# Testar modelo
ollama run llama3.2:3b "Olá!"
```

## 📊 Benchmark
Execute: `python test_models_script.py`
"""
        
        try:
            with open("OLLAMA_GUIDE.md", "w", encoding="utf-8") as f:
                f.write(guide_content)
            print(f"\n📄 Guia criado: OLLAMA_GUIDE.md")
        except Exception as e:
            print(f"⚠️ Erro ao criar guia: {e}")
    
    def run_setup(self):
        """Executa setup completo"""
        print("🚀 SETUP AUTOMÁTICO OLLAMA + RAG")
        print("=" * 40)
        
        # 1. Verificar se está instalado
        if not self.check_ollama_installed():
            if not self.install_ollama():
                print("❌ Falha na instalação. Setup manual necessário.")
                return False
        
        # 2. Verificar se está rodando
        if not self.check_ollama_running():
            if not self.start_ollama():
                print("⚠️ Tente iniciar manualmente: ollama serve")
                print("⏳ Aguarde 10 segundos e execute este script novamente")
                return False
        
        # 3. Selecionar e instalar modelos
        print(f"\n🎯 Modelos atuais: {len(self.list_installed_models())}")
        
        if input("\n📥 Instalar novos modelos? (s/N): ").lower().startswith('s'):
            selected_models = self.show_model_menu()
            if selected_models:
                print(f"\n📦 Instalando {len(selected_models)} modelo(s)...")
                self.install_selected_models(selected_models)
        
        # 4. Testar instalação
        if self.test_installation():
            print("\n🎉 SETUP CONCLUÍDO COM SUCESSO!")
            
            # 5. Criar guia
            self.create_usage_guide()
            
            print("\n🎯 Próximos passos:")
            print("   1. Execute: python test_models_script.py")
            print("   2. Configure seu RAG com o melhor modelo")
            print("   3. Leia o guia: OLLAMA_GUIDE.md")
            
            return True
        else:
            print("\n❌ Setup incompleto. Verifique os erros acima.")
            return False

def main():
    """Função principal"""
    try:
        setup = OllamaSetup()
        setup.run_setup()
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup cancelado pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")

if __name__ == "__main__":
    main()