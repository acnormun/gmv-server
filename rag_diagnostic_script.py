#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Diagnóstico do Sistema RAG
Execute este script para identificar problemas no carregamento de documentos
"""

import os
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print(f"{'='*60}")

def print_section(title):
    print(f"\n{'─'*40}")
    print(f"📋 {title}")
    print(f"{'─'*40}")

def check_imports():
    """Verifica todas as importações necessárias"""
    print_section("VERIFICANDO IMPORTAÇÕES")
    
    results = {}
    
    # Langchain básico
    try:
        from langchain.schema import Document
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("✅ Langchain básico: OK")
        results['langchain_basic'] = True
    except Exception as e:
        print(f"❌ Langchain básico: {e}")
        results['langchain_basic'] = False
    
    # Ollama
    try:
        from langchain_ollama import OllamaLLM, OllamaEmbeddings
        print("✅ Langchain Ollama: OK")
        results['ollama_import'] = True
    except ImportError:
        try:
            from langchain_community.llms import Ollama as OllamaLLM
            from langchain_community.embeddings import OllamaEmbeddings
            print("✅ Langchain Community Ollama: OK")
            results['ollama_import'] = True
        except ImportError:
            try:
                from langchain.llms import Ollama as OllamaLLM
                from langchain.embeddings import OllamaEmbeddings
                print("✅ Langchain Legacy Ollama: OK")
                results['ollama_import'] = True
            except ImportError as e:
                print(f"❌ Todas as importações Ollama falharam: {e}")
                results['ollama_import'] = False
    
    # Utils personalizados
    try:
        from utils.optimized_vector_store import OptimizedVectorStore
        print("✅ OptimizedVectorStore: OK")
        results['vector_store'] = True
    except Exception as e:
        print(f"❌ OptimizedVectorStore: {e}")
        results['vector_store'] = False
    
    try:
        from utils.ultrafast_rag import UltraFastRAG, UltraFastRAGConfig
        print("✅ UltraFastRAG: OK")
        results['ultrafast_rag'] = True
    except Exception as e:
        print(f"❌ UltraFastRAG: {e}")
        results['ultrafast_rag'] = False
    
    # Adaptive RAG
    try:
        import adaptive_rag
        print("✅ adaptive_rag: OK")
        results['adaptive_rag'] = True
    except Exception as e:
        print(f"❌ adaptive_rag: {e}")
        results['adaptive_rag'] = False
    
    return results

def check_ollama_service():
    """Verifica se o Ollama está rodando"""
    print_section("VERIFICANDO SERVIÇO OLLAMA")
    
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Ollama está rodando")
            models = result.stdout.strip().split('\n')
            print(f"📦 Modelos disponíveis: {len(models)-1}")  # -1 para header
            for model in models[1:]:  # Skip header
                print(f"   - {model.split()[0]}")
            return True
        else:
            print(f"❌ Ollama não está rodando: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Timeout ao verificar Ollama")
        return False
    except FileNotFoundError:
        print("❌ Comando 'ollama' não encontrado")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar Ollama: {e}")
        return False

def check_data_directory():
    """Verifica a pasta de dados com foco em PASTA_DESTINO"""
    print_section("VERIFICANDO PASTA DE DADOS")
    
    # 1. Verificar PASTA_DESTINO especificamente
    pasta_destino = os.getenv("PASTA_DESTINO")
    print(f"🔍 PASTA_DESTINO definida: {pasta_destino}")
    
    if not pasta_destino:
        print("❌ Variável PASTA_DESTINO não está definida!")
        print("💡 Como definir:")
        print("   export PASTA_DESTINO=/caminho/para/seus/dados")
        print("   # Ou para pasta atual:")
        print("   export PASTA_DESTINO=$(pwd)/data")
        
        # Verificar alternativas
        possible_paths = ["data", "./data", "../data"]
        print(f"\n🔍 Verificando alternativas:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Alternativa encontrada: {path}")
                stats = analyze_files_in_directory(path)
                return path, stats
        
        return None, {}
    
    # 2. Verificar se PASTA_DESTINO existe
    if not os.path.exists(pasta_destino):
        print(f"❌ Pasta PASTA_DESTINO não existe: {pasta_destino}")
        print(f"💡 Para criar: mkdir -p {pasta_destino}")
        return None, {}
    
    print(f"✅ PASTA_DESTINO existe: {pasta_destino}")
    
    # 3. Verificar permissões
    if not os.access(pasta_destino, os.R_OK):
        print(f"❌ Sem permissão de leitura em: {pasta_destino}")
        return None, {}
    
    print(f"✅ Permissões de leitura: OK")
    
    # 4. Análise detalhada
    stats = analyze_files_in_directory(pasta_destino)
    return pasta_destino, stats

def analyze_files_in_directory(data_path):
    """Analisa arquivos na pasta de dados com critérios detalhados"""
    print(f"\n📊 ANÁLISE DETALHADA: {data_path}")
    
    stats = {
        'total_files': 0,
        'txt_md_files': 0,
        'valid_files': 0,
        'large_files': 0,
        'small_files': 0,
        'empty_files': 0,
        'encoding_errors': 0,
        'directories': 0,
        'sample_files': [],
        'problem_files': []
    }
    
    # Pastas que o RAG ignora (baseado no código original)
    ignored_dirs = {'.rag_cache', 'anonimizados', 'dat', 'mapas', '__pycache__', '.git'}
    
    print(f"🚫 Pastas ignoradas pelo RAG: {', '.join(ignored_dirs)}")
    
    for root, dirs, files in os.walk(data_path):
        # Mostrar quais pastas estão sendo ignoradas
        original_dirs = dirs.copy()
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        ignored_here = set(original_dirs) - set(dirs)
        if ignored_here:
            rel_path = os.path.relpath(root, data_path)
            print(f"   🚫 Ignorando em {rel_path}: {', '.join(ignored_here)}")
        
        stats['directories'] += len(dirs)
        rel_path = os.path.relpath(root, data_path)
        
        if rel_path != "." and len(files) > 0:
            md_txt_count = len([f for f in files if f.lower().endswith(('.txt', '.md'))])
            print(f"   📂 {rel_path}: {len(files)} arquivos ({md_txt_count} .md/.txt)")
        
        for file in files:
            stats['total_files'] += 1
            
            if file.lower().endswith(('.txt', '.md')):
                stats['txt_md_files'] += 1
                filepath = os.path.join(root, file)
                
                try:
                    # Verificar tamanho do arquivo
                    file_size = os.path.getsize(filepath)
                    
                    # Critérios do RAG original
                    if file_size < 100:
                        stats['small_files'] += 1
                        stats['problem_files'].append({
                            'file': file,
                            'path': rel_path,
                            'problem': f'Muito pequeno ({file_size} bytes < 100)',
                            'severity': 'high'
                        })
                        continue
                    else:
                        stats['large_files'] += 1
                    
                    # Tentar ler com encoding UTF-8
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        stats['encoding_errors'] += 1
                        stats['problem_files'].append({
                            'file': file,
                            'path': rel_path,
                            'problem': 'Erro de encoding UTF-8',
                            'severity': 'high'
                        })
                        continue
                    
                    # Verificar se arquivo está vazio
                    if len(content.strip()) == 0:
                        stats['empty_files'] += 1
                        stats['problem_files'].append({
                            'file': file,
                            'path': rel_path,
                            'problem': 'Arquivo vazio',
                            'severity': 'medium'
                        })
                        continue
                    
                    # Verificar critério de conteúdo do RAG (100 chars após limpeza)
                    clean_content = content.strip()
                    if len(clean_content) < 100:
                        stats['problem_files'].append({
                            'file': file,
                            'path': rel_path,
                            'problem': f'Conteúdo insuficiente ({len(clean_content)} chars < 100)',
                            'severity': 'medium'
                        })
                        continue
                    
                    # Arquivo passou em todos os critérios
                    stats['valid_files'] += 1
                    
                    if len(stats['sample_files']) < 5:
                        stats['sample_files'].append({
                            'file': file,
                            'path': rel_path,
                            'size': len(content),
                            'file_size': file_size,
                            'preview': content[:100].replace('\n', ' ')
                        })
                
                except Exception as e:
                    stats['encoding_errors'] += 1
                    stats['problem_files'].append({
                        'file': file,
                        'path': rel_path,
                        'problem': f'Erro ao ler: {str(e)}',
                        'severity': 'high'
                    })
    
    # Relatório detalhado
    print(f"\n📈 ESTATÍSTICAS DETALHADAS:")
    print(f"   📁 Diretórios processados: {stats['directories']}")
    print(f"   📄 Total de arquivos: {stats['total_files']}")
    print(f"   📝 Arquivos .txt/.md: {stats['txt_md_files']}")
    print(f"   ✅ Arquivos válidos (passaram filtros RAG): {stats['valid_files']}")
    print(f"   📏 Arquivos grandes (>100 bytes): {stats['large_files']}")
    print(f"   📉 Arquivos pequenos (<100 bytes): {stats['small_files']}")
    print(f"   📭 Arquivos vazios: {stats['empty_files']}")
    print(f"   🔤 Erros de encoding: {stats['encoding_errors']}")
    
    # Mostrar problemas específicos
    if stats['problem_files']:
        print(f"\n⚠️  PROBLEMAS ENCONTRADOS ({len(stats['problem_files'])}):")
        
        # Agrupar por tipo de problema
        by_severity = {'high': [], 'medium': []}
        for problem in stats['problem_files']:
            by_severity[problem['severity']].append(problem)
        
        for severity, problems in by_severity.items():
            if problems:
                emoji = "🔴" if severity == 'high' else "🟡"
                print(f"\n   {emoji} Problemas {severity.upper()} ({len(problems)}):")
                for prob in problems[:5]:  # Mostrar só os primeiros 5
                    location = f"{prob['path']}/{prob['file']}" if prob['path'] != "." else prob['file']
                    print(f"      • {location}: {prob['problem']}")
                
                if len(problems) > 5:
                    print(f"      ... e mais {len(problems) - 5} arquivos")
    
    # Mostrar exemplos de arquivos válidos
    if stats['sample_files']:
        print(f"\n🔍 EXEMPLOS DE ARQUIVOS VÁLIDOS:")
        for sample in stats['sample_files']:
            location = f"{sample['path']}/{sample['file']}" if sample['path'] != "." else sample['file']
            print(f"   📄 {location}")
            print(f"      Tamanho: {sample['file_size']} bytes → {sample['size']} chars")
            print(f"      Preview: {sample['preview']}...")
    
    # Sugestões baseadas nos problemas
    if stats['valid_files'] == 0:
        print(f"\n💡 SUGESTÕES PARA RESOLVER:")
        
        if stats['small_files'] > 0:
            print(f"   🔧 {stats['small_files']} arquivos muito pequenos (<100 bytes)")
            print(f"      → Reduzir filtro no código: file_size < 20 ao invés de < 100")
        
        if stats['empty_files'] > 0:
            print(f"   🔧 {stats['empty_files']} arquivos vazios")
            print(f"      → Verificar se arquivos têm conteúdo válido")
        
        if stats['encoding_errors'] > 0:
            print(f"   🔧 {stats['encoding_errors']} erros de encoding")
            print(f"      → Converter arquivos para UTF-8")
        
        problem_content = len([p for p in stats['problem_files'] if 'chars <' in p['problem']])
        if problem_content > 0:
            print(f"   🔧 {problem_content} arquivos com pouco conteúdo")
            print(f"      → Reduzir filtro: len(content) < 30 ao invés de < 100")
    
    return stats

def test_rag_loading():
    """Testa o carregamento do RAG com debug detalhado"""
    print_section("TESTANDO CARREGAMENTO RAG")
    
    try:
        # Importar sistema RAG
        try:
            from adaptive_rag import rag_system, init_rag_system, load_data_directory
            print("✅ Módulos RAG importados")
        except Exception as e:
            print(f"❌ Erro ao importar RAG: {e}")
            return False
        
        # Verificar configuração inicial
        print(f"\n📊 CONFIGURAÇÃO INICIAL:")
        print(f"   Data path: {getattr(rag_system, 'data_path', 'N/A')}")
        print(f"   PASTA_DESTINO: {os.getenv('PASTA_DESTINO')}")
        print(f"   Inicializado: {getattr(rag_system, 'is_initialized', False)}")
        print(f"   Documentos: {len(getattr(rag_system, 'documents', []))}")
        print(f"   Vector Store: {getattr(rag_system, 'vector_store', None) is not None}")
        
        # Verificar configurações do RAG
        if hasattr(rag_system, 'config'):
            config = rag_system.config
            print(f"\n⚙️  CONFIGURAÇÕES RAG:")
            print(f"   Chunk size: {getattr(config, 'chunk_size', 'N/A')}")
            print(f"   Chunk overlap: {getattr(config, 'chunk_overlap', 'N/A')}")
            print(f"   Top K: {getattr(config, 'top_k', 'N/A')}")
            print(f"   Max chunks: {getattr(config, 'max_chunks', 'N/A')}")
        
        # Tentar inicializar
        print("\n🚀 INICIALIZANDO SISTEMA...")
        init_success = init_rag_system()
        print(f"   Resultado: {'✅ Sucesso' if init_success else '❌ Falha'}")
        
        if not init_success:
            print("❌ Falha na inicialização - verificando detalhes...")
            
            # Verificar LLM
            try:
                if hasattr(rag_system, 'llm') and rag_system.llm:
                    test_response = rag_system.llm.invoke("teste")
                    print("✅ LLM funcional")
                else:
                    print("❌ LLM não inicializado")
            except Exception as e:
                print(f"❌ Erro no LLM: {e}")
            
            return False
        
        # Debug do carregamento de documentos
        print("\n📚 CARREGANDO DOCUMENTOS COM DEBUG...")
        
        # Interceptar o método para adicionar debug
        original_load = rag_system.load_documents_from_directory
        
        def debug_load():
            """Versão com debug do carregamento"""
            data_path = rag_system.data_path
            print(f"   📁 Pasta configurada: {data_path}")
            
            if not os.path.exists(data_path):
                print(f"   ❌ Pasta não existe: {data_path}")
                return 0
            
            arquivos_encontrados = 0
            arquivos_processados = 0
            arquivos_ignorados = 0
            documentos_criados = 0
            
            for root, dirs, files in os.walk(data_path):
                # Filtros de pasta (igual ao código original)
                original_dirs = dirs.copy()
                pastas_ignoradas = {'.rag_cache', 'anonimizados', 'dat', 'mapas', '__pycache__', '.git'}
                dirs[:] = [d for d in dirs if d not in pastas_ignoradas and not d.startswith('.')]
                
                ignored_dirs = set(original_dirs) - set(dirs)
                if ignored_dirs:
                    rel_path = os.path.relpath(root, data_path)
                    print(f"   🚫 Ignorando pastas em {rel_path}: {ignored_dirs}")
                
                for file in files:
                    if file.lower().endswith(('.txt', '.md')):
                        arquivos_encontrados += 1
                        filepath = os.path.join(root, file)
                        rel_file = os.path.relpath(filepath, data_path)
                        
                        try:
                            # Aplicar mesmos filtros do código original
                            file_size = os.path.getsize(filepath)
                            print(f"   📄 {rel_file}: {file_size} bytes", end="")
                            
                            if file_size < 100:  # Critério original
                                print(" → ❌ Muito pequeno")
                                arquivos_ignorados += 1
                                continue
                            
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                            
                            print(f" → {len(content)} chars", end="")
                            
                            if len(content) < 100:  # Critério original
                                print(" → ❌ Conteúdo insuficiente")
                                arquivos_ignorados += 1
                                continue
                            
                            # Simular limpeza de conteúdo
                            clean_content = content
                            # Aplicar limpezas básicas (simplificado)
                            import re
                            clean_content = re.sub(r'\s+', ' ', clean_content)
                            
                            if len(clean_content.strip()) < 50:  # Critério pós-limpeza
                                print(" → ❌ Insuficiente após limpeza")
                                arquivos_ignorados += 1
                                continue
                            
                            print(" → ✅ Válido")
                            arquivos_processados += 1
                            
                            # Simular criação de chunks
                            if len(clean_content) > 2000:
                                chunks = [clean_content[i:i+2000] for i in range(0, len(clean_content), 1600)]
                                documentos_criados += len(chunks)
                                print(f"      📑 {len(chunks)} chunks criados")
                            else:
                                documentos_criados += 1
                        
                        except Exception as e:
                            print(f" → ❌ Erro: {e}")
                            arquivos_ignorados += 1
            
            print(f"\n   📊 RESULTADO DO DEBUG:")
            print(f"      📄 Arquivos .txt/.md encontrados: {arquivos_encontrados}")
            print(f"      ✅ Arquivos processados: {arquivos_processados}")
            print(f"      ❌ Arquivos ignorados: {arquivos_ignorados}")
            print(f"      📚 Documentos que seriam criados: {documentos_criados}")
            
            # Chamar método original
            return original_load()
        
        # Executar carregamento com debug
        docs_loaded = debug_load()
        print(f"\n   📚 Documentos efetivamente carregados: {docs_loaded}")
        
        if docs_loaded == 0:
            print("❌ NENHUM DOCUMENTO CARREGADO!")
            
            # Análise de causa raiz
            print("\n🔍 ANÁLISE DE CAUSA RAIZ:")
            
            # Verificar se há arquivos que deveriam ser processados
            pasta = getattr(rag_system, 'data_path', os.getenv('PASTA_DESTINO'))
            if pasta and os.path.exists(pasta):
                txt_md_files = []
                for root, dirs, files in os.walk(pasta):
                    for file in files:
                        if file.lower().endswith(('.txt', '.md')):
                            txt_md_files.append(os.path.join(root, file))
                
                print(f"   📄 Total de arquivos .txt/.md na pasta: {len(txt_md_files)}")
                
                if len(txt_md_files) > 0:
                    print("   🔍 Verificando primeiros 3 arquivos:")
                    for filepath in txt_md_files[:3]:
                        try:
                            size = os.path.getsize(filepath)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            rel_path = os.path.relpath(filepath, pasta)
                            
                            issues = []
                            if size < 100:
                                issues.append(f"muito pequeno ({size}B)")
                            if len(content.strip()) < 100:
                                issues.append(f"pouco conteúdo ({len(content.strip())}C)")
                            
                            if issues:
                                print(f"      ❌ {rel_path}: {', '.join(issues)}")
                            else:
                                print(f"      ✅ {rel_path}: deveria ser processado!")
                        
                        except Exception as e:
                            print(f"      ❌ {os.path.basename(filepath)}: erro {e}")
                
                # Sugerir correções específicas
                print("\n   💡 CORREÇÕES SUGERIDAS:")
                print("      1. Reduzir filtro de tamanho: file_size < 20 (ao invés de 100)")
                print("      2. Reduzir filtro de conteúdo: len(content) < 30 (ao invés de 100)")
                print("      3. Aplicar script de correção de emergência")
            
            return False
        else:
            print(f"✅ {docs_loaded} documentos carregados com sucesso!")
            
            # Mostrar amostras detalhadas
            if hasattr(rag_system, 'documents') and rag_system.documents:
                print(f"\n🔍 PRIMEIROS 3 DOCUMENTOS CARREGADOS:")
                for i, doc in enumerate(rag_system.documents[:3]):
                    if hasattr(doc, 'metadata'):
                        meta = doc.metadata
                        content = doc.page_content
                    else:
                        meta = doc.get('metadata', {})
                        content = doc.get('page_content', '')
                    
                    filename = meta.get('filename', 'N/A')
                    processo = meta.get('numero_processo', 'N/A')
                    source = meta.get('source', 'N/A')
                    
                    print(f"   {i+1}. {filename}")
                    print(f"      📍 Caminho: {source}")
                    print(f"      ⚖️  Processo: {processo}")
                    print(f"      📏 Tamanho: {len(content)} chars")
                    print(f"      📖 Preview: {content[:100].replace(chr(10), ' ')}...")
            
            return True
            
    except Exception as e:
        print(f"❌ Erro no teste de carregamento: {e}")
        traceback.print_exc()
        return False

def test_simple_query():
    """Testa uma consulta simples"""
    print_section("TESTANDO CONSULTA SIMPLES")
    
    try:
        from adaptive_rag import rag_system
        
        if not getattr(rag_system, 'is_initialized', False):
            print("❌ Sistema não inicializado")
            return False
        
        if not getattr(rag_system, 'documents', []):
            print("❌ Nenhum documento carregado")
            return False
        
        # Teste simples
        print("🔍 Testando consulta: 'processo'")
        result = rag_system.query("processo", top_k=3)
        
        if 'error' in result:
            print(f"❌ Erro na consulta: {result['error']}")
            return False
        else:
            print("✅ Consulta funcionou!")
            print(f"   Resposta: {len(result.get('answer', ''))} chars")
            print(f"   Docs encontrados: {result.get('documents_found', 0)}")
            return True
            
    except Exception as e:
        print(f"❌ Erro no teste de consulta: {e}")
        return False

def apply_emergency_fixes(file_stats):
    """Aplica correções de emergência baseadas nos problemas encontrados"""
    print_section("APLICANDO CORREÇÕES DE EMERGÊNCIA")
    
    if file_stats.get('valid_files', 0) > 0:
        print("✅ Arquivos válidos encontrados, correção não necessária")
        return True
    
    try:
        from adaptive_rag import rag_system
        print("🔧 Aplicando correções nos filtros do RAG...")
        
        # Salvar métodos originais
        original_load = rag_system.load_documents_from_directory
        
        def flexible_load_documents():
            """Versão mais flexível do carregamento"""
            print("🚨 Carregamento de emergência com filtros flexíveis...")
            
            data_path = rag_system.data_path
            if not os.path.exists(data_path):
                print(f"❌ Pasta não encontrada: {data_path}")
                return 0
            
            documents = []
            stats = {'processed': 0, 'skipped': 0, 'errors': 0}
            
            # Usar critérios muito mais flexíveis
            MIN_FILE_SIZE = 20      # Reduzido de 100
            MIN_CONTENT_SIZE = 10   # Reduzido de 100
            
            for root, dirs, files in os.walk(data_path):
                # Filtrar apenas pastas essenciais (menos restritivo)
                dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                
                for file in files:
                    if not file.lower().endswith(('.txt', '.md')):
                        continue
                    
                    filepath = os.path.join(root, file)
                    
                    try:
                        # Filtro de tamanho mais flexível
                        file_size = os.path.getsize(filepath)
                        if file_size < MIN_FILE_SIZE:
                            stats['skipped'] += 1
                            continue
                        
                        # Tentar múltiplos encodings
                        content = None
                        for encoding in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                with open(filepath, 'r', encoding=encoding) as f:
                                    content = f.read()
                                break
                            except UnicodeDecodeError:
                                continue
                        
                        if content is None:
                            stats['errors'] += 1
                            continue
                        
                        # Filtro de conteúdo mais flexível
                        clean_content = content.strip()
                        if len(clean_content) < MIN_CONTENT_SIZE:
                            stats['skipped'] += 1
                            continue
                        
                        # Criar documento
                        try:
                            from langchain.schema import Document
                        except ImportError:
                            # Fallback para dict simples
                            Document = lambda page_content, metadata: {
                                'page_content': page_content, 'metadata': metadata
                            }
                        
                        metadata = {
                            'filename': file,
                            'source': filepath,
                            'relative_path': os.path.relpath(filepath, data_path),
                            'file_size': file_size,
                            'emergency_load': True
                        }
                        
                        # Extrair número de processo se possível
                        import re
                        processo_match = re.search(r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})', content + file)
                        if processo_match:
                            metadata['numero_processo'] = processo_match.group(1)
                        
                        # Chunking simples se necessário
                        if len(clean_content) > 1500:
                            chunk_size = 1500
                            overlap = 200
                            for i in range(0, len(clean_content), chunk_size - overlap):
                                chunk = clean_content[i:i + chunk_size]
                                if len(chunk.strip()) > MIN_CONTENT_SIZE:
                                    chunk_meta = metadata.copy()
                                    chunk_meta['chunk_index'] = i // (chunk_size - overlap)
                                    chunk_meta['is_chunk'] = True
                                    
                                    doc = Document(page_content=chunk, metadata=chunk_meta)
                                    documents.append(doc)
                        else:
                            metadata['is_chunk'] = False
                            doc = Document(page_content=clean_content, metadata=metadata)
                            documents.append(doc)
                        
                        stats['processed'] += 1
                        print(f"   ✅ {file} ({len(clean_content)} chars)")
                        
                    except Exception as e:
                        stats['errors'] += 1
                        print(f"   ❌ {file}: {e}")
            
            print(f"\n🚨 RESULTADO DA CORREÇÃO:")
            print(f"   ✅ Processados: {stats['processed']}")
            print(f"   ⏭️ Ignorados: {stats['skipped']}")
            print(f"   ❌ Erros: {stats['errors']}")
            print(f"   📚 Documentos criados: {len(documents)}")
            
            # Atualizar sistema RAG
            rag_system.documents = documents
            
            # Tentar criar vector store
            if len(documents) > 0:
                try:
                    rag_system._create_vector_store()
                    print("   ✅ Vector store criado")
                except Exception as e:
                    print(f"   ⚠️ Vector store não criado: {e}")
            
            return len(documents)
        
        # Substituir método de carregamento
        rag_system.load_documents_from_directory = flexible_load_documents
        
        # Ajustar configurações se existirem
        if hasattr(rag_system, 'config'):
            original_config = {}
            config = rag_system.config
            
            # Salvar configurações originais
            for attr in ['chunk_size', 'chunk_overlap', 'top_k', 'max_chunks']:
                if hasattr(config, attr):
                    original_config[attr] = getattr(config, attr)
            
            # Aplicar configurações mais flexíveis
            if hasattr(config, 'chunk_size'):
                config.chunk_size = 1500
            if hasattr(config, 'chunk_overlap'):
                config.chunk_overlap = 200
            if hasattr(config, 'top_k'):
                config.top_k = 8
            if hasattr(config, 'max_chunks'):
                config.max_chunks = 2000
            
            print("   🔧 Configurações ajustadas para serem mais flexíveis")
        
        print("✅ Correções aplicadas com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao aplicar correções: {e}")
        return False

def create_test_files():
    """Cria arquivos de teste se nenhum arquivo válido for encontrado"""
    print_section("CRIANDO ARQUIVOS DE TESTE")
    
    pasta_destino = os.getenv("PASTA_DESTINO")
    if not pasta_destino:
        pasta_destino = "./data_teste"
        print(f"🔧 PASTA_DESTINO não definida, usando: {pasta_destino}")
    
    try:
        os.makedirs(pasta_destino, exist_ok=True)
        
        # Arquivo de teste 1
        test_file_1 = os.path.join(pasta_destino, "processo_teste_1.md")
        content_1 = """---
numero_processo: "1234567-89.2025.8.11.0000"
agravante: "João da Silva"
agravado: "Município de Teste"
valor_causa: "R$ 5.000,00"
assuntos: "Saúde, Liminar, TEA"
---

# Processo de Teste - TEA

Este é um documento de teste para verificar o funcionamento do sistema RAG.

## Informações do Processo

O agravante João da Silva, representado por sua genitora, sustenta que tem direito ao tratamento para Transtorno do Espectro Autista (TEA).

A defesa argumenta que:
- O SUS já fornece fisioterapia e fonoaudiologia
- A terapia ABA não tem evidência científica suficiente
- Os custos são elevados para o município

O valor da causa foi fixado em R$ 5.000,00 com base nos custos do tratamento mensal.

## Decisão Liminar

Deferida a liminar para garantir:
1. Fonoaudiologia especializada (3x por semana)
2. Terapia ocupacional (3x por semana)
3. Atendimento com neuropediatra
4. Professor de apoio especializado

A terapia ABA foi indeferida com base no parecer NAT-JUS que indica falta de evidência científica robusta.
"""
        
        # Arquivo de teste 2
        test_file_2 = os.path.join(pasta_destino, "processo_teste_2.md")
        content_2 = """# Agravo de Instrumento - Saúde

**Processo:** 2345678-90.2025.8.11.0000
**Agravante:** Maria dos Santos
**Agravado:** Estado de Mato Grosso

## Argumentos da Defesa

A agravante sustenta que:
- Há cerceamento de defesa no processo de origem
- O contraditório não foi observado adequadamente
- A urgência do tratamento justifica a concessão da liminar

## Motivação do Recurso

O recurso foi interposto em face de decisão que indeferiu pedido de:
- Cirurgia cardíaca de urgência
- Internação em UTI
- Medicamentos de alto custo

## Fundamentação Legal

Com base no artigo 196 da Constituição Federal, o direito à saúde é garantido pelo Estado através do SUS.

A responsabilidade é solidária entre União, Estados e Municípios.
"""
        
        # Salvar arquivos
        with open(test_file_1, 'w', encoding='utf-8') as f:
            f.write(content_1)
        
        with open(test_file_2, 'w', encoding='utf-8') as f:
            f.write(content_2)
        
        print(f"✅ Arquivos de teste criados:")
        print(f"   📄 {test_file_1} ({len(content_1)} chars)")
        print(f"   📄 {test_file_2} ({len(content_2)} chars)")
        
        # Atualizar variável de ambiente se necessário
        if not os.getenv("PASTA_DESTINO"):
            abs_path = os.path.abspath(pasta_destino)
            print(f"\n🔧 Para definir PASTA_DESTINO execute:")
            print(f"   export PASTA_DESTINO={abs_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar arquivos de teste: {e}")
        return False

def main():
    """Função principal de diagnóstico"""
    print_header("DIAGNÓSTICO AVANÇADO DO SISTEMA RAG")
    
    # 1. Verificar importações
    import_results = check_imports()
    
    # 2. Verificar Ollama
    ollama_ok = check_ollama_service()
    
    # 3. Verificar pasta de dados com foco em PASTA_DESTINO
    data_path, file_stats = check_data_directory()
    
    # 4. Testar carregamento RAG
    rag_loading_ok = False
    if import_results.get('adaptive_rag', False) and data_path:
        rag_loading_ok = test_rag_loading()
    
    # 5. Aplicar correções se necessário
    fixes_applied = False
    if not rag_loading_ok and data_path and file_stats.get('txt_md_files', 0) > 0:
        print(f"\n🔧 DETECTADOS PROBLEMAS DE CARREGAMENTO")
        print(f"   📄 {file_stats.get('txt_md_files', 0)} arquivos .md/.txt encontrados")
        print(f"   ✅ {file_stats.get('valid_files', 0)} arquivos passaram nos filtros")
        
        if file_stats.get('valid_files', 0) == 0:
            try:
                resposta = input("\n❓ Aplicar correções automáticas nos filtros? (s/N): ").lower().strip()
                if resposta in ['s', 'sim', 'y', 'yes']:
                    if apply_emergency_fixes(file_stats):
                        print("✅ Correções aplicadas! Testando novamente...")
                        rag_loading_ok = test_rag_loading()
                        fixes_applied = True
                    else:
                        print("❌ Falha ao aplicar correções")
            except KeyboardInterrupt:
                print("\n⏭️ Pulando correções automáticas")
    
    # 6. Oferecer criação de arquivos de teste se ainda não funcionar
    if not rag_loading_ok and not data_path:
        try:
            resposta = input("\n❓ Criar arquivos de teste? (s/N): ").lower().strip()
            if resposta in ['s', 'sim', 'y', 'yes']:
                if create_test_files():
                    print("✅ Arquivos de teste criados!")
                    print("🔄 Execute o diagnóstico novamente após definir PASTA_DESTINO")
                else:
                    print("❌ Falha ao criar arquivos de teste")
        except KeyboardInterrupt:
            print("\n⏭️ Pulando criação de arquivos de teste")
    
    # 7. Testar consulta se carregamento funcionou
    query_ok = False
    if rag_loading_ok:
        query_ok = test_simple_query()
    
    # Resumo final detalhado
    print_header("RESUMO DETALHADO DO DIAGNÓSTICO")
    
    # Status dos componentes
    components = [
        ("Langchain básico", import_results.get('langchain_basic', False)),
        ("Importação Ollama", import_results.get('ollama_import', False)),
        ("Serviço Ollama", ollama_ok),
        ("Vector Store", import_results.get('vector_store', False)),
        ("UltraFastRAG", import_results.get('ultrafast_rag', False)),
        ("Adaptive RAG", import_results.get('adaptive_rag', False)),
        ("PASTA_DESTINO", data_path is not None),
        ("Arquivos válidos", file_stats.get('valid_files', 0) > 0),
        ("Carregamento RAG", rag_loading_ok),
        ("Consulta funcionando", query_ok)
    ]
    
    print("📊 STATUS DOS COMPONENTES:")
    for name, status in components:
        emoji = "✅" if status else "❌"
        print(f"   {emoji} {name}")
    
    # Estatísticas de arquivos se disponível
    if file_stats:
        print(f"\n📄 ESTATÍSTICAS DE ARQUIVOS:")
        print(f"   📁 Pasta: {data_path}")
        print(f"   📝 Arquivos .txt/.md: {file_stats.get('txt_md_files', 0)}")
        print(f"   ✅ Arquivos válidos: {file_stats.get('valid_files', 0)}")
        print(f"   📉 Arquivos pequenos: {file_stats.get('small_files', 0)}")
        print(f"   📭 Arquivos vazios: {file_stats.get('empty_files', 0)}")
        print(f"   🔤 Erros de encoding: {file_stats.get('encoding_errors', 0)}")
    
    # Próximos passos personalizados
    print(f"\n💡 PRÓXIMOS PASSOS RECOMENDADOS:")
    
    success_count = sum(status for _, status in components)
    
    if success_count >= 8:  # Quase tudo funcionando
        print("🎉 Sistema quase totalmente funcional!")
        if not query_ok:
            print("   🔍 Testar consultas mais específicas")
            print("   📊 Verificar qualidade das respostas")
    elif success_count >= 6:  # Problemas menores
        if not data_path:
            print("   📁 Definir PASTA_DESTINO corretamente")
            print("   📝 Adicionar arquivos .md/.txt na pasta")
        elif file_stats.get('valid_files', 0) == 0:
            if fixes_applied:
                print("   🔄 Reiniciar aplicação após correções")
            else:
                print("   🔧 Aplicar correções nos filtros de carregamento")
                print("   📝 Verificar formato dos arquivos")
    else:  # Problemas maiores
        print("   🚨 Problemas fundamentais detectados:")
        if not ollama_ok:
            print("      1. Instalar e iniciar Ollama: ollama serve")
            print("      2. Baixar modelo: ollama pull qwen3:1.7b")
        if not import_results.get('adaptive_rag'):
            print("      3. Verificar módulos Python (adaptive_rag.py)")
        if not data_path:
            print("      4. Configurar PASTA_DESTINO")
    
    # Comandos úteis
    print(f"\n🛠️  COMANDOS ÚTEIS:")
    if not ollama_ok:
        print("   # Verificar Ollama:")
        print("   ollama list")
        print("   ollama serve")
    
    if not data_path or file_stats.get('valid_files', 0) == 0:
        print("   # Verificar arquivos:")
        pasta = data_path or "${PASTA_DESTINO}"
        print(f"   find {pasta} -name '*.md' -o -name '*.txt' | head -5")
        print(f"   ls -la {pasta}")
    
    if fixes_applied:
        print("   # Testar sistema após correções:")
        print("   python -c \"from adaptive_rag import rag_system; print(rag_system.query('teste'))\"")
    
    # Status final
    if query_ok:
        print(f"\n🎉 SUCESSO TOTAL!")
        print(f"   ✅ Sistema RAG funcionando completamente")
        print(f"   🚀 Pronto para uso em produção")
    elif rag_loading_ok:
        print(f"\n🟡 SUCESSO PARCIAL!")
        print(f"   ✅ Documentos carregados, mas consulta com problemas")
        print(f"   🔍 Verificar LLM e configurações de busca")
    elif fixes_applied:
        print(f"\n🔄 CORREÇÕES APLICADAS!")
        print(f"   🔧 Filtros ajustados para serem mais flexíveis")
        print(f"   🔄 Execute o diagnóstico novamente")
    else:
        print(f"\n❌ PROBLEMAS DETECTADOS!")
        print(f"   📊 {success_count}/10 componentes funcionando")
        print(f"   🔧 Siga os próximos passos acima")

if __name__ == "__main__":
    main()