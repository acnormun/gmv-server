import os
import re
import base64
import tempfile
from datetime import datetime
from PyPDF2 import PdfReader, PdfWriter
import pdf2image
from paddleocr import PaddleOCR
import numpy as np
from utils.progress_step import send_progress_ws

class ProcessadorPJeIntegrado:
    def __init__(self, operation_id=None):
        self.processo_numero = None
        self.pasta_processo = None
        self.ocr_engine = None
        self.operation_id = operation_id
        self.documentos_processados = []
        
    def log_progress(self, step, message, progress):
        if self.operation_id:
            send_progress_ws(self.operation_id, step, message, progress)
        print(f"📄 [PASSO {step}] {message} ({progress}%)")
    
    def inicializar_paddleocr(self):
        try:
            if self.ocr_engine is None:
                configs = [
                    {
                        'use_angle_cls': True, 
                        'lang': 'pt',
                        'use_gpu': False,
                        'show_log': False
                    },
                    {
                        'use_angle_cls': True, 
                        'lang': 'pt',
                        'use_gpu': False
                    },
                    {
                        'lang': 'pt'
                    },
                    {}
                ]
                for i, config in enumerate(configs):
                    try:
                        print(f"🔄 Tentando configuração PaddleOCR #{i+1}...")
                        self.ocr_engine = PaddleOCR(**config)
                        print(f"✅ PaddleOCR inicializado com configuração #{i+1}")
                        break
                    except Exception as e:
                        print(f"⚠️ Configuração #{i+1} falhou: {e}")
                        if i == len(configs) - 1:
                            raise e
                        continue
            return True, "✅ PaddleOCR inicializado com sucesso!"
        except Exception as e:
            return False, f"❌ Erro ao inicializar PaddleOCR: {str(e)}"
    
    def decodificar_pdf_base64(self, dat_base64, nome_processo):
        try:
            if ',' in dat_base64:
                dat_base64 = dat_base64.split(',')[1]
            pdf_bytes = base64.b64decode(dat_base64)
            nome_arquivo = f"{nome_processo.replace('/', '-')}_original.pdf"
            temp_path = os.path.join(tempfile.gettempdir(), nome_arquivo)
            with open(temp_path, 'wb') as f:
                f.write(pdf_bytes)
            print(f"📄 PDF decodificado salvo em: {temp_path}")
            return temp_path, len(pdf_bytes)
        except Exception as e:
            raise Exception(f"Erro ao decodificar PDF base64: {str(e)}")
    
    def extrair_numero_processo(self, texto):
        patterns = [
            r'(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})',
            r'(\d{13,20})',
            r'Processo[:\s]+(\d{4,})',
        ]
        for pattern in patterns:
            match = re.search(pattern, texto)
            if match:
                return match.group(1)
        return None
    
    def extrair_tabela_documentos(self, texto):
        documentos = []
        linhas = texto.split('\n')
        header_idx = -1
        for i, linha in enumerate(linhas):
            if any(keyword in linha.lower() for keyword in ['id.', 'data', 'assinatura', 'movimento', 'documento']):
                if any(char.isdigit() for char in linha):
                    header_idx = i
                    break
        if header_idx == -1:
            print("⚠️ Header da tabela de documentos não encontrado")
            return documentos
        for i in range(header_idx + 1, len(linhas)):
            linha = linhas[i].strip()
            if not linha:
                continue
            match = re.match(r'^(\d{9})\s+(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})\s+(.+)', linha)
            if match:
                id_doc = match.group(1)
                data_assinatura = match.group(2)
                resto = match.group(3)
                campos = [c.strip() for c in re.split(r'\s{2,}', resto.strip()) if c.strip()]
                movimento = campos[0] if len(campos) > 0 else 'Sem movimento'
                documento = campos[1] if len(campos) > 1 else campos[0] if len(campos) == 1 else ''
                documentos.append({
                    'id': id_doc,
                    'data_assinatura': data_assinatura,
                    'movimento': movimento,
                    'documento': documento
                })
        print(f"📋 {len(documentos)} documentos encontrados na tabela")
        return documentos
    
    def encontrar_inicio_documentos(self, pdf_reader, tabela_docs):
        documentos_localizados = []
        for pagina_idx, page in enumerate(pdf_reader.pages):
            try:
                texto_pagina = page.extract_text()
                if not texto_pagina:
                    continue
                patterns = [
                    r'Num\.\s*(\d{9})',
                    r'Número[\s:]+(\d{9})',
                    r'ID[\s:]+(\d{9})',
                ]
                for pattern in patterns:
                    matches = re.findall(pattern, texto_pagina)
                    for id_encontrado in matches:
                        doc_tabela = next((doc for doc in tabela_docs if doc['id'] == id_encontrado), None)
                        if doc_tabela:
                            if not any(d['id_documento'] == id_encontrado for d in documentos_localizados):
                                documentos_localizados.append({
                                    'id_documento': id_encontrado,
                                    'pagina_inicio': pagina_idx + 1,
                                    'nome_documento': doc_tabela['documento']
                                })
                                break
            except Exception as e:
                print(f"⚠️ Erro ao processar página {pagina_idx + 1}: {e}")
                continue
        documentos_localizados = sorted(documentos_localizados, key=lambda x: x['pagina_inicio'])
        print(f"📍 {len(documentos_localizados)} documentos localizados no PDF")
        return documentos_localizados
    
    def calcular_ranges_paginas(self, documentos_localizados, total_paginas):
        if not documentos_localizados:
            return []
        documentos_finais = []
        primeiro_inicio = documentos_localizados[0]['pagina_inicio']
        if primeiro_inicio > 1:
            documentos_finais.append({
                'id_documento': 'INICIAL',
                'nome_documento': 'Documento Inicial',
                'pagina_inicio': 1,
                'pagina_fim': primeiro_inicio - 1
            })
        for i, doc in enumerate(documentos_localizados):
            inicio = doc['pagina_inicio']
            if i + 1 < len(documentos_localizados):
                fim = documentos_localizados[i + 1]['pagina_inicio'] - 1
            else:
                fim = total_paginas
            documentos_finais.append({
                'id_documento': doc['id_documento'],
                'nome_documento': doc['nome_documento'],
                'pagina_inicio': inicio,
                'pagina_fim': fim
            })
        return documentos_finais
    
    def criar_estrutura_pastas(self, pasta_base, numero_processo):
        nome_pasta = re.sub(r'[^\w\-\.]', '_', numero_processo)
        self.pasta_processo = os.path.join(pasta_base, nome_pasta)
        subpastas = ['pdfs_separados', 'markdowns', 'original']
        for subpasta in subpastas:
            os.makedirs(os.path.join(self.pasta_processo, subpasta), exist_ok=True)
        print(f"📁 Estrutura de pastas criada: {self.pasta_processo}")
        return self.pasta_processo
    
    def separar_pdfs(self, pdf_reader, documentos):
        documentos_separados = []
        pasta_pdfs = os.path.join(self.pasta_processo, 'pdfs_separados')
        for doc in documentos:
            try:
                writer = PdfWriter()
                if doc.get('id_documento') == 'INICIAL':
                    nome_arquivo = "00_inicial.pdf"
                else:
                    nome_limpo = re.sub(r'[^\w\s\-]', '', doc.get('nome_documento', 'doc'))
                    nome_limpo = re.sub(r'\s+', '_', nome_limpo)[:30]
                    nome_arquivo = f"{doc['id_documento']}_{nome_limpo}.pdf"
                paginas_adicionadas = 0
                for pag_num in range(doc['pagina_inicio'], doc['pagina_fim'] + 1):
                    pag_idx = pag_num - 1
                    if 0 <= pag_idx < len(pdf_reader.pages):
                        writer.add_page(pdf_reader.pages[pag_idx])
                        paginas_adicionadas += 1
                if paginas_adicionadas > 0:
                    caminho_arquivo = os.path.join(pasta_pdfs, nome_arquivo)
                    with open(caminho_arquivo, 'wb') as output_file:
                        writer.write(output_file)
                    documentos_separados.append({
                        'arquivo': nome_arquivo,
                        'caminho': caminho_arquivo,
                        'nome_documento': doc.get('nome_documento', ''),
                        'paginas_range': f"{doc['pagina_inicio']}-{doc['pagina_fim']}",
                        'id_documento': doc.get('id_documento', ''),
                        'paginas_count': paginas_adicionadas
                    })
                    print(f"✅ PDF separado: {nome_arquivo} ({paginas_adicionadas} páginas)")
            except Exception as e:
                print(f"❌ Erro ao separar documento {doc.get('id_documento', 'N/A')}: {e}")
        return documentos_separados
    
    def paddle_ocr_rapido(self, image):
        try:
            img_array = np.array(image)
            result = self.ocr_engine.ocr(img_array, cls=True)
            texto_extraido = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        texto = line[1][0]
                        confianca = line[1][1]
                        if confianca > 0.7:
                            texto_extraido.append(texto)
            return '\n'.join(texto_extraido) if texto_extraido else "[NENHUM TEXTO DETECTADO]"
        except Exception as e:
            return f"*Erro PaddleOCR: {e}*"
    
    def converter_pdf_para_markdown(self, caminho_pdf, nome_documento=""):
        try:
            reader = PdfReader(caminho_pdf)
            texto_completo = []
            images = pdf2image.convert_from_path(caminho_pdf, dpi=200)
            for i, page in enumerate(reader.pages):
                texto_direto = ""
                try:
                    texto_pagina = page.extract_text()
                    if texto_pagina and len(texto_pagina.strip()) > 10:
                        texto_direto = texto_pagina.strip()
                except:
                    pass
                texto_ocr = ""
                if i < len(images):
                    texto_ocr = self.paddle_ocr_rapido(images[i])
                if texto_direto and texto_ocr and "ERRO" not in texto_ocr.upper():
                    if len(texto_ocr) > len(texto_direto) * 0.8:
                        texto_final = f"**[PDF + OCR]**\n\n{texto_direto}\n\n**[OCR Adicional]**\n\n{texto_ocr}"
                    else:
                        texto_final = f"**[Texto PDF]**\n\n{texto_direto}"
                elif texto_direto:
                    texto_final = f"**[Texto PDF]**\n\n{texto_direto}"
                elif texto_ocr and "ERRO" not in texto_ocr.upper():
                    texto_final = f"**[PaddleOCR]**\n\n{texto_ocr}"
                else:
                    texto_final = "*Página sem conteúdo extraível*"
                texto_completo.append(f"## Página {i+1}\n\n{texto_final}\n\n")
            return "\n".join(texto_completo)
        except Exception as e:
            return f"*Erro no processamento: {e}*"
    
    def processar_todos_documentos_ocr(self, documentos_separados):
        pasta_markdowns = os.path.join(self.pasta_processo, 'markdowns')
        total = len(documentos_separados)
        for i, doc in enumerate(documentos_separados):
            try:
                progress = 50 + int((i + 1) / total * 40)
                self.log_progress(5, f'Processando OCR: {doc["arquivo"]} ({i+1}/{total})', progress)
                texto_md = self.converter_pdf_para_markdown(doc['caminho'], doc['nome_documento'])
                nome_md = doc['arquivo'].replace('.pdf', '.md')
                caminho_md = os.path.join(pasta_markdowns, nome_md)
                front_matter = [
                    "---",
                    f"arquivo: '{doc['arquivo']}'",
                    f"id_documento: '{doc['id_documento']}'",
                    f"nome_documento: '{doc['nome_documento']}'",
                    f"paginas: '{doc['paginas_range']}'",
                    f"paginas_count: {doc['paginas_count']}",
                    "ocr: 'PaddleOCR (Português)'",
                    "---",
                    "",
                ]
                with open(caminho_md, 'w', encoding='utf-8') as f:
                    f.write("\n".join(front_matter))
                    f.write(f"# {doc['nome_documento']}\n\n")
                    f.write(texto_md)
                doc['caminho_markdown'] = caminho_md
                self.documentos_processados.append(doc)
                print(f"✅ Markdown gerado: {nome_md}")
            except Exception as e:
                print(f"❌ Erro ao processar OCR para {doc['arquivo']}: {e}")
        return len(self.documentos_processados)
    
    def processar_pdf_completo(self, dat_base64, numero_processo, pasta_destino):
        try:
            self.log_progress(1, 'Inicializando PaddleOCR...', 5)
            sucesso_ocr, msg_ocr = self.inicializar_paddleocr()
            if not sucesso_ocr:
                raise Exception(msg_ocr)
            self.log_progress(2, 'Decodificando PDF base64...', 10)
            temp_pdf_path, tamanho_bytes = self.decodificar_pdf_base64(dat_base64, numero_processo)
            self.log_progress(3, 'Analisando estrutura do PDF...', 15)
            pdf_reader = PdfReader(temp_pdf_path)
            total_paginas = len(pdf_reader.pages)
            print(f"📄 PDF carregado: {total_paginas} páginas, {tamanho_bytes} bytes")
            self.log_progress(4, 'Extraindo tabela de documentos...', 20)
            texto_inicial = ""
            for i in range(min(5, total_paginas)):
                try:
                    texto_inicial += pdf_reader.pages[i].extract_text() + "\n"
                except:
                    continue
            if not self.processo_numero:
                numero_extraido = self.extrair_numero_processo(texto_inicial)
                if numero_extraido:
                    self.processo_numero = numero_extraido
                    print(f"🔍 Número do processo extraído: {numero_extraido}")
                else:
                    self.processo_numero = numero_processo
            tabela_docs = self.extrair_tabela_documentos(texto_inicial)
            if not tabela_docs:
                print("⚠️ Tabela de documentos não encontrada - processando como documento único")
                tabela_docs = [{
                    'id': 'UNICO',
                    'data_assinatura': datetime.now().strftime('%d/%m/%Y %H:%M'),
                    'movimento': 'Documento único',
                    'documento': 'Processo completo'
                }]
            self.log_progress(5, 'Criando estrutura de pastas...', 25)
            self.criar_estrutura_pastas(pasta_destino, self.processo_numero)
            pasta_original = os.path.join(self.pasta_processo, 'original')
            pdf_original_path = os.path.join(pasta_original, f"{self.processo_numero.replace('/', '-')}_original.pdf")
            with open(temp_pdf_path, 'rb') as src, open(pdf_original_path, 'wb') as dst:
                dst.write(src.read())
            self.log_progress(6, 'Localizando documentos no PDF...', 30)
            documentos_localizados = self.encontrar_inicio_documentos(pdf_reader, tabela_docs)
            if not documentos_localizados and len(tabela_docs) == 1:
                documentos_finais = [{
                    'id_documento': 'COMPLETO',
                    'nome_documento': 'Processo Completo',
                    'pagina_inicio': 1,
                    'pagina_fim': total_paginas
                }]
            else:
                documentos_finais = self.calcular_ranges_paginas(documentos_localizados, total_paginas)
            if not documentos_finais:
                raise Exception("Não foi possível identificar a estrutura de documentos no PDF")
            self.log_progress(7, f'Separando PDF em {len(documentos_finais)} documentos...', 40)
            documentos_separados = self.separar_pdfs(pdf_reader, documentos_finais)
            if not documentos_separados:
                raise Exception("Nenhum documento pôde ser separado")
            self.log_progress(8, 'Iniciando processamento OCR...', 50)
            total_processados = self.processar_todos_documentos_ocr(documentos_separados)
            try:
                os.unlink(temp_pdf_path)
            except:
                pass
            resultado = {
                'sucesso': True,
                'numero_processo': self.processo_numero,
                'pasta_processo': self.pasta_processo,
                'total_paginas': total_paginas,
                'documentos_tabela': len(tabela_docs),
                'documentos_separados': len(documentos_separados),
                'documentos_processados': total_processados,
                'tamanho_original_bytes': tamanho_bytes,
                'arquivos_gerados': {
                    'pdf_original': pdf_original_path,
                    'pdfs_separados': [doc['caminho'] for doc in documentos_separados],
                    'markdowns': [doc['caminho_markdown'] for doc in self.documentos_processados if 'caminho_markdown' in doc]
                }
            }
            self.log_progress(9, 'Processamento PJe concluído com sucesso!', 100)
            print(f"🎉 Processamento concluído:")
            print(f"   📁 Pasta: {self.pasta_processo}")
            print(f"   📄 {len(documentos_separados)} PDFs separados")
            print(f"   📝 {total_processados} Markdowns gerados")
            return resultado
        except Exception as e:
            try:
                if 'temp_pdf_path' in locals():
                    os.unlink(temp_pdf_path)
            except:
                pass
            self.log_progress(0, f'Erro no processamento PJe: {str(e)}', 0)
            raise Exception(f"Erro no processamento PJe: {str(e)}")

def processar_pje_com_progresso(dat_base64, numero_processo, pasta_destino, operation_id=None):
    processador = ProcessadorPJeIntegrado(operation_id)
    return processador.processar_pdf_completo(dat_base64, numero_processo, pasta_destino)