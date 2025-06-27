import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

def extrair_metadados_processo(markdown_text: str) -> Dict[str, str]:
    metadados = {
        'numero_processo': '',
        'classe': '',
        'orgao_julgador_colegiado': '',
        'orgao_julgador': '',
        'data_distribuicao': '',
        'valor_causa': '',
        'processo_referencia': '',
        'assuntos': '',
        'objeto_processo': '',
        'nivel_sigilo': '',
        'justica_gratuita': '',
        'pedido_liminar': '',
        'agravante': '',
        'agravado': '',
        'terceiro_interessado': '',
        'custos_legis': '',
        'representante': ''
    }
    try:
        linhas = markdown_text.split('\n')
        texto_primeira_pagina = '\n'.join(linhas[:100])
        patterns = {
            'numero_processo': [
                r'Número:\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})',
                r'(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})',
                r'Nº\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})'
            ],
            'classe': [
                r'Classe:\s*([A-Z\s]+?)(?:\n|$)',
                r'CLASSE:\s*([A-Z\s]+?)(?:\n|$)'
            ],
            'orgao_julgador_colegiado': [
                r'Órgão julgador colegiado:\s*(.+?)(?:\n|$)',
                r'ÓRGÃO JULGADOR COLEGIADO:\s*(.+?)(?:\n|$)'
            ],
            'orgao_julgador': [
                r'Órgão julgador:\s*(.+?)(?:\n|$)',
                r'ÓRGÃO JULGADOR:\s*(.+?)(?:\n|$)'
            ],
            'data_distribuicao': [
                r'Última distribuição\s*:\s*(\d{2}/\d{2}/\d{4})',
                r'Data da distribuição:\s*(\d{2}/\d{2}/\d{4})',
                r'Distribuição:\s*(\d{2}/\d{2}/\d{4})'
            ],
            'valor_causa': [
                r'Valor da causa:\s*(R\$\s*[\d.,]+)',
                r'VALOR DA CAUSA:\s*(R\$\s*[\d.,]+)'
            ],
            'processo_referencia': [
                r'Processo referência:\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})',
                r'PROCESSO REFERÊNCIA:\s*(\d{7}-\d{2}\.\d{4}\.\d{1}\.\d{2}\.\d{4})'
            ],
            'assuntos': [
                r'Assuntos:\s*(.+?)(?:\n|Objeto do processo:)',
                r'ASSUNTOS:\s*(.+?)(?:\n|OBJETO DO PROCESSO:)'
            ],
            'objeto_processo': [
                r'Objeto do processo:\s*(.+?)(?:\n|Nível de Sigilo:)',
                r'OBJETO DO PROCESSO:\s*(.+?)(?:\n|NÍVEL DE SIGILO:)'
            ],
            'nivel_sigilo': [
                r'Nível de Sigilo:\s*(.+?)(?:\n|$)',
                r'NÍVEL DE SIGILO:\s*(.+?)(?:\n|$)'
            ],
            'justica_gratuita': [
                r'Justiça gratuita\?\s*(.+?)(?:\n|$)',
                r'JUSTIÇA GRATUITA\?\s*(.+?)(?:\n|$)'
            ],
            'pedido_liminar': [
                r'Pedido de liminar ou antecipação de tutela\?\s*(.+?)(?:\n|$)',
                r'PEDIDO DE LIMINAR.*?\?\s*(.+?)(?:\n|$)'
            ]
        }
        for campo, lista_patterns in patterns.items():
            for pattern in lista_patterns:
                match = re.search(pattern, texto_primeira_pagina, re.IGNORECASE | re.MULTILINE)
                if match:
                    metadados[campo] = match.group(1).strip()
                    break
        metadados.update(_extrair_partes_processo(texto_primeira_pagina))
        metadados = _limpar_metadados(metadados)
        logger.info(f"🔍 Metadados extraídos: {len([v for v in metadados.values() if v])} campos preenchidos")
        return metadados
    except Exception as e:
        logger.error(f"❌ Erro ao extrair metadados: {str(e)}")
        return metadados

def _extrair_partes_processo(texto: str) -> Dict[str, str]:
    partes = {
        'agravante': '',
        'agravado': '',
        'terceiro_interessado': '',
        'custos_legis': '',
        'representante': ''
    }
    try:
        patterns_partes = {
            'agravante': [
                r'AGRAVANTE:\s*(.+?)(?:\n|AGRAVADO:)',
                r'Agravante:\s*(.+?)(?:\n|Agravado:)'
            ],
            'agravado': [
                r'AGRAVADO:\s*(.+?)(?:\n|TERCEIRO|Outros participantes)',
                r'Agravado:\s*(.+?)(?:\n|Terceiro|Outros participantes)'
            ],
            'terceiro_interessado': [
                r'TERCEIRO INTERESSADO[)\s]*(.+?)(?:\n|MINISTERIO)',
                r'Terceiro Interessado[)\s]*(.+?)(?:\n|Ministério)'
            ],
            'custos_legis': [
                r'CUSTOS LEGIS[)\s]*(.+?)(?:\n|$)',
                r'Custos Legis[)\s]*(.+?)(?:\n|$)',
                r'MINISTERIO PUBLICO.*?CUSTOS LEGIS[)\s]*(.+?)(?:\n|$)'
            ],
            'representante': [
                r'REPRESENTANTE/NOTICIANTE[)\s]*(.+?)(?:\n|$)',
                r'Representante[)\s]*(.+?)(?:\n|$)'
            ]
        }
        for tipo_parte, lista_patterns in patterns_partes.items():
            for pattern in lista_patterns:
                match = re.search(pattern, texto, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    partes[tipo_parte] = match.group(1).strip()
                    break
    except Exception as e:
        logger.error(f"❌ Erro ao extrair partes: {str(e)}")
    return partes

def _limpar_metadados(metadados: Dict[str, str]) -> Dict[str, str]:
    metadados_limpos = {}
    for chave, valor in metadados.items():
        if valor:
            valor_limpo = re.sub(r'\s+', ' ', valor.strip())
            valor_limpo = valor_limpo.replace('\n', ' ').replace('\r', '')
            if len(valor_limpo) > 200:
                valor_limpo = valor_limpo[:200] + '...'
            metadados_limpos[chave] = valor_limpo
        else:
            metadados_limpos[chave] = ''
    return metadados_limpos

def formatar_metadados_para_markdown(metadados: Dict[str, str]) -> str:
    yaml_lines = ['---', '# METADADOS DO PROCESSO']
    if metadados.get('numero_processo'):
        yaml_lines.append(f"numero_processo: \"{metadados['numero_processo']}\"")
    if metadados.get('classe'):
        yaml_lines.append(f"classe: \"{metadados['classe']}\"")
    if metadados.get('data_distribuicao'):
        yaml_lines.append(f"data_distribuicao: \"{metadados['data_distribuicao']}\"")
    if metadados.get('valor_causa'):
        yaml_lines.append(f"valor_causa: \"{metadados['valor_causa']}\"")
    if metadados.get('orgao_julgador_colegiado'):
        yaml_lines.append(f"orgao_julgador_colegiado: \"{metadados['orgao_julgador_colegiado']}\"")
    if metadados.get('orgao_julgador'):
        yaml_lines.append(f"orgao_julgador: \"{metadados['orgao_julgador']}\"")
    if metadados.get('assuntos'):
        yaml_lines.append(f"assuntos: \"{metadados['assuntos']}\"")
    if metadados.get('objeto_processo'):
        yaml_lines.append(f"objeto_processo: \"{metadados['objeto_processo']}\"")
    yaml_lines.append('# PARTES ENVOLVIDAS')
    if metadados.get('agravante'):
        yaml_lines.append(f"agravante: \"{metadados['agravante']}\"")
    if metadados.get('agravado'):
        yaml_lines.append(f"agravado: \"{metadados['agravado']}\"")
    if metadados.get('terceiro_interessado'):
        yaml_lines.append(f"terceiro_interessado: \"{metadados['terceiro_interessado']}\"")
    if metadados.get('representante'):
        yaml_lines.append(f"representante: \"{metadados['representante']}\"")
    if metadados.get('processo_referencia'):
        yaml_lines.append(f"processo_referencia: \"{metadados['processo_referencia']}\"")
    if metadados.get('nivel_sigilo'):
        yaml_lines.append(f"nivel_sigilo: \"{metadados['nivel_sigilo']}\"")
    if metadados.get('justica_gratuita'):
        yaml_lines.append(f"justica_gratuita: \"{metadados['justica_gratuita']}\"")
    if metadados.get('pedido_liminar'):
        yaml_lines.append(f"pedido_liminar: \"{metadados['pedido_liminar']}\"")
    yaml_lines.append('---\n')
    return '\n'.join(yaml_lines)

def extrair_e_formatar_metadados(markdown_content):
    conteudo_local = str(markdown_content)
    print(f"🔍 [EXTRAÇÃO] Processando conteúdo de {len(conteudo_local)} caracteres")
    print(f"🔍 [EXTRAÇÃO] Primeiros 50 chars: {conteudo_local[:50]}...")
    metadados_extraidos = {}
    try:
        lines = conteudo_local.split('\n')
        for i, line in enumerate(lines[:20]):
            line_clean = line.strip()
            if 'processo' in line_clean.lower() and not metadados_extraidos.get('numero_processo'):
                import re
                match = re.search(r'\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}', line_clean)
                if match:
                    metadados_extraidos['numero_processo'] = match.group()
                    print(f"📋 [EXTRAÇÃO] Número processo encontrado: {metadados_extraidos['numero_processo']}")
        if metadados_extraidos:
            front_matter = "---\n"
            for key, value in metadados_extraidos.items():
                if value:
                    front_matter += f"{key}: {value}\n"
            front_matter += "---"
            print(f"✅ [EXTRAÇÃO] Front matter gerado: {len(metadados_extraidos)} campos")
            return metadados_extraidos, front_matter
        else:
            print("⚠️ [EXTRAÇÃO] Nenhum metadado extraído")
            return {}, ""
    except Exception as e:
        print(f"❌ [EXTRAÇÃO] Erro: {e}")
        return {}, ""

def salvar_arquivo_seguro(caminho, conteudo, processo_numero):
    print(f"💾 [SALVAR] Salvando {processo_numero}: {len(conteudo)} chars em {caminho}")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, 'w', encoding='utf-8', newline='') as f:
        f.write(conteudo)
    if os.path.exists(caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            conteudo_verificacao = f.read()
        if len(conteudo_verificacao) == len(conteudo):
            print(f"✅ [SALVAR] Arquivo salvo corretamente: {len(conteudo_verificacao)} chars")
            pasta = os.path.dirname(caminho)
            nome_arquivo = os.path.basename(caminho)
            for arquivo in os.listdir(pasta):
                if arquivo != nome_arquivo and arquivo.endswith('.md'):
                    caminho_outro = os.path.join(pasta, arquivo)
                    try:
                        with open(caminho_outro, 'r', encoding='utf-8') as f:
                            outro_conteudo = f.read()
                        if outro_conteudo == conteudo_verificacao:
                            print(f"⚠️ [SALVAR] ATENÇÃO: Conteúdo idêntico encontrado em {arquivo}!")
                            return False
                    except:
                        pass
            print(f"✅ [SALVAR] Arquivo único confirmado")
            return True
        else:
            print(f"❌ [SALVAR] Erro de integridade! Original: {len(conteudo)}, Salvo: {len(conteudo_verificacao)}")
            return False
    else:
        print(f"❌ [SALVAR] Arquivo não foi criado!")
        return False