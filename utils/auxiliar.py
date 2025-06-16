# ==========================================
# 🛠️ FUNÇÕES AUXILIARES
# ==========================================

import re
import pandas as pd
import io
import logging
from utils.anonimizacao import AnonimizadorOtimizado

logger = logging.getLogger(__name__)
anonimizador_global = None

def get_anonimizador():
    global anonimizador_global
    if anonimizador_global is None:
        caminho_palavras = "utils/palavras_descartadas.txt"
        anonimizador_global = AnonimizadorOtimizado(caminho_palavras)
    return anonimizador_global


def limpar(valor):
    return str(valor).strip() if valor is not None else ""

def extrair_tabela_md(arquivo_md):
    try:
        with open(arquivo_md, 'r', encoding='utf-8') as f:
            linhas = f.readlines()

        inicio = next((i for i, l in enumerate(linhas) if re.match(r'^\|.+\|$', l)), None)
        if inicio is None:
            logger.warning(f" Nenhuma tabela encontrada em {arquivo_md}")
            return []

        tabela_linhas = []
        for linha in linhas[inicio:]:
            if not linha.strip().startswith('|'):
                break
            if re.match(r'^\|\s*-+\s*\|', linha):
                continue
            tabela_linhas.append(linha.strip())

        if not tabela_linhas:
            logger.warning(f" Tabela vazia em {arquivo_md}")
            return []

        tabela_str = '\n'.join(tabela_linhas)
        
        try:
            df = pd.read_csv(io.StringIO(tabela_str), sep='|', engine='python', skipinitialspace=True)
            df = df.dropna(axis=1, how='all')
            df.columns = [col.strip() for col in df.columns]
        except Exception as e:
            logger.error(f" Erro ao processar CSV: {e}")
            return []

        processos = []
        for _, row in df.iterrows():
            processos.append({
                "numeroProcesso": limpar(row.get("Nº Processo")),
                "tema": limpar(row.get("Tema")),
                "dataDistribuicao": limpar(row.get("Data da Distribuição")),
                "responsavel": limpar(row.get("Responsável")),
                "status": limpar(row.get("Status")),
                "ultimaAtualizacao": limpar(row.get("Última Atualização")),
                "suspeitos": limpar(row.get('Suspeitos')),
                "comentarios": limpar(row.get("Comentários")) if "Comentários" in row else ""
            })
        
        logger.info(f" {len(processos)} processos extraídos de {arquivo_md}")
        return processos
        
    except Exception as e:
        logger.error(f" Erro ao extrair tabela de {arquivo_md}: {str(e)}")
        return []
