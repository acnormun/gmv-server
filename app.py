from flask import Flask, jsonify, request
import pandas as pd
import re
import os
import io
from dotenv import load_dotenv
from flask_cors import CORS
from utils.suspeicao import encontrar_suspeitos

app = Flask(__name__)
CORS(app)

load_dotenv()
PATH_TRIAGEM = os.getenv("PATH_TRIAGEM")
PASTA_DESTINO = os.getenv("PASTA_DESTINO")
PASTA_DAT = os.getenv("PASTA_DAT")

if not PATH_TRIAGEM or not PASTA_DESTINO:
    raise ValueError("Verifique se PATH_TRIAGEM e PASTA_DESTINO estão definidos no .env")

def limpar(valor):
    return str(valor).strip() if valor is not None else ""

def extrair_tabela_md(arquivo_md):
    with open(arquivo_md, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    inicio = next((i for i, l in enumerate(linhas) if re.match(r'^\|.+\|$', l)), None)
    if inicio is None:
        return []

    tabela_linhas = []
    for linha in linhas[inicio:]:
        if not linha.strip().startswith('|'):
            break
        if re.match(r'^\|\s*-+\s*\|', linha):
            continue
        tabela_linhas.append(linha.strip())

    if not tabela_linhas:
        return []

    tabela_str = '\n'.join(tabela_linhas)
    df = pd.read_csv(io.StringIO(tabela_str), sep='|', engine='python', skipinitialspace=True)
    df = df.dropna(axis=1, how='all')
    df.columns = [col.strip() for col in df.columns]

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
    return processos

@app.route('/triagem', methods=['GET'])
def get_processos():
    dados = extrair_tabela_md(PATH_TRIAGEM)
    return jsonify(dados)

@app.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    try:
        data = request.get_json()
        numero = limpar(data.get('numeroProcesso'))
        tema = limpar(data.get('tema'))
        data_dist = limpar(data.get('dataDistribuicao'))
        responsavel = limpar(data.get('responsavel'))
        status = limpar(data.get('status'))
        ultima_att = limpar(data.get('ultimaAtualizacao'))
        markdown = limpar(data.get('markdown'))
        comentarios = limpar(data.get('comentarios'))
        dat_base64 = data.get('dat')
        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt' )

        if not markdown or not numero:
            return jsonify({'error': 'Campos obrigatórios ausentes'}), 400

        nome_arquivo_base = numero.replace('/', '-')
        os.makedirs(PASTA_DESTINO, exist_ok=True)
        os.makedirs(PASTA_DAT, exist_ok=True)

        caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
        caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")

        # Salva markdown
        with open(caminho_md, 'w', encoding='utf-8') as f:
            f.write(markdown)

        # Salva .dat como base64 se enviado
        if dat_base64:
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)

        nova_linha = (
            f"| {numero} "
            f"| {tema} "
            f"| {data_dist} "
            f"| {responsavel} "
            f"| {status} "
            f"| {ultima_att} "
            f"| {suspeitos} "
            f"| {comentarios} |\n"
        )

        if not os.path.exists(PATH_TRIAGEM):
            with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                f.write("# Tabela de Processos\n\n")
                f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-------------|\n")

        with open(PATH_TRIAGEM, 'r', encoding='utf-8') as f:
            linhas = f.readlines()

        indice_separador = next(
            (i for i, linha in enumerate(linhas) if re.match(r'^\|\s*-+\s*\|', linha.strip())),
            None
        )

        if indice_separador is not None:
            while indice_separador + 1 < len(linhas) and not linhas[indice_separador + 1].strip().startswith('|'):
                del linhas[indice_separador + 1]
            linhas.insert(indice_separador + 1, nova_linha)
        else:
            linhas += [
                "| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n",
                "|-------------|------|-----------------------|-------------|--------|----------------------|-------------|\n",
                nova_linha
            ]

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.writelines(linhas)

        return jsonify({"message": "Processo e arquivos salvos com sucesso"}), 201

    except Exception as e:
        print('❤',e)
        return jsonify({'error': str(e)}), 500


@app.route('/triagem/<numero>', methods=['PUT'])
def editar_processo(numero):
    try:
        data = request.get_json()
        processos = extrair_tabela_md(PATH_TRIAGEM)
        processos = [p for p in processos if p['numeroProcesso'] != numero]

        processos.append({
            "numeroProcesso": limpar(data['numeroProcesso']),
            "tema": limpar(data['tema']),
            "dataDistribuicao": limpar(data['dataDistribuicao']),
            "responsavel": limpar(data['responsavel']),
            "status": limpar(data['status']),
            "ultimaAtualizacao": limpar(data['ultimaAtualizacao']),
            "suspeitos": limpar(data['suspeitos']),
            "comentarios": limpar(data.get('comentarios', ''))
        })

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-------------|\n")
            for p in processos:
                f.write(
                    f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} "
                    f"| {p['status']} | {p['ultimaAtualizacao']} | {p.get('comentarios', '')} |\n"
                )

        return jsonify({"message": "Processo atualizado com sucesso"}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>', methods=['DELETE'])
def deletar_processo(numero):
    try:
        processos = extrair_tabela_md(PATH_TRIAGEM)
        processos = [p for p in processos if p['numeroProcesso'] != numero]

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-------------|\n")
            for p in processos:
                f.write(
                    f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} "
                    f"| {p['status']} | {p['ultimaAtualizacao']} | {p.get('comentarios', '')} |\n"
                )

        caminho_md = os.path.join(PASTA_DESTINO, f"{numero.replace('/', '-')}.md")
        if os.path.exists(caminho_md):
            os.remove(caminho_md)

        return jsonify({"message": "Processo excluído com sucesso"}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/triagem/<numero>/dat', methods=['GET'])
def obter_dat(numero):
    try:
        nome_arquivo = f"{numero.replace('/', '-')}.dat"
        caminho = os.path.join(PASTA_DAT, nome_arquivo)

        if not os.path.exists(caminho):
            return jsonify({'error': 'Arquivo .dat não encontrado'}), 404

        with open(caminho, 'r', encoding='utf-8') as f:
            dat_base64 = f.read()

        return jsonify({'base64': dat_base64}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
