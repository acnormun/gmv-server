from flask import Flask, jsonify, request
import pandas as pd
import re
import os
import io
from dotenv import load_dotenv
from flask_cors import CORS
from utils.suspeicao import encontrar_suspeitos

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173"]}})

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

# Correção para POST /triagem/form
@app.route('/triagem/form', methods=['POST'])
def receber_processo_com_markdown():
    print("📝 Solicitação POST /triagem/form recebida")
    try:
        data = request.get_json()
        print(f"📄 Dados recebidos: {data}")
        
        numero = limpar(data.get('numeroProcesso'))
        tema = limpar(data.get('tema'))
        data_dist = limpar(data.get('dataDistribuicao'))
        responsavel = limpar(data.get('responsavel'))
        status = limpar(data.get('status'))
        markdown = limpar(data.get('markdown'))
        comentarios = limpar(data.get('comentarios'))
        dat_base64 = data.get('dat')
        
        # DATA ATUAL AUTOMÁTICA - sempre seta data de hoje
        from datetime import datetime
        ultima_att = datetime.now().strftime('%Y-%m-%d')
        print(f"📅 Data de distribuição informada: {data_dist}")
        print(f"📅 Última atualização automática: {ultima_att}")
        
        print(f"📄 Processando processo: {numero}")
        
        suspeitos = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
        print(f"🔍 Suspeitos encontrados: {suspeitos}")

        if not numero:
            print("⚠️ Número do processo obrigatório")
            return jsonify({'error': 'Número do processo é obrigatório'}), 400

        nome_arquivo_base = numero.replace('/', '-')
        os.makedirs(PASTA_DESTINO, exist_ok=True)
        os.makedirs(PASTA_DAT, exist_ok=True)

        caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
        caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")

        # Salva markdown se fornecido
        if markdown and markdown.strip():
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"💾 Markdown salvo: {caminho_md}")

        # Salva .dat como base64 se enviado
        if dat_base64 and dat_base64.strip():
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            print(f"💾 Arquivo DAT salvo: {caminho_dat}")

        # Converte lista de suspeitos para string
        suspeitos_str = ', '.join(suspeitos) if suspeitos else ''

        nova_linha = (
            f"| {numero} "
            f"| {tema} "
            f"| {data_dist} "
            f"| {responsavel} "
            f"| {status} "
            f"| {ultima_att} "
            f"| {suspeitos_str} "
            f"| {comentarios} |\n"
        )

        if not os.path.exists(PATH_TRIAGEM):
            print(f"📝 Criando novo arquivo de triagem: {PATH_TRIAGEM}")
            with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
                f.write("# Tabela de Processos\n\n")
                f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
                f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")

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
                "|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n",
                nova_linha
            ]

        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.writelines(linhas)
        
        print(f"✅ Processo {numero} salvo com sucesso")
        return jsonify({"message": "Processo e arquivos salvos com sucesso"}), 201

    except Exception as e:
        print(f"❌ Erro em POST /triagem/form: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Correção para PUT /triagem/<numero>
@app.route('/triagem/<numero>', methods=['PUT'])
def editar_processo(numero):
    print(f"✏️ Solicitação PUT /triagem/{numero} recebida")
    try:
        data = request.get_json()
        print(f"📝 Dados recebidos: {data}")
        
        # Extrai processos existentes
        processos = extrair_tabela_md(PATH_TRIAGEM)
        
        # Encontra o processo existente para preservar suspeitos se necessário
        processo_existente = next((p for p in processos if p['numeroProcesso'] == numero), None)
        suspeitos_existentes = processo_existente.get('suspeitos', '') if processo_existente else ''
        
        # Remove o processo antigo da lista
        processos = [p for p in processos if p['numeroProcesso'] != numero]
        
        # DATA ATUAL AUTOMÁTICA - sempre seta data de hoje para última atualização
        from datetime import datetime
        ultima_att = datetime.now().strftime('%Y-%m-%d')
        print(f"📅 Última atualização automática: {ultima_att}")
        
        # Determina como lidar com suspeitos
        markdown = data.get('markdown', '')
        suspeitos_calculados = ''
        
        if markdown and markdown.strip():
            # Se há markdown novo, recalcula suspeitos
            try:
                suspeitos_lista = encontrar_suspeitos(markdown, './utils/suspeitos.txt')
                suspeitos_calculados = ', '.join(suspeitos_lista) if suspeitos_lista else ''
                print(f"🔍 Suspeitos recalculados: {suspeitos_calculados}")
            except Exception as e:
                print(f"⚠️ Erro ao calcular suspeitos: {e}")
                suspeitos_calculados = suspeitos_existentes
        else:
            # Se não há markdown, mantém suspeitos existentes
            suspeitos_calculados = suspeitos_existentes
            print(f"🔄 Mantendo suspeitos existentes: {suspeitos_calculados}")
        
        # Salva markdown atualizado se fornecido
        if markdown and markdown.strip():
            nome_arquivo_base = numero.replace('/', '-')
            os.makedirs(PASTA_DESTINO, exist_ok=True)
            caminho_md = os.path.join(PASTA_DESTINO, f"{nome_arquivo_base}.md")
            
            with open(caminho_md, 'w', encoding='utf-8') as f:
                f.write(markdown)
            print(f"💾 Markdown atualizado: {caminho_md}")
        
        # Salva arquivo DAT se fornecido
        dat_base64 = data.get('dat')
        if dat_base64 and dat_base64.strip():
            nome_arquivo_base = numero.replace('/', '-')
            os.makedirs(PASTA_DAT, exist_ok=True)
            caminho_dat = os.path.join(PASTA_DAT, f"{nome_arquivo_base}.dat")
            
            with open(caminho_dat, 'w', encoding='utf-8') as f:
                f.write(dat_base64)
            print(f"💾 Arquivo DAT atualizado: {caminho_dat}")
        
        # Cria o processo atualizado (SEMPRE usa data atual para última atualização)
        processo_atualizado = {
            "numeroProcesso": limpar(data['numeroProcesso']),
            "tema": limpar(data['tema']),
            "dataDistribuicao": limpar(data['dataDistribuicao']),  # Mantém a data original
            "responsavel": limpar(data['responsavel']),
            "status": limpar(data['status']),
            "ultimaAtualizacao": ultima_att,  # SEMPRE data atual
            "suspeitos": suspeitos_calculados,
            "comentarios": limpar(data.get('comentarios', ''))
        }
        
        # Adiciona o processo atualizado à lista
        processos.append(processo_atualizado)
        
        # Reescreve o arquivo de triagem
        with open(PATH_TRIAGEM, 'w', encoding='utf-8') as f:
            f.write("# Tabela de Processos\n\n")
            f.write("| Nº Processo | Tema | Data da Distribuição | Responsável | Status | Última Atualização | Suspeitos | Comentários |\n")
            f.write("|-------------|------|-----------------------|-------------|--------|----------------------|-----------|-------------|\n")
            for p in processos:
                f.write(
                    f"| {p['numeroProcesso']} | {p['tema']} | {p['dataDistribuicao']} | {p['responsavel']} "
                    f"| {p['status']} | {p['ultimaAtualizacao']} | {p['suspeitos']} | {p.get('comentarios', '')} |\n"
                )

        print(f"✅ Processo {numero} atualizado com sucesso")
        return jsonify({"message": "Processo atualizado com sucesso"}), 200

    except KeyError as e:
        print(f"❌ Campo obrigatório ausente em PUT /triagem/{numero}: {str(e)}")
        return jsonify({'error': f'Campo obrigatório ausente: {str(e)}'}), 400
    except Exception as e:
        print(f"❌ Erro em PUT /triagem/{numero}: {str(e)}")
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
