import json
import requests
import os
import base64
import urllib3
import zipfile
import shutil
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Config
LOCAL_VERSION_FILE = 'version.json'
REPO_OWNER = 'acnormun'
REPO_NAME = 'gmv-server'
FILE_PATH = 'version.json'
BRANCH = 'main'
TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}"}
IGNORAR_ARQUIVOS = ['.env', 'version.json', 'check_version.py', 'update_temp']

def ler_versao_local():
    if not os.path.exists(LOCAL_VERSION_FILE):
        return None
    with open(LOCAL_VERSION_FILE, 'r', encoding='utf-8') as f:
        return json.load(f).get("version")

def buscar_versao_remota():
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{FILE_PATH}"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        if res.status_code == 200:
            content = res.json().get('content', '')
            decoded_json = json.loads(base64.b64decode(content).decode('utf-8'))
            return decoded_json.get("version")
        else:
            print("Erro:", res.status_code, res.text)
    except Exception as e:
        print("Erro ao verificar versão online:", e)
    return None

def comparar_versoes(local, remoto):
    if not local or not remoto:
        return "Versão local ou remota inválida."
    if local == remoto:
        return f" App está atualizado. Versão: {local}"
    return f"⬆️ Atualização disponível: local {local} → nova {remoto}"

def baixar_zip_do_repositorio():
    zip_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/zipball/{BRANCH}"
    print("🔄 Baixando atualização...")
    try:
        res = requests.get(zip_url, headers=HEADERS, stream=True, verify=False)
        if res.status_code == 200:
            with zipfile.ZipFile(BytesIO(res.content)) as z:
                z.extractall("update_temp/")
            print(" Atualização baixada e extraída para a pasta 'update_temp/'")
        else:
            print("❌ Falha ao baixar atualização:", res.status_code, res.text)
    except Exception as e:
        print("❌ Erro ao baixar o ZIP:", e)

def aplicar_atualizacao():
    subpastas = [nome for nome in os.listdir("update_temp") if os.path.isdir(os.path.join("update_temp", nome))]
    if not subpastas:
        print("❌ Estrutura de atualização inválida.")
        return

    nova_pasta = os.path.join("update_temp", subpastas[0])

    for item in os.listdir(nova_pasta):
        if item in IGNORAR_ARQUIVOS:
            continue
        src = os.path.join(nova_pasta, item)
        dst = os.path.join(".", item)

        # Remove antigo
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)

        # Copia novo
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    shutil.rmtree("update_temp")
    print(" Atualização aplicada com sucesso.")

if __name__ == '__main__':
    local = ler_versao_local()
    remoto = buscar_versao_remota()
    resultado = comparar_versoes(local, remoto)
    print(resultado)

    if "Atualização disponível" in resultado:
        baixar_zip_do_repositorio()
        aplicar_atualizacao()

        # Atualiza o version.json local
        with open(LOCAL_VERSION_FILE, 'w', encoding='utf-8') as f:
            json.dump({"version": remoto}, f, indent=2)
        print(f"🆕 Versão atualizada para {remoto}")
