# GMV Server

Este projeto fornece uma API Flask utilizada no sistema de Gerenciamento de Monitoramento de Valores (GMV). O servidor oferece rotas HTTP e suporte a WebSocket para processamento de documentos, anonimização e integração com um sistema de RAG (Retrieval Augmented Generation).

## Requisitos

- Python 3.10 ou superior
- Dependências listadas em `requirements.txt`

Instale-as com:

```bash
pip install -r requirements.txt
```

## Configuração

O script `utils/auto_setup.py` cria e carrega automaticamente o arquivo `.env` com as variáveis necessárias:

- `PATH_TRIAGEM` – caminho do arquivo principal de triagem (markdown)
- `PASTA_DESTINO` – pasta onde os markdowns processados são gravados
- `PASTA_DAT` – pasta onde ficam os arquivos `.dat`

Execute o servidor com:

```bash
python app.py
```

O servidor inicia em `http://0.0.0.0:5000`.

## Principais Endpoints

| Método e Rota                  | Descrição |
|--------------------------------|-----------|
| `GET /health`                  | Verifica se o serviço está online |
| `GET /process-info`            | Informações de processo e variáveis de ambiente |
| `GET /triagem`                 | Lista processos registrados |
| `POST /triagem/form`           | Submete um novo processo para análise |
| `GET /triagem/status`          | Acompanha operações em andamento |
| `GET /triagem/<numero>/dat`    | Baixa o arquivo `.dat` de um processo |
| `PUT /triagem/<numero>`        | Atualiza metadados de um processo |
| `DELETE /triagem/<numero>`     | Remove um processo |
| `GET /api/rag/status`          | Status do subsistema RAG |
| `POST /api/rag/init`           | Inicializa o RAG e carrega documentos |
| `POST /api/rag/query`          | Consulta o RAG com pergunta livre |
| `POST /api/rag/query-with-context` | Consulta o RAG apenas em processos específicos |
| `POST /api/rag/reload`         | Recarrega documentos de triagem |
| `GET /api/system/stats`        | Estatísticas de uso do servidor |
| `GET /api/system/health`       | Verificação de saúde básica |

Eventos WebSocket adicionais são utilizados para acompanhar progresso de operações longas.

## Logs

Todas as ações são registradas em `server.log`. A configuração padrão grava logs em arquivo e no console.

## Versão

A versão atual da aplicação é `1.0.0`, definida em `version.json`.

## Autor

Desenvolvido por :honeybee: **ACNormun**