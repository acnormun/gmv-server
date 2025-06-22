# gmv-server

Aplicação Flask para triagem de processos e consulta via RAG.

## Variáveis de ambiente
- `PATH_TRIAGEM`
- `PASTA_DESTINO`
- `PASTA_DAT`
- `RAG_DB_PATH` – pasta onde ficam os documentos base para a RAG.

## Endpoints principais
- `GET /health`
- `POST /triagem/form`
- `GET /rag/status`
- `GET /rag/suggestions`
- `POST /rag/query` (JSON: `{ "question": "..." }`)
