import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency
    ollama = None

class AdaptiveRAG:
    """Implementação simples de RAG usando embeddings locais e Ollama."""

    def __init__(self, data_path: str, model_name: str = "llama3", embed_model: str = "all-MiniLM-L6-v2") -> None:
        self.data_path = data_path
        self.model_name = model_name
        self.embed_model = SentenceTransformer(embed_model)
        self.docs: List[dict] = []
        self.embeddings: np.ndarray | None = None
        self.load_documents()

    def load_documents(self) -> None:
        texts = []
        self.docs = []
        for root, _, files in os.walk(self.data_path):
            for name in files:
                if name.endswith('.txt') or name.endswith('.md'):
                    path = os.path.join(root, name)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        self.docs.append({'path': path, 'text': text})
                        texts.append(text)
                    except Exception:
                        continue
        if texts:
            self.embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        else:
            dim = self.embed_model.get_sentence_embedding_dimension()
            self.embeddings = np.zeros((0, dim))

    def query(self, question: str, top_k: int = 3) -> List[dict]:
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        q_emb = self.embed_model.encode([question], convert_to_numpy=True)
        scores = cosine_similarity(q_emb, self.embeddings)[0]
        idxs = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idxs:
            results.append({'path': self.docs[i]['path'], 'text': self.docs[i]['text'], 'score': float(scores[i])})
        return results

    def generate(self, question: str, context: str) -> str:
        prompt = f"Use o contexto a seguir para responder a pergunta:\n{context}\nPergunta: {question}"
        if ollama is None:
            return "Ollama não disponível"
        try:
            resp = ollama.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}])
            return resp.get('message', {}).get('content', '')
        except Exception as e:  # pragma: no cover - ambiente pode não ter ollama
            return f"Erro ao gerar resposta: {e}"

    def chat(self, question: str, top_k: int = 3) -> dict:
        docs = self.query(question, top_k)
        contexto = "\n\n".join(d['text'] for d in docs)
        answer = self.generate(question, contexto)
        return {
            'answer': answer,
            'references': [{'path': d['path'], 'score': d['score']} for d in docs]
        }
