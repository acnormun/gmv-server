import numpy as np
import torch
import os
from typing import List, Dict, Optional, Tuple, Union
import json
import logging
import time
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MatryoshkaEmbedding:
    """
    Implementação de Matryoshka Representation Learning para embeddings adaptativos.
    As técnicas Matryoshka criam embeddings "aninhados" onde as primeiras dimensões
    contêm informação hierárquica, permitindo uso adaptativo baseado na complexidade.
    """
    def __init__(self,
                 model_name: str = os.path.join("models", "paraphrase-MiniLM-L3-v2"),
                 matryoshka_dims: Optional[List[int]] = None,
                 device: str = "auto"):
        self.model_name = model_name
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        try:
            self.model = SentenceTransformer(model_name, device=str(self.device))
            self.model.eval()
            logger.info(f"Modelo {model_name} carregado no dispositivo {self.device}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {model_name}: {e}")
            raise
        if matryoshka_dims is None:
            base_dim = self.model.get_sentence_embedding_dimension()
            self.matryoshka_dims = []
            dim = 16
            while dim <= base_dim:
                self.matryoshka_dims.append(dim)
                dim *= 2
            if self.matryoshka_dims[-1] != base_dim:
                self.matryoshka_dims.append(base_dim)
        else:
            self.matryoshka_dims = sorted(matryoshka_dims)
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info(f"Dimensões Matryoshka: {self.matryoshka_dims}")
    
    def _get_cache_key(self, texts: List[str], target_dims: List[int]) -> str:
        text_hash = hash(tuple(texts))
        dims_hash = hash(tuple(sorted(target_dims)))
        return f"{text_hash}_{dims_hash}"
    
    def encode_matryoshka(self, 
                         texts: Union[str, List[str]], 
                         target_dims: Optional[List[int]] = None,
                         normalize: bool = True,
                         use_cache: bool = True) -> Dict[int, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]
        if target_dims is None:
            target_dims = self.matryoshka_dims.copy()
        else:
            target_dims = [d for d in target_dims if d <= max(self.matryoshka_dims)]
        cache_key = self._get_cache_key(texts, target_dims) if use_cache else None
        if use_cache and cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]
        self.cache_misses += 1
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
            )
        except Exception as e:
            logger.error(f"Erro na geração de embeddings: {e}")
            raise
        matryoshka_embeddings = {}
        for dim in target_dims:
            if dim <= embeddings.shape[1]:
                sub_embedding = embeddings[:, :dim].cpu().numpy()
                if normalize and dim < embeddings.shape[1]:
                    norms = np.linalg.norm(sub_embedding, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1, norms)
                    sub_embedding = sub_embedding / norms
                matryoshka_embeddings[dim] = sub_embedding
            else:
                logger.warning(f"Dimensão {dim} maior que embedding base {embeddings.shape[1]}")
        if use_cache and cache_key:
            self.embedding_cache[cache_key] = matryoshka_embeddings
        return matryoshka_embeddings
    
    def adaptive_search(self, 
                       query: str, 
                       documents: List[str], 
                       threshold_config: Optional[Dict[int, float]] = None,
                       max_candidates: int = 100) -> List[Tuple[int, float, int]]:
        if threshold_config is None:
            threshold_config = {
                32: 0.3,
                64: 0.5, 
                128: 0.7,
                256: 0.8,
                384: 0.85
            }
        available_dims = [d for d in self.matryoshka_dims if d in threshold_config]
        available_dims.sort()
        query_embeddings = self.encode_matryoshka([query], available_dims)
        doc_embeddings = self.encode_matryoshka(documents, available_dims)
        candidates = list(range(len(documents)))
        final_scores = {}
        dimension_used = {}
        for dim in available_dims:
            if not candidates:
                break
            query_emb = query_embeddings[dim][0]
            doc_embs = doc_embeddings[dim]
            candidate_similarities = []
            for candidate_idx in candidates:
                similarity = np.dot(doc_embs[candidate_idx], query_emb)
                candidate_similarities.append((candidate_idx, similarity))
            threshold = threshold_config.get(dim, 0.5)
            new_candidates = []
            for candidate_idx, score in candidate_similarities:
                if score >= threshold:
                    new_candidates.append(candidate_idx)
                    final_scores[candidate_idx] = score
                    dimension_used[candidate_idx] = dim
            candidates = new_candidates
            if len(candidates) <= max_candidates // 4 or dim == available_dims[-1]:
                break
        results = []
        for doc_idx, score in final_scores.items():
            dim = dimension_used[doc_idx]
            results.append((doc_idx, score, dim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_optimal_dimension(self, text: str) -> int:
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        avg_word_length = char_count / max(1, word_count)
        avg_sentence_length = word_count / sentence_count
        complexity_score = 0
        if word_count > 100:
            complexity_score += 3
        elif word_count > 30:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        if avg_sentence_length > 20:
            complexity_score += 2
        elif avg_sentence_length > 10:
            complexity_score += 1
        if avg_word_length > 6:
            complexity_score += 1
        if complexity_score <= 1:
            return min(64, max(self.matryoshka_dims))
        elif complexity_score <= 3:
            return min(128, max(self.matryoshka_dims))
        elif complexity_score <= 5:
            return min(256, max(self.matryoshka_dims))
        else:
            return max(self.matryoshka_dims)
    
    def benchmark_dimensions(self, 
                           test_queries: List[str], 
                           test_docs: List[str],
                           runs: int = 3) -> Dict[int, Dict[str, float]]:
        results = {}
        for dim in self.matryoshka_dims:
            dim_results = {
                'encoding_time': [],
                'memory_usage': 0,
                'query_similarities': [],
                'doc_similarities': []
            }
            for run in range(runs):
                start_time = time.time()
                query_embeddings = self.encode_matryoshka(test_queries, [dim])
                doc_embeddings = self.encode_matryoshka(test_docs, [dim])
                encoding_time = time.time() - start_time
                dim_results['encoding_time'].append(encoding_time)
                query_embs = query_embeddings[dim]
                doc_embs = doc_embeddings[dim]
                similarities = np.dot(query_embs, doc_embs.T)
                dim_results['query_similarities'].extend(similarities.flatten())
                doc_similarities = np.dot(doc_embs, doc_embs.T)
                mask = np.triu(np.ones_like(doc_similarities, dtype=bool), k=1)
                dim_results['doc_similarities'].extend(doc_similarities[mask])
            dim_results['memory_usage'] = query_embeddings[dim].nbytes + doc_embeddings[dim].nbytes
            results[dim] = {
                'avg_encoding_time': np.mean(dim_results['encoding_time']),
                'std_encoding_time': np.std(dim_results['encoding_time']),
                'memory_usage_bytes': dim_results['memory_usage'],
                'memory_usage_mb': dim_results['memory_usage'] / (1024 * 1024),
                'avg_query_similarity': np.mean(dim_results['query_similarities']),
                'std_query_similarity': np.std(dim_results['query_similarities']),
                'avg_doc_diversity': 1 - np.mean(dim_results['doc_similarities']),
                'efficiency_score': dim / (np.mean(dim_results['encoding_time']) * 1000)
            }
        return results
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.embedding_cache)
        }
    
    def clear_cache(self):
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache de embeddings limpo")
    
    def save_embeddings(self, filepath: str, embeddings_dict: Dict[int, np.ndarray]):
        try:
            save_data = {
                'model_name': self.model_name,
                'matryoshka_dims': self.matryoshka_dims,
                'embeddings': {str(k): v.tolist() for k, v in embeddings_dict.items()}
            }
            with open(filepath, 'w') as f:
                json.dump(save_data, f)
            logger.info(f"Embeddings salvos em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar embeddings: {e}")
            raise
    
    def load_embeddings(self, filepath: str) -> Dict[int, np.ndarray]:
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            if save_data['model_name'] != self.model_name:
                logger.warning(f"Modelo diferente: {save_data['model_name']} vs {self.model_name}")
            embeddings_dict = {
                int(k): np.array(v) for k, v in save_data['embeddings'].items()
            }
            logger.info(f"Embeddings carregados de {filepath}")
            return embeddings_dict
        except Exception as e:
            logger.error(f"Erro ao carregar embeddings: {e}")
            raise

def create_matryoshka_model(model_name: str = os.path.join("models", "paraphrase-MiniLM-L3-v2"),
                           custom_dims: Optional[List[int]] = None) -> MatryoshkaEmbedding:
    return MatryoshkaEmbedding(
        model_name=model_name,
        matryoshka_dims=custom_dims
    )

MATRYOSHKA_CONFIGS = {
    'fast': {
        'model_name': os.path.join('models', 'paraphrase-MiniLM-L3-v2'),
        'dimensions': [32, 64, 128],
        'thresholds': {32: 0.4, 64: 0.6, 128: 0.8}
    },
    'balanced': {
        'model_name': os.path.join('models', 'paraphrase-MiniLM-L3-v2'), 
        'dimensions': [32, 64, 128, 256],
        'thresholds': {32: 0.3, 64: 0.5, 128: 0.7, 256: 0.8}
    },
    'accurate': {
        'model_name': os.path.join('models', 'paraphrase-MiniLM-L3-v2'),
        'dimensions': [64, 128, 256, 384, 768],
        'thresholds': {64: 0.2, 128: 0.4, 256: 0.6, 384: 0.7, 768: 0.8}
    },
    'multilingual': {
        'model_name': os.path.join('models', 'paraphrase-multilingual-MiniLM-L12-v2'),
        'dimensions': [32, 64, 128, 256, 384],
        'thresholds': {32: 0.3, 64: 0.5, 128: 0.7, 256: 0.8, 384: 0.85}
    }
}