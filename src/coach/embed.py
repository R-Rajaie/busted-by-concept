"""Local embeddings and reranking via fastembed (ONNX, CPU)."""

from functools import cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "BAAI/bge-reranker-base"
# bge models expect this prefix on queries (not on documents) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL)


@cache
def _reranker() -> TextCrossEncoder:
    return TextCrossEncoder(RERANK_MODEL)


def embed(texts: list[str]) -> np.ndarray:
    """Unit-normalized document embeddings, shape (n, EMBED_DIM)."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    vecs = np.array(list(_embedder().embed(texts)), dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def embed_query(query: str) -> np.ndarray:
    return embed([QUERY_PREFIX + query])[0]


def rerank(query: str, texts: list[str]) -> list[float]:
    """Relevance in (0, 1) for each text."""
    if not texts:
        return []
    scores = np.array(list(_reranker().rerank(query, texts)), dtype=np.float32)
    return (1 / (1 + np.exp(-scores))).tolist()
