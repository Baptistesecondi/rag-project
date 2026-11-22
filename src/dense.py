"""Dense retrieval: embed chunks with a sentence-embedding model and rank them by cosine similarity."""
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from ingest import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on queries (not on passages) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

Encoder = Callable[[list[str]], np.ndarray]


def load_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    """Return a function that turns texts into L2-normalised vectors."""
    from sentence_transformers import SentenceTransformer  # imported lazily: optional dependency

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 100
        )

    return encode


class DenseIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        encoder: Encoder | None = None,
        model_name: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = "data/cache",
        query_prefix: str = QUERY_PREFIX,
    ):
        self.chunks = list(chunks)
        self.encode = encoder or load_encoder(model_name)
        self.query_prefix = query_prefix
        self.vectors = self._embed_chunks(model_name, cache_dir)

    def _embed_chunks(self, model_name: str, cache_dir: str | Path | None) -> np.ndarray:
        if not self.chunks:
            return np.zeros((0, 0))
        texts = [c.text for c in self.chunks]
        if cache_dir is None:
            return np.asarray(self.encode(texts))

        # The cache key depends on the model and on every chunk, so it changes with the corpus.
        digest = hashlib.sha256(model_name.encode())
        for c in self.chunks:
            digest.update(f"{c.doc_id}\0{c.chunk_id}\0{c.text}\0".encode())
        path = Path(cache_dir) / f"{model_name.replace('/', '_')}_{digest.hexdigest()[:16]}.npy"
        if path.exists():
            return np.load(path)
        vectors = np.asarray(self.encode(texts))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors)
        return vectors

    def search(
        self, query: str, top_k: int = 3, min_score: float = -1.0
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k (chunk, cosine similarity) pairs above min_score, best first."""
        if not self.chunks:
            return []
        query_vector = np.asarray(self.encode([self.query_prefix + query]))[0]
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > min_score]
