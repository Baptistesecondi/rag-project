"""Hybrid retrieval: fuse the BM25 and dense rankings with Reciprocal Rank Fusion (RRF)."""
from ingest import Chunk


class HybridIndex:
    def __init__(self, bm25, dense, candidates: int = 50, k: int = 60):
        """candidates: how many results to take from each method; k: RRF smoothing constant."""
        self.bm25 = bm25
        self.dense = dense
        self.candidates = candidates
        self.k = k

    def search(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        """Each chunk scores sum(1 / (k + rank)) over the rankings it appears in."""
        fused: dict[tuple[str, int], float] = {}
        chunks: dict[tuple[str, int], Chunk] = {}
        for index in (self.bm25, self.dense):
            for rank, (chunk, _) in enumerate(index.search(query, top_k=self.candidates), start=1):
                key = (chunk.doc_id, chunk.chunk_id)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.k + rank)
                chunks[key] = chunk
        best = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(chunks[key], score) for key, score in best if score > min_score]
