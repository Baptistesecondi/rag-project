"""A small BM25 index, written without external dependencies."""
import math
import re
from collections import Counter

from ingest import Chunk
from stopwords import STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words, without stop words."""
    tokens = (token.lower() for token in TOKEN_PATTERN.findall(text))
    return [t for t in tokens if t not in STOP_WORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.n = len(self.chunks)

        self.term_freqs = [Counter(tokenize(c.text)) for c in self.chunks]
        self.doc_lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_doc_length = (sum(self.doc_lengths) / self.n) if self.n else 1.0

        self.doc_freqs: Counter = Counter()
        for tf in self.term_freqs:
            self.doc_freqs.update(tf.keys())

    def _idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, top_k: int = 3, min_score: float = 0.0
    ) -> list[tuple[Chunk, float]]:
        """Return the top_k (chunk, score) pairs scoring above min_score, best first."""
        query_terms = set(tokenize(query))
        scored = []
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                length_norm = 1 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_length
                score += self._idf(term) * freq * (self.k1 + 1) / (freq + self.k1 * length_norm)
            if score > min_score:
                scored.append((score, i))

        scored.sort(reverse=True)
        return [(self.chunks[i], score) for score, i in scored[:top_k]]