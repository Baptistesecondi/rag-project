"""Unit tests for chunking and BM25 retrieval. Run with: python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluate import hit_rank  # noqa: E402
from generate import build_prompt, extract_text, extractive_answer  
from ingest import Chunk, build_chunks, chunk_text  # noqa: E402
from retrieve import BM25Index, tokenize  # noqa: E402


class ChunkTextTests(unittest.TestCase):
    def test_overlapping_chunks(self):
        text = " ".join(f"w{i}" for i in range(10))
        chunks = chunk_text(text, size=4, overlap=1)
        self.assertEqual(chunks[0], "w0 w1 w2 w3")
        self.assertEqual(chunks[1], "w3 w4 w5 w6")
        self.assertEqual(chunks[-1], "w6 w7 w8 w9")
        self.assertEqual(len(chunks), 3)

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(chunk_text(" "), [])

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            chunk_text("a b c", size=3, overlap=3)

    def test_build_chunks_keeps_source(self):
        chunks = build_chunks({"a.md": "one two three"}, size=2, overlap=0)
        self.assertEqual([c.doc_id for c in chunks], ["a.md", "a.md"])
        self.assertEqual([c.chunk_id for c in chunks], [0, 1])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("fruit.md", 0, "apples are red and sweet fruit"),
            Chunk("markets.md", 0, "stock prices rose sharply after the report"),
            Chunk("cooking.md", 0, "bake the bread until golden brown"),
        ]
        self.index = BM25Index(self.chunks)

    def test_most_relevant_chunk_ranks_first(self):
        hits = self.index.search("stock prices", top_k=2)
        self.assertEqual(hits[0][0].doc_id, "markets.md")

    def test_no_match_returns_nothing(self):
        self.assertEqual(self.index.search("quantum chromodynamics"), [])

    def test_empty_index_does_not_crash(self):
        self.assertEqual(BM25Index([]).search("anything"), [])

    def test_top_k_limits_results(self):
        self.assertLessEqual(len(self.index.search("red sweet bread golden", top_k=1)), 1)

    def test_stop_words_are_ignored(self):
        self.assertEqual(tokenize("What is the threat to the homeland?"), ["threat", "homeland"])

    def test_query_of_only_stop_words_returns_nothing(self):
        self.assertEqual(self.index.search("what is the"), [])

    def test_min_score_filters_weak_hits(self):
        all_hits = self.index.search("stock bread", top_k=3)
        best = max(score for _, score in all_hits)
        filtered = self.index.search("stock bread", top_k=3, min_score=best - 1e-9)
        self.assertEqual(len(filtered), 1)


class EvaluateTests(unittest.TestCase):
    def test_hit_rank_finds_first_matching_document(self):
        hits = [(Chunk("uk/a.txt", 0, "x"), 2.0), (Chunk("fr/b.txt", 0, "y"), 1.0)]
        self.assertEqual(hit_rank(hits, ["b.txt"]), 2)
        self.assertIsNone(hit_rank(hits, ["c.txt"]))


class PromptTests(unittest.TestCase):
    def test_prompt_contains_numbered_sources_and_question(self):
        hits = [(Chunk("a.md", 0, "the sky is blue"), 1.0)]
        prompt = build_prompt("What colour is the sky?", hits)
        self.assertIn("[1] (a.md) the sky is blue", prompt)
        self.assertIn("Question: What colour is the sky?", prompt)

    def test_extract_text_skips_thinking_blocks(self):
        from types import SimpleNamespace as Block

        content = [Block(type="thinking", thinking="..."), Block(type="text", text="China [1].")]
        self.assertEqual(extract_text(content), "China [1].")

    def test_extractive_fallback_shows_passages(self):
        hits = [(Chunk("a.md", 0, "the sky is blue"), 1.0)]
        self.assertIn("the sky is blue", extractive_answer(hits))


if __name__ == "__main__":
    unittest.main()