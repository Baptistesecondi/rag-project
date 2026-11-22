"""Measure retrieval quality on a set of questions with known source documents.

Run with: python src/evaluate.py              (BM25, dense and hybrid)
     or:  python src/evaluate.py --method bm25
"""
import argparse
import json
from pathlib import Path

from ingest import build_chunks, load_documents
from retrieve import BM25Index


def hit_rank(hits, expected: list[str]) -> int | None:
    """Return the 1-based rank of the first hit whose document matches, or None."""
    for rank, (chunk, _) in enumerate(hits, start=1):
        if any(name in chunk.doc_id for name in expected):
            return rank
    return None


def evaluate(index, questions: list[dict], top_k: int, min_score: float | None = None) -> dict:
    kwargs = {} if min_score is None else {"min_score": min_score}
    ranks = [
        hit_rank(index.search(q["question"], top_k=top_k, **kwargs), q["expected"])
        for q in questions
    ]
    found = [r for r in ranks if r is not None]
    return {
        "ranks": ranks,
        "recall": len(found) / len(questions),
        "mrr": sum(1 / r for r in found) / len(questions),
    }


def build_indexes(chunks, method: str) -> dict:
    indexes = {}
    bm25 = BM25Index(chunks)
    if method in ("bm25", "all"):
        indexes["bm25"] = bm25
    if method in ("dense", "hybrid", "all"):
        from dense import DenseIndex
        from hybrid import HybridIndex

        dense = DenseIndex(chunks)
        if method in ("dense", "all"):
            indexes["dense"] = dense
        if method in ("hybrid", "all"):
            indexes["hybrid"] = HybridIndex(bm25, dense)
    return indexes


def threshold_sweep(index: BM25Index, questions, off_topic, top_k: int) -> None:
    print("\nBM25 threshold calibration:")
    print("  min_score  recall  off-topic questions rejected")
    for threshold in [0, 5, 8, 10, 12, 15]:
        recall = evaluate(index, questions, top_k, threshold)["recall"]
        rejected = sum(not index.search(q, top_k=1, min_score=threshold) for q in off_topic)
        print(f"  {threshold:>9}  {recall:>6.0%}  {rejected}/{len(off_topic)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on known questions.")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--questions", default="data/eval/questions.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "all"], default="all")
    args = parser.parse_args()

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions, off_topic = spec["questions"], spec["off_topic"]
    chunks = build_chunks(load_documents(args.data))
    indexes = build_indexes(chunks, args.method)
    results = {name: evaluate(index, questions, args.top_k) for name, index in indexes.items()}

    names = list(results)
    print(f"Rank of the expected document (top {args.top_k}, '-' = missed):")
    print("  " + "".join(f"{n:>8}" for n in names) + "  question")
    for i, q in enumerate(questions):
        cells = "".join(f"{results[n]['ranks'][i] or '-':>8}" for n in names)
        print(f"  {cells}  {q['question']}")

    print(f"\n  {'method':<8} Recall@{args.top_k}   MRR")
    for n in names:
        print(f"  {n:<8} {results[n]['recall']:>8.0%}  {results[n]['mrr']:.2f}")

    if "bm25" in indexes:
        threshold_sweep(indexes["bm25"], questions, off_topic, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
