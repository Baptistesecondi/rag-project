"""Measure retrieval quality on a set of questions with known source documents.

Run with: python src/evaluate.py
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


def evaluate(index: BM25Index, questions: list[dict], top_k: int, min_score: float) -> dict:
    ranks = [
        hit_rank(index.search(q["question"], top_k=top_k, min_score=min_score), q["expected"])
        for q in questions
    ]
    found = [r for r in ranks if r is not None]
    return {
        "ranks": ranks,
        "recall": len(found) / len(questions),
        "mrr": sum(1 / r for r in found) / len(questions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on known questions.")
    parser.add_argument("--data", default="data/raw")
    parser.add_argument("--questions", default="data/eval/questions.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    spec = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions, off_topic = spec["questions"], spec["off_topic"]
    index = BM25Index(build_chunks(load_documents(args.data)))

    result = evaluate(index, questions, args.top_k, min_score=0.0)
    print(f"Per question (top {args.top_k}, no threshold):")
    for q, rank in zip(questions, result["ranks"]):
        status = f"rank {rank}" if rank else "MISSED"
        print(f"  [{status:>7}] {q['question']}")
    print(f"\nRecall@{args.top_k}: {result['recall']:.0%}   MRR: {result['mrr']:.2f}")

    print("\nThreshold calibration:")
    print("  min_score  recall  off-topic questions rejected")
    for threshold in [0, 5, 8, 10, 12, 15]:
        recall = evaluate(index, questions, args.top_k, threshold)["recall"]
        rejected = sum(not index.search(q, top_k=1, min_score=threshold) for q in off_topic)
        print(f"  {threshold:>9}  {recall:>6.0%}  {rejected}/{len(off_topic)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())