"""Command-line entry point: index the documents, retrieve passages, answer the question."""
import argparse
import sys

from generate import answer
from ingest import build_chunks, load_documents
from retrieve import BM25Index


def build_index(chunks, method: str):
    bm25 = BM25Index(chunks)
    if method == "bm25":
        return bm25
    from dense import DenseIndex

    dense = DenseIndex(chunks)
    if method == "dense":
        return dense
    from hybrid import HybridIndex

    return HybridIndex(bm25, dense)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about your documents.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--data", default="data/raw", help="Folder with .txt or .md files")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve")
    parser.add_argument(
        "--min-score", type=float, default=8.0, help="BM25 only: ignore passages scoring below this"
    )
    parser.add_argument(
        "--method", choices=["bm25", "dense", "hybrid"], default="bm25", help="Retrieval method"
    )
    args = parser.parse_args(argv)

    docs = load_documents(args.data)
    if not docs:
        print(f"No .txt or .md files found in {args.data}", file=sys.stderr)
        return 1

    index = build_index(build_chunks(docs), args.method)
    if args.method == "bm25":
        hits = index.search(args.question, top_k=args.top_k, min_score=args.min_score)
    else:
        hits = index.search(args.question, top_k=args.top_k)
    if not hits:
        print("No relevant passage found for this question.")
        return 0

    print(answer(args.question, hits))
    print("\nSources:")
    for i, (chunk, score) in enumerate(hits, start=1):
        print(f"[{i}] {chunk.doc_id} (chunk {chunk.chunk_id}, score {score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
