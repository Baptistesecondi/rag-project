"""Command-line entry point: index the documents, retrieve passages, answer the question."""
import argparse
import sys

from generate import answer
from ingest import build_chunks, load_documents
from retrieve import BM25Index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about your documents.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--data", default="data/raw", help="Folder with .txt or .md files")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve")
    parser.add_argument("--min-score", type=float, default=8.0, help="Ignore passages scoring below this")
    args = parser.parse_args(argv)

    docs = load_documents(args.data)
    if not docs:
        print(f"No .txt or .md files found in {args.data}", file=sys.stderr)
        return 1

    index = BM25Index(build_chunks(docs))
    hits = index.search(args.question, top_k=args.top_k, min_score=args.min_score)
    if not hits:
        print("No relevant passage found for this question.")
        return 0

    print(answer(args.question, hits))
    print("\nSources:")
    for i, (chunk, score) in enumerate(hits, start=1):
        print(f"[{i}] {chunk.doc_id} (chunk {chunk.chunk_id}, score {score:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
