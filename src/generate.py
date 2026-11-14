"""Build a grounded prompt and call an LLM, with a fallback when no API key is set."""
import os

from ingest import Chunk

DEFAULT_MODEL = "claude-haiku-5-5"

PROMPT_TEMPLATE = """Answer the question using only the sources below.
Cite the sources you use as [n]. If the sources do not contain the answer, say so.

Sources:
{sources}

Question: {question}
"""


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    sources = "\n\n".join(
        f"[{i}] ({chunk.doc_id}) {chunk.text}" for i, (chunk, _) in enumerate(hits, start=1)
    )
    return PROMPT_TEMPLATE.format(sources=sources, question=question)


def extractive_answer(hits: list[tuple[Chunk, float]]) -> str:
    """Fallback without an LLM: show the passages that were retrieved."""
    lines = ["No ANTHROPIC_API_KEY set, so showing the most relevant passages:"]
    for i, (chunk, _) in enumerate(hits, start=1):
        lines.append(f"\n[{i}] {chunk.text}")
    return "\n".join(lines)


def answer(question: str, hits: list[tuple[Chunk, float]], model: str | None = None) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return extractive_answer(hits)

    import anthropic  # imported lazily so the project runs without the package

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model or os.environ.get("RAG_MODEL", DEFAULT_MODEL),
        max_tokens=500,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return response.content[0].text
