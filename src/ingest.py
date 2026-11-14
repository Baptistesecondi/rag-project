"""Load documents from a folder and split them into overlapping word chunks."""
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def load_documents(folder: str | Path) -> dict[str, str]:
    """Return {relative_path: text} for every supported file under folder."""
    folder = Path(folder)
    docs: dict[str, str] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs[str(path.relative_to(folder))] = path.read_text(encoding="utf-8")
    return docs


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into chunks of `size` words, each overlapping the previous by `overlap` words."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(docs: dict[str, str], size: int = 200, overlap: int = 40) -> list[Chunk]:
    """Chunk every document, keeping track of its source."""
    return [
        Chunk(doc_id=doc_id, chunk_id=i, text=piece)
        for doc_id, text in docs.items()
        for i, piece in enumerate(chunk_text(text, size, overlap))
    ]
