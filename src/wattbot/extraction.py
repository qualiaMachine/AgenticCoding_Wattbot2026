from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    document_id: str
    page: int
    chunk_index: int
    text: str


def extract_corpus(corpus_dir: Path, extraction_dir: Path, chunk_words: int = 350, overlap_words: int = 50) -> list[Chunk]:
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    chunks: list[Chunk] = []
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = " ".join((page.extract_text() or "").split())
            chunks.extend(_chunk_page(pdf_path.stem, page_number, text, chunk_words, overlap_words))
    extraction_dir.mkdir(parents=True, exist_ok=True)
    output = extraction_dir / "chunks.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(path: Path) -> list[Chunk]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunk file was not found: {path}. Run extract first.")
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _chunk_page(document_id: str, page: int, text: str, size: int, overlap: int) -> list[Chunk]:
    words = text.split()
    chunks: list[Chunk] = []
    for start in range(0, len(words), size - overlap):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(Chunk(document_id, page, len(chunks), " ".join(window)))
        if start + size >= len(words):
            break
    return chunks

