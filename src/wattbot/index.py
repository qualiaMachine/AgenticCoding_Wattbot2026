from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import Settings, require_model_settings
from .extraction import Chunk


def build_index(chunks: list[Chunk], settings: Settings, batch_size: int = 32) -> bool:
    if not chunks:
        raise ValueError("No chunks to embed")
    serialized_chunks = json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, sort_keys=True)
    content_hash = hashlib.sha256(serialized_chunks.encode("utf-8")).hexdigest()
    manifest_path = settings.index_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("content_hash") == content_hash
            and manifest.get("embedding_model") == settings.embedding_model
            and (settings.index_dir / "vectors.npy").is_file()
        ):
            return False
    require_model_settings(settings)
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=[chunk.text for chunk in chunks[start : start + batch_size]],
        )
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    settings.index_dir.mkdir(parents=True, exist_ok=True)
    np.save(settings.index_dir / "vectors.npy", matrix)
    (settings.index_dir / "chunks.json").write_text(serialized_chunks, encoding="utf-8")
    manifest_path.write_text(
        json.dumps({"content_hash": content_hash, "embedding_model": settings.embedding_model}, indent=2),
        encoding="utf-8",
    )
    return True


def retrieve(question: str, settings: Settings, top_k: int = 8) -> list[tuple[Chunk, float]]:
    require_model_settings(settings)
    vectors_path = settings.index_dir / "vectors.npy"
    chunks_path = settings.index_dir / "chunks.json"
    if not vectors_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError("Index was not found. Run index first.")
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    embedding = np.asarray(
        client.embeddings.create(model=settings.embedding_model, input=[question]).data[0].embedding,
        dtype=np.float32,
    )
    embedding /= np.linalg.norm(embedding)
    scores = np.load(vectors_path) @ embedding
    chunks = [Chunk(**chunk) for chunk in json.loads(chunks_path.read_text(encoding="utf-8"))]
    indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[index], float(scores[index])) for index in indices]
