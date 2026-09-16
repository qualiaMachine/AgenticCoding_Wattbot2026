from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def source_url(row: dict[str, str]) -> str:
    for column in ("url", "ref_url", "pinned_url", "source_url"):
        if row.get(column):
            return row[column]
    raise ValueError(f"Metadata source {row['id']} does not contain a URL column")


def download_corpus(
    metadata: list[dict[str, str]], corpus_dir: Path, limit: int | None = None, attempts: int = 3
) -> list[dict[str, str]]:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    selected = metadata[:limit] if limit else metadata
    for row in selected:
        document_id = row["id"]
        destination = corpus_dir / f"{document_id}.pdf"
        record = {"id": document_id, "url": source_url(row), "fetched_at": datetime.now(UTC).isoformat()}
        if destination.is_file() and destination.stat().st_size:
            record.update(status="cached", sha256=_sha256(destination), path=destination.name)
        else:
            request = urllib.request.Request(record["url"], headers={"User-Agent": "WattBot2026-baseline/0.1"})
            for attempt in range(1, attempts + 1):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
                        output.write(response.read())
                    record.update(
                        status="downloaded",
                        sha256=_sha256(destination),
                        path=destination.name,
                        attempts=attempt,
                    )
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    destination.unlink(missing_ok=True)
                    if attempt == attempts:
                        record.update(status="failed", error=str(error), attempts=attempt)
                    else:
                        time.sleep(attempt)
        manifest[document_id] = record
        time.sleep(0.25)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return [manifest[row["id"]] for row in selected]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
