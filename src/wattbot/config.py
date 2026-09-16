from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifacts_dir: Path
    api_key: str | None
    base_url: str | None
    embedding_model: str
    chat_model: str

    @property
    def corpus_dir(self) -> Path:
        return self.artifacts_dir / "corpus"

    @property
    def extraction_dir(self) -> Path:
        return self.artifacts_dir / "extraction"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"


def _configured_path(variable: str, default: str) -> Path:
    path = Path(os.getenv(variable, default))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_settings() -> Settings:
    load_dotenv(REPOSITORY_ROOT / ".env")
    return Settings(
        data_dir=_configured_path("WATTBOT_DATA_DIR", "data"),
        artifacts_dir=_configured_path("WATTBOT_ARTIFACTS_DIR", "artifacts"),
        api_key=os.getenv("WATTBOT_API_KEY"),
        base_url=os.getenv("WATTBOT_BASE_URL"),
        embedding_model=os.getenv("WATTBOT_EMBEDDING_MODEL", "qwen3-vl-embedding-8b"),
        chat_model=os.getenv("WATTBOT_CHAT_MODEL", "qwen3.8-27b"),
    )


def require_model_settings(settings: Settings) -> None:
    missing = [
        name
        for name, value in {
            "WATTBOT_API_KEY": settings.api_key,
            "WATTBOT_BASE_URL": settings.base_url,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Missing {', '.join(missing)}. Copy .env.example to .env and configure BadgerBrain."
        )

