from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SUBMISSION_COLUMNS = (
    "id",
    "question",
    "answer",
    "answer_value",
    "answer_unit",
    "ref_id",
    "ref_url",
    "supporting_materials",
    "explanation",
)
METADATA_REQUIRED_COLUMNS = {"id", "title"}
QUESTION_REQUIRED_COLUMNS = {"id", "question"}


@dataclass(frozen=True)
class Dataset:
    metadata: list[dict[str, str]]
    train_questions: list[dict[str, str]]
    test_questions: list[dict[str, str]]


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset file was not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required_columns - fieldnames
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(sorted(missing))}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name} contains duplicate ids")
    return rows


def load_dataset(data_dir: Path) -> Dataset:
    return Dataset(
        metadata=read_csv(data_dir / "metadata.csv", METADATA_REQUIRED_COLUMNS),
        train_questions=read_csv(data_dir / "train_QA.csv", QUESTION_REQUIRED_COLUMNS),
        test_questions=read_csv(data_dir / "test_Q.csv", QUESTION_REQUIRED_COLUMNS),
    )


def dataset_summary(dataset: Dataset) -> dict[str, object]:
    summary: dict[str, object] = {
        "metadata_documents": len(dataset.metadata),
        "training_questions": len(dataset.train_questions),
        "test_questions": len(dataset.test_questions),
    }
    if dataset.train_questions:
        headers = set(dataset.train_questions[0])
        for column in ("evidence_type", "is_NA", "answer_type"):
            if column in headers:
                summary[column] = dict(Counter(row.get(column, "") for row in dataset.train_questions))
    return summary


def validate_submission(rows: list[dict[str, str]], questions: list[dict[str, str]]) -> None:
    expected_ids = [row["id"] for row in questions]
    actual_ids = [row.get("id", "") for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("Prediction IDs must exactly match the source question rows and their order")
    for number, row in enumerate(rows, start=1):
        missing = [column for column in SUBMISSION_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Prediction row {number} is missing columns: {', '.join(missing)}")
        if not row["explanation"].strip():
            raise ValueError(f"Prediction row {number} has an empty explanation")
        if row["answer_value"].strip().lower() == "is_blank":
            for column in ("ref_id", "ref_url", "supporting_materials"):
                if row[column].strip().lower() not in {"", "is_blank"}:
                    raise ValueError(f"Prediction row {number} is_blank answer has non-blank {column}")


def write_submission(path: Path, rows: list[dict[str, str]], questions: list[dict[str, str]]) -> None:
    validate_submission(rows, questions)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

