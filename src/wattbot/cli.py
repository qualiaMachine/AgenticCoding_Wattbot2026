from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import load_settings
from .corpus import download_corpus
from .data import dataset_summary, load_dataset, write_submission
from .extraction import extract_corpus, load_chunks
from .generation import generate_answer
from .index import build_index, retrieve


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WattBot 2026 baseline RAG pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Validate and summarize competition CSVs")
    download = commands.add_parser("download", help="Download and cache source PDFs")
    download.add_argument("--limit", type=int)
    extract = commands.add_parser("extract", help="Extract page-provenance chunks from cached PDFs")
    extract.add_argument("--chunk-words", type=int, default=350)
    extract.add_argument("--overlap-words", type=int, default=50)
    commands.add_parser("index", help="Embed extracted chunks and save a vector index")
    query = commands.add_parser("retrieve", help="Inspect semantic retrieval results")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=8)
    predict = commands.add_parser("predict", help="Generate a submission CSV")
    predict.add_argument("--split", choices=("train", "test"), required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--top-k", type=int, default=8)
    score = commands.add_parser("score", help="Run competition Score.py on training predictions")
    score.add_argument("predictions", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = load_settings()
    if args.command == "inspect":
        print(json.dumps(dataset_summary(load_dataset(settings.data_dir)), indent=2))
    elif args.command == "download":
        records = download_corpus(load_dataset(settings.data_dir).metadata, settings.corpus_dir, args.limit)
        print(json.dumps({"downloaded": sum(r["status"] == "downloaded" for r in records), "records": records}, indent=2))
    elif args.command == "extract":
        chunks = extract_corpus(settings.corpus_dir, settings.extraction_dir, args.chunk_words, args.overlap_words)
        print(f"Extracted {len(chunks)} chunks to {settings.extraction_dir / 'chunks.jsonl'}")
    elif args.command == "index":
        rebuilt = build_index(load_chunks(settings.extraction_dir / "chunks.jsonl"), settings)
        print(f"{'Saved' if rebuilt else 'Reused'} vector index at {settings.index_dir}")
    elif args.command == "retrieve":
        for chunk, score in retrieve(args.question, settings, args.top_k):
            print(f"{score:.4f} {chunk.document_id} p.{chunk.page}: {chunk.text[:500]}\n")
    elif args.command == "predict":
        dataset = load_dataset(settings.data_dir)
        questions = dataset.train_questions if args.split == "train" else dataset.test_questions
        if args.limit:
            questions = questions[: args.limit]
        metadata = {row["id"]: row for row in dataset.metadata}
        rows = []
        traces = []
        for question in questions:
            contexts = retrieve(question["question"], settings, args.top_k)
            prediction = generate_answer(question, contexts, metadata, settings)
            rows.append(prediction)
            traces.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "prediction": prediction,
                    "retrieval": [
                        {
                            "document_id": chunk.document_id,
                            "page": chunk.page,
                            "chunk_index": chunk.chunk_index,
                            "score": score,
                            "text": chunk.text,
                        }
                        for chunk, score in contexts
                    ],
                }
            )
        write_submission(args.output, rows, questions)
        trace_path = args.output.with_suffix(".traces.jsonl")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces), encoding="utf-8"
        )
        print(f"Wrote {len(rows)} predictions to {args.output} and traces to {trace_path}")
    elif args.command == "score":
        dataset = load_dataset(settings.data_dir)
        scorer = settings.data_dir / "Score.py"
        if not scorer.is_file():
            raise FileNotFoundError(f"Official scorer was not found: {scorer}")
        result = subprocess.run([sys.executable, str(scorer), str(settings.data_dir / "train_QA.csv"), str(args.predictions)])
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
