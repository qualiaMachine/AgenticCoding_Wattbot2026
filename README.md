# WattBot 2026 baseline

This repository is a reproducible minimum viable retrieval-augmented generation
pipeline for the [WattBot 2026](https://www.kaggle.com/competitions/WattBot2026)
competition. It intentionally prioritizes inspectability and local scoring over
advanced retrieval, OCR, or a chatbot interface.

## Setup

1. Use Python 3.11 or newer, then install the project:

   ```powershell
   python -m pip install -e .
   ```

2. Copy `.env.example` to `.env`, then add the BadgerBrain API key and
   OpenAI-compatible base URL. Do not commit `.env`.

3. Download `metadata.csv`, `train_QA.csv`, `test_Q.csv`, and `Score.py` from
   the competition Data tab into `data\`.

## Pipeline

```powershell
# Validate the Kaggle inputs and inspect question conventions.
wattbot inspect

# Cache source PDFs. Begin with a small sample while configuring access.
wattbot download --limit 5

# Preserve document ID and page provenance in each text chunk.
wattbot extract

# Build a persistent semantic index using the configured BadgerBrain embedding model.
wattbot index

# Examine what a query retrieves before asking an answer model.
wattbot retrieve "How many metric tons of CO2 equivalent were reported for training Llama 3.1 405B?"

# Generate a small training slice and score it with the competition's exact scorer.
wattbot predict --split train --limit 10 --output artifacts\predictions\train-sample.csv
wattbot score artifacts\predictions\train-sample.csv

# After inspection and scoring, generate the complete submission.
wattbot predict --split test --output artifacts\predictions\submission.csv
```

Generated PDFs, extracted chunks, indices, traces, and predictions live under
`artifacts\` by default and are ignored by Git. The retrieval command shows
document IDs, pages, scores, and excerpts; generated prediction rows retain
the selected citations, verbatim support, and explanation.

## Baseline boundaries

The initial pipeline handles text-extractable PDFs and semantic retrieval.
Figure/table OCR, lexical retrieval, reranking, question-type routing,
multi-document reasoning, and a user-facing chatbot are deliberately deferred
until a scored baseline establishes which failure modes are worth improving.
