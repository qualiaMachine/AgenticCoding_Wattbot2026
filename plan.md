# WattBot 2026 — MVP Pipeline Plan

## Verdict: is a naive single-hop RAG a good MVP?

**Yes — a plain "embed → retrieve top-k → single LLM call → format row" pipeline is a good MVP**, on the
condition that it is scoped to prose/table facts and treats figures, multi-hop reconciliation, and
attribution nuance as *known gaps* rather than something the MVP tries to solve.

Why it's good:
- It is buildable in days, not weeks, using tools already available (BadgerBrain-hosted
  `qwen3-vl-embedding-8b` for embeddings, `qwen3-8b`/`qwen3-27b` for generation).
- It exercises the **entire** pipeline end-to-end (download → parse → chunk → index → retrieve →
  generate → format → score), so every later component swap (better chunking, reranking, multi-hop,
  vision OCR, abstention calibration) can be A/B'd against a real, scored baseline instead of a guess.
- `train_QA.csv` + `Score.py` already give a free, authoritative evaluation loop — the MVP's only job is
  to produce a CSV that loop can score, broken down by evidence type (fact, figure, combination,
  attribution, reconciliation, abstention).
- The scoring rubric (0.75 value / 0.20 citation F1 / 0.05 is_NA) rewards getting the *shape* of an
  answer right (right doc, right value format, correct abstention) more than getting every hard
  reasoning question right — a naive pipeline can already score non-trivially on "locate a stated fact"
  and "attribute a claim" questions, which is enough to be a meaningful baseline.

Why it's *not* the end state (and shouldn't be mistaken for one):
- It will systematically fail on figure-only values (no OCR/vision), multi-document combination
  questions (no multi-hop retrieval), reconciliation questions (no contradiction-aware reasoning), and
  will likely over- or under-abstain without explicit calibration against `train_QA.csv`'s evidence-type
  flag.
- Citation F1 punishes both under- and over-citing, so a naive "always cite the top-k" retriever will
  cap the citation component regardless of how good generation is.

These gaps are exactly what should be A/B'd against the MVP next (vision/OCR pass, reranker, multi-hop
retrieval, contradiction-aware prompting, abstention thresholding) — not reasons to delay shipping it.

## Plan

Each step lists what to build and the concrete check that proves it works before moving on.

1. **Pull the competition assets**
   - Build: download `metadata.csv`, `train_QA.csv`, `test_Q.csv`, `Score.py`,
     `CONTRIBUTE_QUESTIONS.md` from the Data tab; fetch every PDF from the pinned URLs in
     `metadata.csv`.
   - Verified when: row counts match the spec (100+ corpus docs, 245 train rows, 317 test rows);
     every metadata URL either downloads successfully or is logged as a named exception.

2. **Extract text from PDFs**
   - Build: parse each PDF into page-level text (e.g. PyMuPDF/pdfplumber), keeping `doc_id` + page
     number as metadata.
   - Verified when: every doc in `metadata.csv` has a parse result or a logged failure; a manual
     spot-check of 5 PDFs shows extracted text matches what's visibly on the page.

3. **Chunk and index**
   - Build: split page text into overlapping passages, embed with `qwen3-vl-embedding-8b` via
     BadgerBrain, store vectors + metadata (`doc_id`, page) in a vector store (FAISS/Chroma).
   - Verified when: for 10 hand-picked "locate a stated fact" questions from `train_QA.csv`, the
     ground-truth `ref_id` document appears in the top-k retrieved chunks.

4. **Naive retrieve-and-generate**
   - Build: for each question, retrieve top-k chunks, prompt `qwen3-8b`/`qwen3-27b` to produce
     `answer`, `answer_value`, `ref_id` (restricted to retrieved docs), `supporting_materials`
     (verbatim quote), `explanation`; instruct it to output `is_blank` when the retrieved context
     doesn't support an answer.
   - Verified when: running on 20 sampled train questions produces well-formed output for every row
     (no missing fields) on manual inspection.

5. **Format the submission CSV**
   - Build: map model output to the required columns and conventions (blank all evidence columns when
     `answer_value` is `is_blank`; encode ranges as `(low,high)`; never emit `<`, `>`, `~` in
     `answer_value`).
   - Verified when: a schema check confirms all required columns exist, `explanation` is non-empty on
     every row, and abstained rows have every evidence column blank.

6. **Score locally against the free baseline**
   - Build: run `python Score.py train_QA.csv my_train_predictions.csv` over the full 245-row train
     set.
   - Verified when: `Score.py` completes without error, returns a single WattBot Score in `[0, 1]`,
     and prints the per-evidence-type breakdown.

7. **Establish the A/B baseline**
   - Build: record the overall score and the per-category breakdown (fact / figure / combination /
     attribution / reconciliation / abstention) as the MVP's fixed reference point.
   - Verified when: the breakdown clearly ranks which evidence type is weakest, giving a prioritized
     list of components to test next (e.g. OCR for figures, multi-hop retrieval for combination
     questions).

8. **Produce a first test-set submission**
   - Build: run the same pipeline over `test_Q.csv` (317 questions) and write `submission.csv`.
   - Verified when: the file has exactly 317 rows in the required schema and the public leaderboard
     accepts it and returns a score.

## Open clarifying questions

1. Is there any existing code/notebook for this challenge beyond what's on Kaggle, or are we starting
   from a blank repo (this repo currently only has a README)?
2. For the MVP, is it acceptable to explicitly skip figure/table OCR and vision-model reading (treat
   those questions as an expected gap), or do you want minimal vision-model support in scope from day
   one?
3. Any preference on vector store (FAISS vs. Chroma vs. a simple in-memory cosine index) or should the
   MVP just use whatever is fastest to stand up?
4. Should the MVP target only the local `Score.py` loop against `train_QA.csv`, or should step 8
   (an actual test-set leaderboard submission) be part of the MVP itself?
5. Do we already have the PDFs downloaded locally, or does download/parsing need to be built from
   scratch against `metadata.csv`'s pinned URLs?
6. Any constraints on which BadgerBrain models to use (e.g., must we use `qwen3-vl-embedding-8b`
   specifically), or is swapping in another embedding/generation model acceptable during MVP
   iteration?
