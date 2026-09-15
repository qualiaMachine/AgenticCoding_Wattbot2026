# WattBot 2026 — MVP Pipeline Plan

## Is "the full system" a good MVP? No.

Reading the challenge page literally, a first build could try to handle everything at once:
download and OCR every PDF (including figures), retrieve across the whole corpus, combine
numbers across documents, attribute company claims correctly, reconcile conflicting sources,
calibrate refusal, *and* wrap it in a sub-30-second chatbot UI. That is not a good MVP:

- **Too many hard problems bundled together.** Figure/vision OCR, cross-document arithmetic,
  claim attribution, and source reconciliation are each separate, uncertain-effort R&D tracks.
  Building all of them before the first scored run means a failure anywhere hides the real
  baseline and we can't tell which component to blame.
- **No proof the harness works yet.** `Score.py` and the submission schema are the ground
  truth for "did we build the right thing." Nothing should be built before we've run
  `Score.py` once, successfully, against *some* predictions file — even a trivial one.
- **The chatbot (bonus track) isn't scored.** Building an interface before a scored pipeline
  exists is solving the wrong problem first.
- **It violates the MVP goal stated in the prompt itself:** "something we can get running
  quickly and understand end-to-end... the baseline we A/B new components against." A system
  that requires OCR + reconciliation + attribution logic before it produces one row of output
  cannot be stood up quickly, and it gives us nothing to A/B against.

## A better starting point

Scope the MVP to the *easiest* question category only — "locate a stated fact" — using
plain-text retrieval and a single LLM call, and explicitly punt every other question type to
`is_blank` for now. This is fast to build, exercises every stage of the real pipeline
(data → retrieval → generation → scored submission), and gives a real (if low) leaderboard
number to improve on. Every later investment (OCR, reconciliation, combination reasoning,
attribution prompting, UI) gets A/B tested by rerunning `Score.py` against this baseline.

## Steps

1. **Pull the competition data.**
   Download `metadata.csv`, `train_QA.csv`, `test_Q.csv`, `Score.py`, and
   `CONTRIBUTE_QUESTIONS.md` from the Kaggle Data tab into `data/`.
   *Know it works:* `train_QA.csv` loads with 245 rows, `test_Q.csv` with 317 rows, and both
   share the columns documented on the challenge page.

2. **Sanity-check the scoring harness before writing any pipeline code.**
   Build a trivial "predict `is_blank` for every row" submission from `train_QA.csv` and run
   `python Score.py train_QA.csv trivial_predictions.csv`.
   *Know it works:* the script exits cleanly and prints the per-component breakdown
   (answer_value / ref_id / is_NA). We now trust the submission format and scoring mechanics
   before any modeling work starts.

3. **Acquire and parse the corpus — text only, no OCR yet.**
   Download the PDFs referenced in `metadata.csv` (respecting the download etiquette on the
   Data tab) and extract plain text with a standard PDF text extractor.
   *Know it works:* log a parse success rate per document; most documents yield non-trivial
   extracted text. Documents that fail (scanned pages, figure-only content) are logged and
   explicitly deferred, not fixed here.

4. **Chunk and index the corpus.**
   Fixed-size chunking with overlap, embed with an available model (BadgerBrain
   `qwen3-vl-embedding-8b`, or a local sentence-embedding fallback), store in a simple vector
   index (e.g. FAISS/Chroma).
   *Know it works:* for a handful of "locate a stated fact" questions in `train_QA.csv`,
   confirm the known supporting passage appears in the top-k retrieved chunks.

5. **Add a minimal generation step.**
   Prompt an available BadgerBrain chat model (e.g. `qwen3.8-27b`) with the retrieved chunks
   and the question, instructing it to fill `answer`, `answer_value`, `ref_id`,
   `supporting_materials`, `explanation` in the exact submission schema, and to output
   `is_blank` when unsure.
   *Know it works:* manually inspect 15–20 outputs for correct schema conformance (no empty
   `explanation`, `ref_id` values that exist in `metadata.csv`, etc.).

6. **Explicitly scope v1 to one question type.**
   Tag each `train_QA.csv` row by the difficulty categories on the challenge page (fact
   lookup, figure read, combination, attribution, reconciliation, refusal). For v1, only
   attempt "locate a stated fact" and true refusals; force every other category to `is_blank`.
   *Know it works:* the tagging pass reports what fraction of `train_QA.csv` the MVP is
   actually attempting, so a low score is understood, not a surprise.

7. **Run the MVP end-to-end against `train_QA.csv` and score it.**
   Generate predictions for all 245 rows with the steps above and run
   `python Score.py train_QA.csv mvp_predictions.csv`.
   *Know it works:* we get a non-zero overall score, and the printed breakdown by evidence
   type roughly matches what we intentionally attempted (decent score on fact-lookup rows,
   near-zero contribution from categories we skipped, correct credit on `is_NA` for
   refusals).

8. **Generate a formatted test submission.**
   Run the same pipeline over `test_Q.csv` and produce `submission.csv` in the required
   column format.
   *Know it works:* the file has all required columns, no empty `explanation` cells, and
   passes the same structural checks `Score.py` applies to `train_QA.csv` (even though the
   test set can't be locally scored).

9. **Tag this as the baseline and A/B everything else against it.**
   Commit/tag the repo state that produced step 7's score. Every subsequent improvement
   (figure OCR, cross-document combination, attribution framing, source reconciliation,
   better retrieval, chatbot UI) is measured by rerunning `Score.py` on `train_QA.csv` and
   comparing to this tagged baseline — not by intuition.
   *Know it works:* a documented baseline score exists, tied to a specific commit/tag, that
   every later change is diffed against.
