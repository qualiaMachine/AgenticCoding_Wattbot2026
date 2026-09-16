# WattBot MVP Pipeline

## Assessment

This is a good MVP direction, provided it is deliberately a submission pipeline
rather than a full chatbot. It directly exercises the competition's critical
path end to end: source-backed retrieval, exact citations, normalized answer
values, abstention, and local scoring. It can establish a reproducible score on
`train_QA.csv` before experiments with rerankers, vision/OCR, more
sophisticated reasoning, or a user interface.

The MVP should optimize for traceability and iteration speed, not leaderboard
maximum. Each generated row should retain the retrieved excerpts, selected
document IDs, and model rationale so errors can be categorized from `Score.py`
results.

## Scope

- Use the BadgerBrain OpenAI-compatible API for the Qwen embedding model and an
  instruction model.
- Build a command-line Python pipeline that produces the competition CSV format.
- Download source PDFs from the pinned URLs in `metadata.csv` with caching and
  retry-aware, polite fetching.
- Support text-based PDF extraction only in the baseline. OCR/vision, a
  chatbot, and advanced multi-document decomposition are deferred.
- Evaluate only against `train_QA.csv` during development; generate a separate
  prediction file for `test_Q.csv` after the pipeline is stable.

## Ordered implementation plan

1. **Establish the Python project and configuration contract.**
   - Create a minimal dependency-managed Python layout, `.env.example`, and
     documentation for Kaggle data placement plus BadgerBrain credentials/model
     names.
   - Keep secrets out of source control; expose endpoint, API key, embedding
     model, extraction model, and run paths through configuration.
   - **Works when:** a setup command validates configuration and can list the
     expected dataset files without making a model request.

2. **Inspect and validate competition inputs.**
   - Add loaders for `metadata.csv`, `train_QA.csv`, and `test_Q.csv` that
     verify required columns, IDs, URLs, and submission schema expectations.
   - Produce a compact dataset summary, including training question evidence
     types and unanswerable-question count, to guide error analysis.
   - **Works when:** the inspection command reports valid row counts and fails
     clearly for absent or malformed input.

3. **Acquire and cache the source corpus.**
   - Download each metadata URL to an ignored local cache, retain a manifest of
     status, checksum, timestamp, and source ID, and make repeat runs reuse
     successful downloads.
   - Record download failures explicitly instead of silently substituting other
     sources.
   - **Works when:** a corpus command can fetch a small configurable sample,
     rerun without re-downloading it, and report every failed source.

4. **Extract, normalize, and chunk source text with provenance.**
   - Parse text-based PDFs page by page; normalize whitespace while preserving
     source ID, page number, chunk index, and enough surrounding text for
     verbatim evidence.
   - Persist chunks in an inspectable local format so extraction can be rerun
     independently of retrieval.
   - **Works when:** an inspection command prints chunks and page provenance for
     selected documents, including the original evidence text.

5. **Build a persistent vector index using BadgerBrain embeddings.**
   - Batch embeddings for provenance-bearing chunks; persist vectors and
     metadata locally, keyed by corpus-manifest/content hashes so unchanged
     chunks are not re-embedded.
   - Implement top-k semantic retrieval and expose query, score, source ID,
     page, and chunk text for debugging.
   - **Works when:** an inspected training question returns relevant corpus
     excerpts with their exact document IDs and pages.

6. **Generate constrained, evidence-grounded answers.**
   - Send the question plus a bounded set of retrieved chunks to the
     BadgerBrain instruction model with a strict structured-output schema
     matching the submission fields.
   - Require evidence quotations from provided chunks and source IDs from their
     metadata; normalize numbers, ranges, units, categorical values, Boolean
     values, and abstentions before output.
   - Enforce valid unanswerable rows: `answer_value` and all evidence fields are
     `is_blank`, while `answer` is a refusal and `explanation` remains non-empty.
   - **Works when:** a small fixed training slice yields schema-valid rows, has
     no invented source IDs, and every answer includes a non-empty explanation.

7. **Produce submissions and evaluate them with the official scorer.**
   - Add commands to run a specified input split, write the exact CSV column
     order, and invoke the supplied `Score.py` against training predictions.
   - Save run configuration and per-question retrieval/output traces beside
     each predictions file for reproducibility and error analysis.
   - **Works when:** `python Score.py train_QA.csv <predictions>` runs
     successfully and the pipeline can generate a schema-valid `test_Q.csv`
     submission.

8. **Add targeted regression coverage and baseline documentation.**
   - Test dataset/schema validation, chunk provenance, submission serialization,
     `is_blank` behavior, and response normalization with local fixtures.
   - Document the one-command end-to-end workflow, expected artifacts, known
     limitations, and the initial training score/breakdown.
   - **Works when:** targeted tests pass without network or model credentials,
     and a new contributor can follow the README to reproduce a scored training
     run.

## Deferred experiments after the baseline

- Hybrid lexical/vector retrieval and reranking to improve document precision.
- Question-type routing for direct lookup, numerical derivation, reconciliation,
  and abstention.
- Vision/OCR extraction for figure- and table-heavy documents.
- Multi-document evidence planning and citation-set selection.
- A citation-forward chatbot that uses the same pipeline.

## Key considerations

- Citation precision matters nearly as much as answer quality: retrieve broadly,
  but cite only the final evidence set.
- `Score.py` and `train_QA.csv` define all formatting and normalization
  decisions; implementation should be adjusted to their observed conventions
  rather than assumptions.
- The baseline must be reproducible: cached documents, pinned model
  configuration, and saved traces are required for fair A/B comparisons.
