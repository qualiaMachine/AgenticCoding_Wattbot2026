import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from wattbot.config import Settings
from wattbot.data import SUBMISSION_COLUMNS, read_csv, validate_submission, write_submission
from wattbot.extraction import _chunk_page
from wattbot.generation import normalize_prediction
from wattbot.index import build_index


class DataTests(unittest.TestCase):
    def setUp(self):
        self.question = {"id": "q1", "question": "How much?"}

    def test_blank_answer_clears_evidence(self):
        result = normalize_prediction(
            {"answer_value": "is_blank", "answer": "", "explanation": "No relevant source."},
            self.question,
            {},
        )
        self.assertEqual(result["ref_id"], "is_blank")
        self.assertEqual(result["supporting_materials"], "is_blank")
        self.assertTrue(result["answer"])

    def test_invalid_citation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside metadata"):
            normalize_prediction(
                {"answer_value": "3", "ref_id": "unknown", "explanation": "Evidence."},
                self.question,
                {},
            )

    def test_submission_requires_explanation(self):
        row = {column: "" for column in SUBMISSION_COLUMNS}
        row.update(self.question, answer_value="3")
        with self.assertRaisesRegex(ValueError, "empty explanation"):
            validate_submission([row], [self.question])

    def test_submission_serializes_expected_column_order(self):
        row = {column: "" for column in SUBMISSION_COLUMNS}
        row.update(self.question, answer="Three", answer_value="3", explanation="A source states three.")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prediction.csv"
            write_submission(output, [row], [self.question])
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0].split(","), list(SUBMISSION_COLUMNS))

    def test_chunks_preserve_page_and_document(self):
        chunks = _chunk_page("paper1", 2, "one two three four five", 3, 1)
        self.assertEqual([chunk.text for chunk in chunks], ["one two three", "three four five"])
        self.assertTrue(all(chunk.document_id == "paper1" and chunk.page == 2 for chunk in chunks))

    def test_unchanged_index_does_not_need_model_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(root, root, None, None, "embedding", "chat")
            settings.index_dir.mkdir()
            chunks = _chunk_page("paper1", 1, "one two", 10, 1)
            (settings.index_dir / "vectors.npy").write_bytes(b"placeholder")
            content_hash = hashlib.sha256(
                json.dumps([chunk.__dict__ for chunk in chunks], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            (settings.index_dir / "manifest.json").write_text(
                json.dumps({"content_hash": content_hash, "embedding_model": "embedding"}),
                encoding="utf-8",
            )
            self.assertFalse(build_index(chunks, settings))


if __name__ == "__main__":
    unittest.main()
