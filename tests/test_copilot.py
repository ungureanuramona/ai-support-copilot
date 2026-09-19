import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from copilot import (
    LLM_MODEL_NAME,
    generate_support_plan,
    load_documents,
)


class TestLoadDocuments(unittest.TestCase):
    def test_load_documents_reads_text_files(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            folder = Path(temporary_folder)

            (folder / "api_errors.txt").write_text(
                "A 500 response is a server-side error.",
                encoding="utf-8",
            )
            (folder / "authentication.txt").write_text(
                "A 401 response means invalid credentials.",
                encoding="utf-8",
            )

            documents = load_documents(folder)

            self.assertEqual(len(documents), 2)
            self.assertEqual(
                {document["source"] for document in documents},
                {"api_errors.txt", "authentication.txt"},
            )

    def test_load_documents_ignores_non_text_files(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            folder = Path(temporary_folder)

            (folder / "guide.txt").write_text(
                "Support guidance",
                encoding="utf-8",
            )
            (folder / "notes.md").write_text(
                "This file should be ignored",
                encoding="utf-8",
            )

            documents = load_documents(folder)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["source"], "guide.txt")


class TestGenerateSupportPlan(unittest.TestCase):
    @patch("copilot.chat")
    def test_generate_support_plan_uses_ticket_and_sources(self, mocked_chat):
        mocked_chat.return_value = SimpleNamespace(
            message=SimpleNamespace(
                content="## Category\n\nAPI Error"
            )
        )

        relevant_documents = [
            {
                "source": "api_errors.txt",
                "content": "A 500 response is a server-side error.",
                "score": 0.80,
            }
        ]

        result = generate_support_plan(
            "API requests are failing",
            "Customers receive a 500 response.",
            relevant_documents,
        )

        self.assertEqual(result, "## Category\n\nAPI Error")

        call_arguments = mocked_chat.call_args.kwargs

        self.assertEqual(call_arguments["model"], LLM_MODEL_NAME)
        self.assertEqual(call_arguments["options"], {"temperature": 0})

        prompt = call_arguments["messages"][0]["content"]

        self.assertIn("API requests are failing", prompt)
        self.assertIn("Customers receive a 500 response.", prompt)
        self.assertIn("api_errors.txt", prompt)


if __name__ == "__main__":
    unittest.main()