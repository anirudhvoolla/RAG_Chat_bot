"""Run from the project root:  python -m unittest discover tests"""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.loader import chunk_text
from rag.retriever import BM25Index
from rag import generator


class ChunkTests(unittest.TestCase):
    def test_overlap(self):
        text = " ".join(str(i) for i in range(100))
        chunks = chunk_text(text, size=40, overlap=10)
        self.assertEqual(chunks[0].split()[30:], chunks[1].split()[:10])

    def test_empty(self):
        self.assertEqual(chunk_text("", 40, 10), [])


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.index = BM25Index([
            {"source": "a.md", "text": "Cats are small domestic animals that purr."},
            {"source": "b.md", "text": "Python is a programming language used for data science."},
            {"source": "c.md", "text": "The Eiffel Tower is located in Paris, France."},
        ])

    def test_finds_relevant_chunk(self):
        hits = self.index.search("Where is the Eiffel Tower?", k=1)
        self.assertEqual(hits[0][1]["source"], "c.md")

    def test_no_match_returns_empty(self):
        self.assertEqual(self.index.search("quantum chromodynamics"), [])


class GeminiGeneratorTests(unittest.TestCase):
    def test_sends_gemini_request_and_maps_assistant_history(self):
        response = io.BytesIO(
            json.dumps(
                {
                    "candidates": [{
                        "content": {"parts": [{"text": "Answer"}, {"text": " here"}]}
                    }]
                }
            ).encode()
        )
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        hits = [(1.0, {"source": "guide.md", "text": "Relevant facts."})]

        with patch.object(generator.config, "GEMINI_API_KEY", "test-key"), \
                patch.object(generator.config, "GEMINI_MODEL", "test-model"), \
                patch(
                    "rag.generator.urllib.request.urlopen", return_value=response
                ) as urlopen:
            answer = generator.ask_gemini("What happened?", hits, history)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(answer, "Answer here")
        self.assertEqual(
            request.full_url,
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "test-model:generateContent",
        )
        self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
        self.assertEqual(
            [item["role"] for item in payload["contents"]],
            ["user", "model", "user"],
        )
        self.assertIn("Relevant facts.", payload["contents"][-1]["parts"][0]["text"])
        self.assertIn(
            "ONLY the provided context",
            payload["systemInstruction"]["parts"][0]["text"],
        )

    def test_streams_gemini_server_sent_events(self):
        events = [
            {"candidates": [{"content": {"parts": [{"text": "Fast"}]}}]},
            {"candidates": [{"content": {"parts": [{"text": " answer"}]}}]},
        ]
        response = io.BytesIO(
            b"".join(
                b"data: " + json.dumps(event).encode() + b"\n\n"
                for event in events
            )
        )
        hits = [(1.0, {"source": "guide.md", "text": "Relevant facts."})]

        with patch.object(generator.config, "GEMINI_API_KEY", "test-key"), \
                patch.object(generator.config, "GEMINI_MODEL", "test-model"), \
                patch(
                    "rag.generator.urllib.request.urlopen", return_value=response
                ) as urlopen:
            chunks = list(generator.stream_gemini("Question?", hits, []))

        self.assertEqual(chunks, ["Fast", " answer"])
        self.assertTrue(
            urlopen.call_args.args[0].full_url.endswith(
                "test-model:streamGenerateContent?alt=sse"
            )
        )


if __name__ == "__main__":
    unittest.main()
