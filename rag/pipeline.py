"""Ties loader, retriever and generator together into one object."""

import config
from rag.generator import ask_gemini, extractive_answer, llm_enabled, stream_gemini
from rag.loader import build_chunks
from rag.retriever import BM25Index


class RAGPipeline:
    def __init__(self, docs_dir=None):
        chunks = build_chunks(
            docs_dir or config.DOCS_DIR, config.CHUNK_WORDS, config.CHUNK_OVERLAP
        )
        self.index = BM25Index(chunks)

    @property
    def mode(self):
        return "Gemini" if llm_enabled() else "extractive (no API key)"

    def answer(self, question, history=None):
        """Return (answer_text, hits) where hits is [(score, chunk)]."""
        history = history or []
        hits = self.index.search(question, k=config.TOP_K)
        if not hits:
            return "I couldn't find anything relevant in the documents.", []
        if llm_enabled():
            try:
                return ask_gemini(question, hits, history), hits
            except Exception as e:  # network / API problem -> graceful fallback
                return extractive_answer(question, hits) + f"\n(LLM unavailable: {e})", hits
        return extractive_answer(question, hits), hits

    def stream_answer(self, question, history=None):
        """Yield answer text events and source metadata for the web chat."""
        history = history or []
        hits = self.index.search(question, k=config.TOP_K)
        sources = sorted({chunk["source"] for _, chunk in hits})
        if not hits:
            yield {"type": "replace", "text": "I couldn't find anything relevant in the documents."}
            yield {"type": "sources", "sources": []}
            return

        if llm_enabled():
            try:
                has_text = False
                for text in stream_gemini(question, hits, history):
                    has_text = has_text or bool(text)
                    if text:
                        yield {"type": "chunk", "text": text}
                if not has_text:
                    raise RuntimeError("Gemini returned an empty response.")
            except Exception as e:  # network / API problem -> graceful fallback
                fallback = extractive_answer(question, hits)
                yield {"type": "replace", "text": f"{fallback}\n(LLM unavailable: {e})"}
        else:
            yield {"type": "replace", "text": extractive_answer(question, hits)}

        yield {"type": "sources", "sources": sources}


def format_sources(hits):
    """Unique source files with their best retrieval score."""
    best = {}
    for score, c in hits:
        best[c["source"]] = max(score, best.get(c["source"], 0))
    return ", ".join(f"{name} ({s:.1f})" for name, s in best.items())
