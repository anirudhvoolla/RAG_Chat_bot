"""Step 5: turn retrieved chunks into an answer (Gemini API, or an offline fallback)."""

import json
import re
import urllib.request

import config
from rag.retriever import tokenize

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context. If the context does not contain the answer, say you don't know. "
    "Mention the source file names you used."
)


def llm_enabled():
    return bool(config.GEMINI_API_KEY)


def _gemini_request(question, hits, history, streaming=False):
    context = "\n\n".join(
        f"[Source {n}: {c['source']}]\n{c['text']}" for n, (_, c) in enumerate(hits, 1)
    )
    contents = [
        {
            "role": "model" if message["role"] == "assistant" else "user",
            "parts": [{"text": message["content"]}],
        }
        for message in history[-6:]
    ]
    contents.append(
        {
            "role": "user",
            "parts": [{"text": f"Context:\n{context}\n\nQuestion: {question}"}],
        }
    )
    body = json.dumps(
        {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 600},
        }
    ).encode()
    action = ":streamGenerateContent?alt=sse" if streaming else ":generateContent"
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_MODEL}{action}",
        data=body,
        headers={
            "content-type": "application/json",
            "x-goog-api-key": config.GEMINI_API_KEY,
        },
    )
    return req


def ask_gemini(question, hits, history):
    """Call the Gemini generateContent API using only the standard library."""
    req = _gemini_request(question, hits, history)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return "".join(
        part.get("text", "")
        for part in data["candidates"][0]["content"]["parts"]
    )


def stream_gemini(question, hits, history):
    """Yield answer text as Gemini generates it via server-sent events."""
    req = _gemini_request(question, hits, history, streaming=True)
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if "error" in data:
                raise RuntimeError(data["error"].get("message", "Gemini API error"))
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    text = part.get("text", "")
                    if text:
                        yield text


def extractive_answer(question, hits):
    """Offline fallback: return the sentences that best overlap the question."""
    q = set(tokenize(question))
    sentences = []
    for _, c in hits:
        for s in re.split(r"(?<=[.!?])\s+", c["text"]):
            overlap = len(q & set(tokenize(s)))
            if overlap:
                sentences.append((overlap, s.strip(), c["source"]))
    sentences.sort(key=lambda x: -x[0])
    best = sentences[:3]
    if not best:
        return "I couldn't find that in the documents."
    sources = ", ".join(sorted({x[2] for x in best}))
    return " ".join(s for _, s, _ in best) + f"  (from: {sources})"
