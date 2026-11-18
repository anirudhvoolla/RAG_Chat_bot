#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) chatbot.

Pipeline:
  1. LOAD     - read .txt / .md files from the docs/ folder
  2. CHUNK    - split each document into overlapping word windows
  3. INDEX    - build a BM25 keyword index over the chunks (pure Python)
  4. RETRIEVE - for each question, fetch the top-k most relevant chunks
  5. GENERATE - send the question + retrieved chunks to Gemini (if an API key
                is set); otherwise fall back to an extractive answer built
                from the best-matching sentences.

Usage:
  python rag_chatbot.py              # command-line chat
  python rag_chatbot.py --web        # browser UI at http://localhost:8000
  python rag_chatbot.py --docs notes # use a different documents folder

Optional: set GEMINI_API_KEY=... to get LLM-written answers.
No third-party packages are required.
"""

import argparse
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CHUNK_WORDS = 120      # words per chunk
CHUNK_OVERLAP = 30     # words shared between neighbouring chunks
TOP_K = 3              # chunks passed to the generator
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

STOPWORDS = set(
    "a an and are as at be by for from has have how i in is it its of on or "
    "that the this to was what when where which who why will with you your do "
    "does can about into than then so if not no".split()
)


# ----------------------------------------------------------------------------
# 1-2. Load and chunk documents
# ----------------------------------------------------------------------------
def load_documents(folder):
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            # turn markdown headings ("## Title") into plain sentences ("Title.")
            text = re.sub(r"^#+\s*(.+?)\s*$", r"\1.", text, flags=re.MULTILINE)
            docs.append((path.name, text))
    return docs


def fmt_sources(hits):
    """Unique source files with their best retrieval score."""
    best = {}
    for score, c in hits:
        best[c["source"]] = max(score, best.get(c["source"], 0))
    return ", ".join(f"{name} ({s:.1f})" for name, s in best.items())


def chunk_text(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


# ----------------------------------------------------------------------------
# 3-4. BM25 index and retrieval
# ----------------------------------------------------------------------------
def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks  # list of dicts: {"source": str, "text": str}
        self.k1, self.b = k1, b
        self.tokens = [tokenize(c["text"]) for c in chunks]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df = Counter()
        for t in self.tokens:
            df.update(set(t))
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def search(self, query, k=TOP_K):
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            score = 0.0
            for w in q:
                if w not in tf:
                    continue
                num = tf[w] * (self.k1 + 1)
                den = tf[w] + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                score += self.idf[w] * num / den
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scored[:k]]


def build_index(folder):
    chunks = []
    for name, text in load_documents(folder):
        for piece in chunk_text(text):
            chunks.append({"source": name, "text": piece})
    if not chunks:
        sys.exit(f"No .txt or .md files found in '{folder}'. Add some documents first.")
    return BM25Index(chunks)


# ----------------------------------------------------------------------------
# 5. Generation
# ----------------------------------------------------------------------------
def ask_gemini(question, hits, history):
    """Call the Gemini generateContent API using only the standard library."""
    context = "\n\n".join(
        f"[Source {n}: {c['source']}]\n{c['text']}" for n, (_, c) in enumerate(hits, 1)
    )
    system = (
        "You are a helpful assistant that answers questions using ONLY the provided "
        "context. If the context does not contain the answer, say you don't know. "
        "Mention the source file names you used."
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
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 600},
    }).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=body,
        headers={
            "content-type": "application/json",
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return "".join(
        part.get("text", "")
        for part in data["candidates"][0]["content"]["parts"]
    )


def extractive_answer(question, hits):
    """Fallback with no LLM: return the sentences that best overlap the question."""
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
    return " ".join(s for _, s, _ in best) + f"  (from: {', '.join(sorted({x[2] for x in best}))})"


def answer(index, question, history):
    hits = index.search(question)
    if not hits:
        return "I couldn't find anything relevant in the documents.", []
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return ask_gemini(question, hits, history), hits
        except Exception as e:  # network / API problems -> graceful fallback
            return extractive_answer(question, hits) + f"\n(LLM unavailable: {e})", hits
    return extractive_answer(question, hits), hits


# ----------------------------------------------------------------------------
# Interfaces: CLI and web
# ----------------------------------------------------------------------------
def run_cli(index):
    mode = "Gemini" if os.environ.get("GEMINI_API_KEY") else "extractive (no API key)"
    print(f"RAG chatbot ready - {len(index.chunks)} chunks indexed, answer mode: {mode}")
    print("Type your question, or 'quit' to exit.\n")
    history = []
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if not q:
            continue
        reply, hits = answer(index, q, history)
        print(f"\nBot: {reply}")
        if hits:
            print("     sources: " + fmt_sources(hits))
        print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>RAG Chatbot</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:0 auto;padding:16px;background:#f6f7f9}
h1{font-size:20px} #log{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;height:60vh;overflow:auto}
.m{margin:8px 0;padding:8px 12px;border-radius:8px;white-space:pre-wrap}
.u{background:#e8f0fe;text-align:right}.b{background:#f1f3f4}.s{font-size:12px;color:#666}
form{display:flex;gap:8px;margin-top:10px}input{flex:1;padding:10px;border:1px solid #ccc;border-radius:8px}
button{padding:10px 16px;border:0;border-radius:8px;background:#1a73e8;color:#fff;cursor:pointer}
</style>
<h1>RAG Chatbot</h1><div id=log></div>
<form id=f><input id=q placeholder="Ask a question about the documents..." autofocus><button>Send</button></form>
<script>
const log=document.getElementById('log'),f=document.getElementById('f'),q=document.getElementById('q');
let history=[];
function add(cls,text,src){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;
 if(src){const s=document.createElement('div');s.className='s';s.textContent='sources: '+src;d.appendChild(s)}
 log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
f.onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t)return;q.value='';add('u',t);
 const w=add('b','Thinking...');
 try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:t,history})});
 const j=await r.json();w.remove();add('b',j.answer,j.sources.join(', '));
 history.push({role:'user',content:t},{role:'assistant',content:j.answer})}
 catch(err){w.textContent='Error: '+err}};
</script>"""


def run_web(index, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/ask":
                return self._send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            reply, hits = answer(index, data.get("question", ""), data.get("history", []))
            out = {"answer": reply, "sources": sorted({c["source"] for _, c in hits})}
            self._send(200, json.dumps(out).encode(), "application/json")

        def log_message(self, *args):
            pass

    print(f"Web UI running at http://localhost:{port}  (Ctrl+C to stop)")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Simple RAG chatbot")
    ap.add_argument("--docs", default="docs", help="folder of .txt/.md documents")
    ap.add_argument("--web", action="store_true", help="serve a browser UI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ask", help="ask a single question and exit")
    args = ap.parse_args()

    index = build_index(args.docs)
    if args.ask:
        reply, hits = answer(index, args.ask, [])
        print(reply)
        print("sources:", fmt_sources(hits))
    elif args.web:
        run_web(index, args.port)
    else:
        run_cli(index)


if __name__ == "__main__":
    main()
