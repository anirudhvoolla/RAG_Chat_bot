#!/usr/bin/env python3
"""Entry point.

    python app.py              # command-line chat
    python app.py --web        # browser UI
    python app.py --ask "..."  # one-off question
"""

import argparse

import config
from rag.pipeline import RAGPipeline, format_sources
from server import run_web


def run_cli(pipeline):
    print(f"RAG chatbot ready - {len(pipeline.index.chunks)} chunks indexed, "
          f"answer mode: {pipeline.mode}")
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
        reply, hits = pipeline.answer(q, history)
        print(f"\nBot: {reply}")
        if hits:
            print("     sources: " + format_sources(hits))
        print()
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]


def main():
    ap = argparse.ArgumentParser(description="Simple RAG chatbot")
    ap.add_argument("--docs", default=None, help="folder of .txt/.md documents")
    ap.add_argument("--web", action="store_true", help="serve a browser UI")
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--ask", help="ask a single question and exit")
    args = ap.parse_args()

    pipeline = RAGPipeline(args.docs)
    if args.ask:
        reply, hits = pipeline.answer(args.ask)
        print(reply)
        print("sources:", format_sources(hits))
    elif args.web:
        run_web(pipeline, args.port)
    else:
        run_cli(pipeline)


if __name__ == "__main__":
    main()
