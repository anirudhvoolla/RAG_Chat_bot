# Simple RAG Chatbot

A small Retrieval-Augmented Generation chatbot that answers questions from your own documents. Pure Python, no third-party packages.

## Project structure

```
rag_chatbot/
├── app.py              # entry point (CLI, --web, --ask)
├── config.py           # settings, reads .env
├── server.py           # web server + /ask endpoint
├── .env                # your settings and API key (not for sharing)
├── .env.example        # template for .env
├── requirements.txt    # (no external dependencies)
├── rag/
│   ├── loader.py       # load documents + chunking
│   ├── retriever.py    # BM25 index + top-k search
│   ├── generator.py    # Gemini API call / offline fallback
│   └── pipeline.py     # combines the steps
├── templates/
│   └── index.html      # chat page
├── static/
│   ├── style.css
│   └── script.js
├── docs/               # knowledge base: put your .txt / .md files here
└── tests/
    └── test_rag.py
```

## How it works

| Step | Where | What happens |
|------|-------|--------------|
| Load | `rag/loader.py` | Reads every `.txt` / `.md` file in `docs/` |
| Chunk | `rag/loader.py` | 120-word windows with 30-word overlap |
| Index | `rag/retriever.py` | BM25 keyword index over the chunks |
| Retrieve | `rag/retriever.py` | Top 3 chunks per question |
| Generate | `rag/generator.py` | Gemini answers from the chunks; without an API key, the best-matching sentences are returned |

## Run it

```bash
# optional: copy .env.example to .env and set GEMINI_API_KEY

python app.py                          # command-line chat
python app.py --web                    # browser UI at http://localhost:8000
python app.py --ask "What is BM25?"    # one-off question
python -m unittest discover tests      # run tests
```

Options: `--docs <folder>`, `--port <number>`. All defaults are in `.env`.

## Possible extensions

- Swap BM25 for embeddings and a vector store (FAISS, Chroma)
- Add PDF support with `pypdf`
- Add re-ranking or citation highlighting
