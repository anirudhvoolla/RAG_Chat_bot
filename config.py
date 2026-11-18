"""Central configuration. Reads settings from the .env file and environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_env(path=BASE_DIR / ".env"):
    """Minimal .env parser (KEY=VALUE per line). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

# LLM
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

# Retrieval
DOCS_DIR = BASE_DIR / os.environ.get("DOCS_DIR", "docs")
CHUNK_WORDS = int(os.environ.get("CHUNK_WORDS", 120))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 30))
TOP_K = int(os.environ.get("TOP_K", 3))

# Web
PORT = int(os.environ.get("PORT", 8000))
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
