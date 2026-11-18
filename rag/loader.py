"""Step 1-2: load documents from disk and split them into overlapping chunks."""

import re
import sys
from pathlib import Path


def load_documents(folder):
    """Return [(filename, text)] for every .txt / .md file under `folder`."""
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            # turn markdown headings ("## Title") into plain sentences ("Title.")
            text = re.sub(r"^#+\s*(.+?)\s*$", r"\1.", text, flags=re.MULTILINE)
            docs.append((path.name, text))
    return docs


def chunk_text(text, size, overlap):
    """Split text into windows of `size` words that share `overlap` words."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def build_chunks(folder, size, overlap):
    """Return a list of {"source": filename, "text": chunk} dicts."""
    chunks = []
    for name, text in load_documents(folder):
        for piece in chunk_text(text, size, overlap):
            chunks.append({"source": name, "text": piece})
    if not chunks:
        sys.exit(f"No .txt or .md files found in '{folder}'. Add some documents first.")
    return chunks
