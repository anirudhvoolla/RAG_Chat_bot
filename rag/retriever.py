"""Step 3-4: BM25 keyword index and top-k retrieval (pure Python)."""

import math
import re
from collections import Counter

STOPWORDS = set(
    "a an and are as at be by for from has have how i in is it its of on or "
    "that the this to was what when where which who why will with you your do "
    "does can about into than then so if not no".split()
)


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks  # list of {"source": str, "text": str}
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

    def search(self, query, k=3):
        """Return the top-k [(score, chunk)] pairs, best first."""
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
