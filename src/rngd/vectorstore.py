"""Lightweight local vector store over the knowledge_base markdown docs.

Tries a real sentence-embedding model first (semantic search); falls back to
TF-IDF (scikit-learn) automatically if the embedding model can't be loaded
(offline, no HF access, etc) or if RNGD_USE_TFIDF=1 is set. Both backends
expose the same `.retrieve(query, k)` interface so the rest of the codebase
never needs to know which one is active.

Everything runs in-process, in memory — no external vector DB service. The
corpus is small (a handful of markdown docs) so re-embedding on startup costs
well under a second with TF-IDF, or a few seconds with sentence-transformers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config


@dataclass
class Chunk:
    text: str
    source: str
    heading: str
    source_kind: str = "synthetic"  # "synthetic" (data/knowledge_base/*.md) or "real" (docs/* — RNGD's own data)


def _load_chunks(kb_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_path in sorted(kb_dir.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        sections = re.split(r"\n(?=#{1,3} )", text)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            heading_match = re.match(r"#{1,3}\s*(.+)", section)
            heading = heading_match.group(1) if heading_match else md_path.stem
            # Further split long sections into ~800 char windows so retrieval
            # doesn't hand an agent an entire multi-topic document.
            for i in range(0, len(section), 900):
                window = section[i : i + 1000]
                if len(window.strip()) < 20:
                    continue
                chunks.append(Chunk(text=window, source=md_path.name, heading=heading))
    return chunks


class _EmbeddingBackend:
    name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[Chunk]):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, k: int) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [
            {
                "text": self.chunks[i].text,
                "source": self.chunks[i].source,
                "heading": self.chunks[i].heading,
                "source_kind": self.chunks[i].source_kind,
                "score": float(scores[i]),
            }
            for i in top
        ]


class _TfidfBackend:
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, k: int) -> list[dict]:
        q_vec = self.vectorizer.transform([query])
        scores = (self.matrix @ q_vec.T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [
            {
                "text": self.chunks[i].text,
                "source": self.chunks[i].source,
                "heading": self.chunks[i].heading,
                "source_kind": self.chunks[i].source_kind,
                "score": float(scores[i]),
            }
            for i in top
        ]


def _load_real_doc_chunks() -> list[Chunk]:
    from . import real_docs

    return [Chunk(text=c.text, source=c.source, heading=c.heading, source_kind="real") for c in real_docs.load_all_chunks()]


class VectorStore:
    def __init__(self, kb_dir: Path | None = None):
        self.kb_dir = kb_dir or config.KB_DIR
        self.chunks = _load_chunks(self.kb_dir) + _load_real_doc_chunks()
        self.real_chunk_count = sum(1 for c in self.chunks if c.source_kind == "real")
        self.backend_name = "tfidf"
        if config.USE_TFIDF:
            self._backend = _TfidfBackend(self.chunks)
        else:
            try:
                self._backend = _EmbeddingBackend(self.chunks)
                self.backend_name = self._backend.name
            except Exception:
                self._backend = _TfidfBackend(self.chunks)
                self.backend_name = "tfidf (embedding backend unavailable)"

    def retrieve(self, query: str, k: int = 4) -> list[dict]:
        return self._backend.retrieve(query, k)

    def retrieve_as_context(self, query: str, k: int = 4) -> str:
        hits = self.retrieve(query, k)
        blocks = []
        for h in hits:
            blocks.append(f"[source: {h['source']} | {h['heading']}]\n{h['text']}")
        return "\n\n".join(blocks)

    def retrieve_multi_as_context(self, queries: list[str], k_each: int = 2) -> str:
        """Issues several narrower queries and merges deduped hits. On a small,
        multi-topic corpus like this one, a single broad query tends to be
        dominated by whichever section best matches the query's dominant
        terms, crowding out other sections an agent also needs (e.g. a
        zoning query for "setbacks height FAR parking fire ADA" can rank
        Accessibility above Bulk Standards even though both are needed) —
        several targeted queries recall more of the relevant corpus than one
        broad query at the same total k.
        """
        seen: set[tuple[str, str]] = set()
        blocks = []
        for q in queries:
            for h in self.retrieve(q, k_each):
                key = (h["source"], h["heading"])
                if key in seen:
                    continue
                seen.add(key)
                blocks.append(f"[source: {h['source']} | {h['heading']}]\n{h['text']}")
        return "\n\n".join(blocks)


_singleton: VectorStore | None = None


def get_vectorstore() -> VectorStore:
    global _singleton
    if _singleton is None:
        _singleton = VectorStore()
    return _singleton
