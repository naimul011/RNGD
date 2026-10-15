# src/rngd/vectorstore.py
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

def retrieve_multi_as_context(self, queries: list[str], k_each: int = 2) -> str:
    """Issues several narrower queries and merges deduped hits. On a small,
    multi-topic corpus like this one, a single broad query tends to be
    dominated by whichever section best matches the query's dominant
    terms, crowding out other sections an agent also needs (e.g. a
    zoning query for "setbacks height FAR parking fire ADA" can rank
    Accessibility above Bulk Standards even though both are needed) --
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
