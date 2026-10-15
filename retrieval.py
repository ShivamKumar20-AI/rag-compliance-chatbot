"""Hybrid retrieval: dense (FAISS) plus keyword (BM25), merged with reciprocal rank fusion.

Dense embeddings find passages with similar meaning but are weak on exact
numbers, dates and defined terms ("EUR 35 000 000", "2 August 2026"). BM25
matches those exact tokens. Fusing the two rankings recovers questions that
either method misses alone (see evaluate_retrieval.py).
"""
import re

from rank_bm25 import BM25Okapi

RRF_K = 60
POOL = 20  # candidates taken from each ranking before fusion


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever:
    def __init__(self, vectorstore, k=4):
        self.vectorstore = vectorstore
        self.k = k
        self.docs = list(vectorstore.docstore._dict.values())
        self.bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])

    def invoke(self, query):
        dense = self.vectorstore.similarity_search(query, k=POOL)
        dense_ids = [self._doc_id(d) for d in dense]

        scores = self.bm25.get_scores(tokenize(query))
        # Only passages that share at least one query term take part; ranking the
        # zero-score passages would add arbitrary index-order noise to the fusion.
        keyword_ids = [i for i in sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)[:POOL]
                       if scores[i] > 0]

        fused = {}
        for ranking in (dense_ids, keyword_ids):
            for rank, doc_id in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(fused, key=fused.get, reverse=True)[: self.k]
        return [self.docs[i] for i in best]

    def _doc_id(self, doc):
        # similarity_search returns the stored Document objects' content; match on text + page.
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
        if not hasattr(self, "_by_key"):
            self._by_key = {
                (d.metadata.get("source"), d.metadata.get("page"), d.page_content): i
                for i, d in enumerate(self.docs)
            }
        return self._by_key[key]


def format_docs(docs):
    """Label each passage with its source and printed page so the model can cite it."""
    parts = []
    for d in docs:
        name = str(d.metadata.get("source", "unknown")).split("/")[-1]
        page = d.metadata.get("page_label", d.metadata.get("page", "?"))
        parts.append(f"[{name}, page {page}]\n{d.page_content}")
    return "\n\n".join(parts)
