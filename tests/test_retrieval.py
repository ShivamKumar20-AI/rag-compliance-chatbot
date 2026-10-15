import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import HybridRetriever, format_docs, tokenize
from text_cleaning import clean_text


def make_doc(text, source="docs/a.pdf", page=0, page_label=None):
    meta = {"source": source, "page": page}
    if page_label:
        meta["page_label"] = page_label
    return SimpleNamespace(page_content=text, metadata=meta)


class FakeStore:
    """Stands in for FAISS: returns documents in a fixed 'semantic' order."""

    def __init__(self, docs, semantic_order):
        self.docstore = SimpleNamespace(_dict={i: d for i, d in enumerate(docs)})
        self._docs = docs
        self._order = semantic_order

    def similarity_search(self, query, k):
        return [self._docs[i] for i in self._order[:k]]


def test_clean_text_keeps_word_boundaries():
    # Regression: an earlier regex removed spaces between lowercase words.
    assert clean_text("the use of  biometric\nidentification") == "the use of biometric identification"


def test_tokenize_ignores_case_and_punctuation():
    assert tokenize("EUR 35,000-000!") == ["eur", "35", "000", "000"]


def test_keyword_match_rescues_a_passage_the_embeddings_rank_last():
    docs = [make_doc(f"general text about topic {i}") for i in range(30)]
    docs.append(make_doc("Fines of EUR 35 000 000 apply to prohibited practices."))
    # Embeddings rank the fine passage last of 31; BM25 should rank it first.
    store = FakeStore(docs, semantic_order=list(range(31)))
    retriever = HybridRetriever(store, k=3)
    top = retriever.invoke("EUR 35 000 000 fine")
    assert any("EUR 35 000 000" in d.page_content for d in top)


def test_returns_at_most_k_unique_documents():
    docs = [make_doc(f"passage number {i} about audits") for i in range(10)]
    store = FakeStore(docs, semantic_order=list(range(10)))
    top = HybridRetriever(store, k=4).invoke("audits")
    assert len(top) == 4
    assert len({id(d) for d in top}) == 4


def test_format_docs_labels_source_and_printed_page():
    out = format_docs([make_doc("Some text.", source="docs/eu-ai-act.pdf", page=51, page_label="52")])
    assert out.startswith("[eu-ai-act.pdf, page 52]")
    assert "Some text." in out
