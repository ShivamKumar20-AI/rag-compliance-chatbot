"""Retrieval evaluation for the RAG chatbot.

Each question in eval/questions.json has a source PDF and an exact phrase from
the source text that contains the answer. A question counts as a hit at k if
any of the top-k retrieved chunks (from that source) contains the phrase, with
case, spacing and punctuation ignored. This measures retrieval only: it does
not call the LLM, so it needs no API key.

    python evaluate_retrieval.py                  # production cleaning (ingest.clean_text)
    python evaluate_retrieval.py --cleaning legacy  # the original regex that removed spaces
"""
import argparse
import json
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingest import PDF_PATHS, clean_text

KS = (1, 4, 8)


def legacy_clean_text(text):
    text = re.sub(r"(?<=[a-z])\s(?=[a-z])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_index(cleaner):
    docs = []
    for path in PDF_PATHS:
        for page in PyPDFLoader(path).load():
            page.page_content = cleaner(page.page_content)
            docs.append(page)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaning", choices=["production", "legacy"], default="production")
    args = ap.parse_args()
    cleaner = clean_text if args.cleaning == "production" else legacy_clean_text

    questions = json.load(open("eval/questions.json"))
    store, n_chunks = build_index(cleaner)
    print(f"Cleaning: {args.cleaning} | chunks: {n_chunks} | questions: {len(questions)}\n")

    hits = {k: 0 for k in KS}
    rr = 0.0
    misses = []
    for q in questions:
        results = store.similarity_search(q["question"], k=max(KS))
        gold = norm(q["answer_phrase"])
        rank = next(
            (i + 1 for i, d in enumerate(results)
             if d.metadata.get("source") == q["source"] and gold in norm(d.page_content)),
            None,
        )
        for k in KS:
            hits[k] += rank is not None and rank <= k
        rr += 1 / rank if rank else 0
        if rank is None or rank > 4:
            misses.append((q["id"], q["question"], rank))

    n = len(questions)
    for k in KS:
        print(f"Hit@{k}: {hits[k]}/{n} = {hits[k] / n:.1%}")
    print(f"MRR: {rr / n:.3f}")
    if misses:
        print("\nNot in top 4:")
        for i, text, rank in misses:
            print(f"  Q{i}: {text}  (rank {rank or 'not in top 8'})")


if __name__ == "__main__":
    main()
