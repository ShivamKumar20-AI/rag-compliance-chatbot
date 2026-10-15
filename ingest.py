from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re

PDF_PATHS = [
    "docs/eu-ai-act.pdf",
    "docs/NIST.AI.100-1.pdf",
    "docs/NIST.CSWP.29.pdf"
]

def clean_text(text):
    # Collapse line breaks and repeated whitespace only. An earlier version also
    # removed single spaces between lowercase letters, which glued words together
    # ("theuseofbiometric...") and cut retrieval hit@4 from 72% to 44% (see
    # evaluate_retrieval.py).
    return re.sub(r'\s+', ' ', text).strip()

def ingest():
    docs = []
    for path in PDF_PATHS:
        print(f"Loading {path}...")
        loader = PyPDFLoader(path)
        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
        docs.extend(pages)

    print(f"Splitting {len(docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    print(f"Building FAISS index from {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print(f"Done! Vectorstore saved with {len(chunks)} chunks.")

if __name__ == "__main__":
    ingest()