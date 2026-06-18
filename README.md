# rag-compliance-chatbot

A RAG-based (Retrieval-Augmented Generation) chatbot for querying AI compliance documents using LangChain, FAISS, and Groq LLM with a Streamlit web UI.

## Live Demo
🔗 [Try it here](YOUR_STREAMLIT_URL)

## Documents Indexed
- 📄 EU AI Act (Regulation EU 2024/1689)
- 📄 NIST Cybersecurity Framework 2.0 (NIST CSF 2.0)
- 📄 NIST AI Risk Management Framework (AI RMF 1.0)

## Tech Stack
- **LangChain** — RAG pipeline and document loading
- **FAISS** — Vector similarity search
- **HuggingFace** — Sentence embeddings (all-MiniLM-L6-v2)
- **Groq** — LLM inference (llama-3.3-70b-versatile)
- **Streamlit** — Web UI
- **PyPDF** — PDF text extraction

## How It Works
1. PDFs are loaded and cleaned to fix formatting issues
2. Text is split into chunks and embedded using HuggingFace
3. Chunks are stored in a FAISS vector index
4. User queries are matched against the index
5. Relevant chunks are passed to Groq LLM to generate an answer

## Run Locally
```bash
git clone https://github.com/ShivamKumar20-AI/rag-compliance-chatbot.git
cd rag-compliance-chatbot
conda create -n rag-chatbot python=3.10 -y
conda activate rag-chatbot
pip install -r requirements.txt
```

Add your Groq API key to a `.env` file:
