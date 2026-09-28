# RAG Compliance Chatbot

RAG‑based chatbot that answers questions about AI regulations (EU AI Act, NIST CSF, NIST AI RMF) with citations.  
🚀 **Live demo:** https://rag-compliance-chatbot.streamlit.app/

---

## What it does

- Lets users ask natural‑language questions about AI governance documents.
- Retrieves relevant passages from:
  - EU AI Act
  - NIST Cybersecurity Framework (CSF)
  - NIST AI Risk Management Framework (AI RMF)
- Generates concise answers with **article/section citations**.
- Provides a chat interface for multi‑turn conversations.

---

## Model & approach

**Task:** Question answering over long regulatory documents.

**Data**
- PDFs / text of:
  - EU AI Act
  - NIST CSF
  - NIST AI RMF
- Chunked into smaller passages with metadata (source, article/section).

**RAG pipeline**
- Embedding model (e.g., sentence‑transformers) + **FAISS** index for retrieval.
- **LangChain** to orchestrate:
  - Query embedding
  - Retrieval of top‑k relevant chunks
  - Prompt construction with context
- **Groq LLM** (e.g., Llama 3) for answer generation.
- Prompt instructs the model to:
  - Answer using only the retrieved context.
  - Cite the exact article/section IDs where possible.

**Interface**
- **FastAPI** backend exposing a chat/completion endpoint.
- **Streamlit** frontend for a simple chat UI.

---

## Results

- Answers include **explicit citations** (e.g., “EU AI Act, Article X”).
- In manual testing, time to locate relevant clauses drops from minutes to seconds.
- Supports multi‑turn conversation with context retention across messages.

---

## How to run locally

**Requirements**
- Python 3.9+
- API key for Groq (or your chosen LLM provider).

**Setup**

```bash
# Clone the repo
git clone [https://github.com/ShivamKumar20-AI/rag-compliance-chatbot.git](https://github.com/ShivamKumar20-AI/rag-compliance-chatbot.git)
cd rag-compliance-chatbot

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Configure environment**

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**Run backend (FastAPI)**

```bash
uvicorn app.main:app --reload
```

(Adjust `app.main:app` to match your actual FastAPI app path.)

**Run frontend (Streamlit)**

```bash
streamlit run ui.py
```

(Adjust `ui.py` to your actual Streamlit file name.)

Then open the Streamlit URL in your browser (usually `http://localhost:8501`).

---

## Example usage

**Via the Streamlit UI**

1. Start the backend and frontend as above.
2. In the chat UI, ask questions like:
   - “What are the main obligations for high‑risk AI systems under the EU AI Act?”
   - “How does NIST AI RMF suggest managing AI risks?”
3. The bot returns an answer with citations, e.g.:
   - “Under the EU AI Act, high‑risk systems must have a risk management system, data governance, and technical documentation (Article X, Section Y).”

**Via the API (example curl)**

```bash
curl http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does the EU AI Act say about high-risk AI systems?",
    "conversation_id": "test-123"
  }'
```

(Adjust endpoint and payload to match your actual API schema.)

---

## Deployment

- Backend: FastAPI service (Dockerised).
- Frontend: Streamlit app.
- Hosted on a cloud platform (e.g., Fly.io, Render, or similar).
- Live demo: https://rag-compliance-chatbot.streamlit.app/

---

## Limitations & next steps

**Limitations**
- Retrieval can miss relevant passages for very long or complex articles.
- No formal evaluation harness (e.g., Q/A pairs with ground truth).
- Logging and PII handling are minimal; not yet enterprise‑ready.

**Next steps**
- Build a small evaluation set of Q/A pairs and measure answer quality.
- Improve chunking and metadata filtering (e.g., by article/section).
- Add structured logging and basic privacy controls.
- Extend to a multi‑tenant setup with user accounts and usage tracking.

---
