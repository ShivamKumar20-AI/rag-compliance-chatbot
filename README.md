# rag-compliance-chatbot

A RAG-based (Retrieval-Augmented Generation) chatbot for querying AI compliance documents using LangChain, FAISS, and Groq LLM with a Streamlit web UI.

## Live Demo Try It Here
🔗 [rag-compliance-chatbot.streamlit.app/]

## Documents Indexed
- 📄 EU AI Act (Regulation EU 2024/1689)
- 📄 NIST Cybersecurity Framework 2.0 (NIST CSF 2.0)
- 📄 NIST AI Risk Management Framework (AI RMF 1.0)

## Tech Stack
- **LangChain** — RAG pipeline and document loading
- **FAISS** — Vector similarity search
- **HuggingFace** — Sentence embeddings (all-MiniLM-L6-v2)
- **Groq** — LLM inference (openai/gpt-oss-120b)
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


## Evaluation

`evaluate_retrieval.py` measures retrieval quality on 25 questions (`eval/questions.json`) across the three documents. Each question names the source PDF and an exact phrase containing the answer; a question is a hit at k if any of the top-k retrieved chunks contains that phrase. It does not call the LLM, so no API key is needed.

| Index | Hit@1 | Hit@4 | Hit@8 | MRR |
|---|---|---|---|---|
| Original text cleaning | 20% | 44% | 56% | 0.301 |
| Fixed text cleaning (current) | 48% | 72% | 72% | 0.567 |

The first run showed that the original cleaning step removed single spaces between lowercase letters, turning text into strings like `purposessetoutinthatpointonly`. Fixing it raised Hit@4 from 44% to 72%.

Remaining weaknesses: questions about specific numbers and dates (fine amounts, application dates) and questions that match the CSF table of contents or glossary instead of the relevant section. Hit@8 equals Hit@4, so a larger k will not fix this. Hybrid keyword plus vector retrieval is the likely next step. This evaluation covers retrieval only, not answer quality.

```bash
python evaluate_retrieval.py                    # current pipeline
python evaluate_retrieval.py --cleaning legacy  # original cleaning, for comparison
```

## Security and Responsible AI Considerations

- Uses public EU AI Act and NIST CSF 2.0 documentation only.
- Designed as an informational portfolio prototype, not legal or compliance advice.
- Provides source references so users can verify relevant material.
- Does not intentionally collect or use real personal data.
- Future improvements: authentication, access controls, prompt-injection testing,
  output monitoring and evaluation of response quality.
