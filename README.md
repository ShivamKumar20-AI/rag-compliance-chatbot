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
- **rank_bm25** — Keyword search, fused with FAISS results (hybrid retrieval)
- **HuggingFace** — Sentence embeddings (all-MiniLM-L6-v2)
- **Groq** — LLM inference (openai/gpt-oss-120b)
- **Streamlit** — Web UI
- **PyPDF** — PDF text extraction

## How It Works
1. PDFs are loaded and cleaned to fix formatting issues
2. Text is split into chunks and embedded using HuggingFace
3. Chunks are stored in a FAISS vector index
4. A query is matched two ways: FAISS embedding search and BM25 keyword search, merged by reciprocal rank fusion
5. The top 6 chunks, labelled with document and page, are passed to the Groq LLM, which is asked to cite them

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

| Pipeline | Hit@1 | Hit@4 | Hit@6 | MRR |
|---|---|---|---|---|
| Original text cleaning, dense retrieval | 20% | 44% | n/a | 0.301 |
| Fixed text cleaning, dense retrieval | 48% | 72% | 72% | 0.567 |
| Fixed cleaning, hybrid retrieval (**current, k=6**) | 60% | 68% | **84%** | 0.669 |

Two changes produced these gains:

1. **Text cleaning.** The original step removed single spaces between lowercase letters, turning text into strings like `purposessetoutinthatpointonly`. Fixing it raised Hit@4 from 44% to 72%.
2. **Hybrid retrieval.** `retrieval.py` merges FAISS embedding search with BM25 keyword search using reciprocal rank fusion, and the app passes 6 passages to the model. Hit@6 rises from 72% (dense) to 84%, and the right passage is ranked first more often (Hit@1 48% to 60%). At k=4 hybrid is one question behind dense, so the gain comes from k=6, not from fusion alone.

Remaining weaknesses: questions whose answer is a specific number or date that the question does not repeat (fine amounts, application dates) and questions about CSF structure, where the table of contents competes with the real section. This measures retrieval only, not answer quality: the LLM step is not scored.

```bash
python evaluate_retrieval.py                          # dense retrieval
python evaluate_retrieval.py --retriever hybrid       # current pipeline
python evaluate_retrieval.py --cleaning legacy        # original cleaning, for comparison
pytest                                                # unit tests (no model downloads)
```

## Security and Responsible AI Considerations

- Uses public EU AI Act and NIST CSF 2.0 documentation only.
- Designed as an informational portfolio prototype, not legal or compliance advice.
- Provides source references so users can verify relevant material.
- Does not intentionally collect or use real personal data.
- Future improvements: authentication, access controls, prompt-injection testing,
  output monitoring and evaluation of response quality.
