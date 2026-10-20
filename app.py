import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from retrieval import HybridRetriever, format_docs
from integrity import verify_checksums
from ratelimit import SlidingWindowLimiter
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Groq retires models periodically; llama-3.3-70b-versatile was shut down on
# 16 August 2026. Current catalogue: https://console.groq.com/docs/deprecations
MODEL_NAME = "openai/gpt-oss-120b"

MAX_QUESTION_CHARS = 1000
MAX_QUESTIONS_PER_MINUTE = 5        # per browser session
MAX_QUESTIONS_PER_MINUTE_ALL = 30   # shared by every visitor of the running app

st.set_page_config(page_title="RAG Compliance Chatbot", page_icon="🤖")
st.title("RAG Compliance Chatbot")
st.caption("Ask questions about the EU AI Act, NIST CSF 2.0 or NIST AI RMF")

@st.cache_resource
def get_global_limiter():
    return SlidingWindowLimiter(MAX_QUESTIONS_PER_MINUTE_ALL)


def allow_question():
    """Two limits: one per browser session, one shared by everyone."""
    now = time.time()
    recent = [t for t in st.session_state.get("question_times", []) if now - t < 60]
    st.session_state.question_times = recent
    if len(recent) >= MAX_QUESTIONS_PER_MINUTE or not get_global_limiter().allow():
        return False
    recent.append(now)
    return True


@st.cache_resource
def load_chain():
    # The index contains a pickle file, which can run code when loaded. Only load it if its
    # files still match the checksums recorded when the index was built (see ingest.py).
    if not verify_checksums("vectorstore"):
        st.error("The search index failed its integrity check, so the app has stopped.")
        st.stop()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embeddings,
                                   allow_dangerous_deserialization=True)
    # Hybrid (embeddings + BM25) retrieval of 6 passages: 84% hit rate vs 72% for
    # dense top-4 on the 25-question set in eval/ (see README).
    retriever = HybridRetriever(vectorstore, k=6)
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0)

    prompt = PromptTemplate.from_template("""
You are a compliance assistant. Use only the context below to answer the question.
Cite the document and page for each claim, for example (eu-ai-act.pdf, page 52).
If the context does not contain the answer, say you don't know.

Context: {context}

Question: {question}

Answer:""")

    chain = (
        {"context": RunnableLambda(retriever.invoke) | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

chain, retriever = load_chain()
st.info(
    "Your question is sent to Groq, an outside service, to generate the answer. "
    "Do not enter personal or confidential information."
)
query = st.text_input("Ask a compliance question:", max_chars=MAX_QUESTION_CHARS)

if query and len(query) > MAX_QUESTION_CHARS:
    st.error(f"That question is too long (limit {MAX_QUESTION_CHARS} characters).")
elif query and not allow_question():
    st.error("Too many questions just now. Please wait a minute and try again.")
elif query:
    try:
        with st.spinner("Searching documents..."):
            result = chain.invoke(query)
            sources = retriever.invoke(query)
    except Exception as exc:
        st.error(
            "Sorry - the assistant could not answer that just now. "
            "The language model service returned an error."
        )
        st.caption(f"{type(exc).__name__}: this has been logged for investigation.")
        print(f"[rag-compliance-chatbot] query failed ({len(query)} chars): {type(exc).__name__}")
    else:
        st.markdown("### Answer")
        st.write(result)
        with st.expander("Sources"):
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page_label', doc.metadata.get('page', '?'))})")