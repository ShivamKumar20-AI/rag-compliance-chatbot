import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Groq retires models periodically; llama-3.3-70b-versatile was shut down on
# 16 August 2026. Current catalogue: https://console.groq.com/docs/deprecations
MODEL_NAME = "openai/gpt-oss-120b"

st.set_page_config(page_title="RAG Compliance Chatbot", page_icon="🤖")
st.title("RAG Compliance Chatbot")
st.caption("Ask questions about the EU AI Act, NIST CSF 2.0 or NIST AI RMF")

@st.cache_resource
def load_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("vectorstore", embeddings,
                                   allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(model_name=MODEL_NAME, temperature=0)

    prompt = PromptTemplate.from_template("""
You are a compliance assistant. Use the context below to answer the question.
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever

chain, retriever = load_chain()
query = st.text_input("Ask a compliance question:")

if query:
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
        print(f"[rag-compliance-chatbot] query failed: {type(exc).__name__}: {exc}")
    else:
        st.markdown("### Answer")
        st.write(result)
        with st.expander("Sources"):
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Unknown')} (page {doc.metadata.get('page', '?')})")