from preprocessing import pdf_loader, text_split
from vectorstore import create_vectorstore

from langchain_classic import hub
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_retries=2
)

st.title("PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and "retrieval_chain" not in st.session_state:
    with st.spinner("Processing PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        docs = pdf_loader("temp.pdf")
        chunks = text_split(docs)
        retriever = create_vectorstore(chunks)
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        st.session_state['retrieval_chain'] = retrieval_chain

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about your PDF")

if user_input and "retrieval_chain" in st.session_state:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    result = st.session_state['retrieval_chain'].invoke({"input": user_input})
    answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)