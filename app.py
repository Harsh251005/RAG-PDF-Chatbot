from preprocessing import pdf_loader, text_split
from vectorstore import create_vectorstore

from langchain_classic import hub
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_retries=2
)

st.title("PDF Chatbot")

uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and "retrieval_chain" not in st.session_state:
    with st.spinner("Processing PDFs..."):
        pdf_paths = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            pdf_paths.append(temp_path)

        docs = pdf_loader(pdf_paths)
        chunks = text_split(docs)
        retriever = create_vectorstore(chunks)

        rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
        history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        retrieval_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
        st.session_state['retrieval_chain'] = retrieval_chain
        st.session_state['chat_history'] = []

        for path in pdf_paths:
            os.remove(path)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask a question about your PDFs")

if user_input and "retrieval_chain" in st.session_state:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    result = st.session_state['retrieval_chain'].invoke({
        "input": user_input,
        "chat_history": st.session_state['chat_history']
    })

    answer = result["answer"]

    st.session_state['chat_history'].append(HumanMessage(content=user_input))
    st.session_state['chat_history'].append(AIMessage(content=answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)

        with st.expander("Sources"):
            for doc in result["context"]:
                st.write(f"**File:** {doc.metadata.get('file_path', 'Unknown')} | **Page:** {doc.metadata.get('page', 'Unknown')}")
                st.write(doc.page_content)
                st.divider()