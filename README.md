# RAG PDF Chatbot

A conversational AI app that lets you upload multiple PDFs and ask questions about their content. Supports follow-up questions with full chat history awareness and source citations.

## Features

- **Multi-document support** — upload and query multiple PDFs simultaneously
- **Conversational memory** — follow-up questions work correctly using history-aware retrieval
- **Source citations** — every answer shows the source document and page number it came from

## How It Works

1. Upload one or more PDFs via the UI
2. Documents are parsed, split into chunks, embedded, and stored in a local Chroma vector store
3. When you ask a question, it gets rephrased based on chat history for accurate retrieval
4. Relevant chunks are retrieved and passed to the LLM as context
5. The LLM answers based strictly on the document content, with sources shown below

## Tech Stack

- **LangChain** — RAG pipeline, history-aware retriever, document loaders, text splitters
- **Groq** — LLM inference (Qwen3-32B)
- **ChromaDB** — Local vector store for storing and searching embeddings
- **HuggingFace Embeddings** — `sentence-transformers/all-mpnet-base-v2`
- **Streamlit** — Chat UI
- **PyMuPDF** — PDF parsing

## Project Structure

```
RAG PDF Chatbot/
├── preprocessing.py      # PDF loading and text splitting
├── vectorstore.py        # Embedding and Chroma vector store creation
├── app.py                # Streamlit UI and RAG chain
├── .env                  # API keys (not committed)
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Groq API key
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Run the app
   ```
   streamlit run app.py
   ```

## What I Learned

- How RAG pipelines work end-to-end
- Why chunking strategy (size and overlap) directly impacts answer quality
- How `create_history_aware_retriever` rephrases follow-up questions into standalone queries before retrieval — and why this matters
- How LangChain's `HumanMessage` and `AIMessage` objects manage conversation state
- How to extract source metadata from retrieved documents for citations