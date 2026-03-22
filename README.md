# RAG PDF Chatbot

A conversational AI app that lets you upload a PDF and ask questions about its content. Built using LangChain, Groq, ChromaDB, and Streamlit.

## How It Works

1. Upload a PDF via the UI
2. The document is parsed, split into chunks, and embedded into a local Chroma vector store
3. When you ask a question, relevant chunks are retrieved and passed to the LLM as context
4. The LLM answers based strictly on the document content

## Tech Stack

- **LangChain** — RAG pipeline, document loaders, text splitters, retrieval chain
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
- Chunking strategies and why chunk size/overlap matters
- How vector similarity search retrieves relevant context
- Managing Streamlit session state to avoid re-processing on every interaction
