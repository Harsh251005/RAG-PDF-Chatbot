from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_loader(pdf_path):
    all_docs = []
    for docs in pdf_path:
        loader = PyMuPDFLoader(docs, mode="single")
        doc = loader.load()
        all_docs.extend(doc)
    return all_docs

def text_split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(document)
    return texts