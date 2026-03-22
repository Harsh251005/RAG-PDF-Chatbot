from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_loader(pdf_path):
    loader = PyMuPDFLoader(pdf_path, mode="single")
    docs = loader.load()
    return docs

def text_split(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(document)
    return texts