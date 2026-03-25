"""
DOCUMENT INGESTION PIPELINE
Loads documents, chunks them, creates embeddings, stores in ChromaDB
"""

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DOCUMENTS_DIR = "./documents"
CHROMA_DB_DIR = "./chroma_db"
OLLAMA_URL = "http://localhost:11434"

def load_documents(directory: str) -> list:
    """ Load all supported documents from a directory. """

    documents = []
    dir_path = Path(directory)

    # Load PDFs
    for pdf_file in dir_path.glob("**/*.pdf"):
        print(f"Loading PDF: {pdf_file}")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    # Load text files
    for txt_file in dir_path.glob("**/*.txt"):
        print(f"Loading TXT: {txt_file}")
        loader = TextLoader(str(txt_file))
        documents.extend(loader.load())

    # Load markdown files
    for md_file in dir_path.glob("**/*.md"):
        print(f"Loading MD: {md_file}")
        loader = TextLoader(str(md_file))
        documents.extend(loader.load())

    print(f"\nLoaded {len(documents)} document(s)")
    return documents

def chunk_documents(documents: list) -> list:
    """Split documents into smaller chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: list) -> Chroma:
    """Create embeddings and store in ChromaDB"""
    print("Creating embeddings (this may take a while)...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_URL
    )

    vectorestore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    print(f"Vector store created at {CHROMA_DB_DIR}")
    return vectorestore

def main():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)

    if not any(Path(DOCUMENTS_DIR).glob("**/*.*")):
        print(f"No documents found in {DOCUMENTS_DIR}")
        print("Add some PDFs, text files, or markdown files and run again.")
        return
    
    documents = load_documents(DOCUMENTS_DIR)
    chunks = chunk_documents(documents)
    create_vector_store(chunks)

    print("\n ingetion complete! Run query.py to ask questions.")

if __name__ == "__main__":
    main()