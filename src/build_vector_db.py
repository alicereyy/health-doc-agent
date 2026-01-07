# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os
from pathlib import Path


def load_documents_from_data():
    """Load all documents from the data/ directory"""
    docs = []

    # Load pdf documents from data/ directory
    pdf_files = Path("data/").glob("*.pdf")
    for pdf_path in pdf_files:
        print(f"Loading PDF : {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    return docs


def main():
    """Build and persist Chroma vector store"""

    # Load documents
    documents = load_documents_from_data()
    print(f"Number of documents loaded : {len(documents)}")
    if not documents:
        print("No documents found in the data/ directory.")
        return

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # ~1000 characters per chunk
        chunk_overlap=200,    # 200 characters overlap
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Number of chunks after splitting : {len(split_docs)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Model for 384-dim vectors, good for general use
        encode_kwargs={'normalize_embeddings': True}  # Normalize vectors, important for cosine similarity
    )

    # Build Chroma vector store
    persist_directory = "chroma_db"

    # Check if directory exists, if so, remove it to avoid conflicts
    if os.path.exists(persist_directory):
        print("Warning: Persist directory exists. Overwriting...")
        import shutil
        shutil.rmtree(persist_directory)

    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    main()
