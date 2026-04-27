"""
helper
"""
import requests
import os

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from config.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, CHROMA_PATH

class Helper:

    def __init__(self):
        pass

    def delete_document(self, source_name: str):
        """Delete all chunks from a specific document"""
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

            # Get IDs of all chunks with this source
            collection = vectorstore._collection
            results = collection.get(include=["metadatas"], where={"source": source_name})
            
            if results["ids"]:
                collection.delete(ids=results["ids"])
                return len(results["ids"])
            return 0
        except Exception as e:
            raise e
        
    def get_indexed_documents(self):
        """Get list of unique documents and their chunk counts"""
        if not os.path.exists(CHROMA_PATH):
            return {}
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

            # Get all documents from the collection
            collection = vectorstore._collection
            results = collection.get(include=["metadatas"])

            # Count chunks per source
            doc_counts = {}
            for metadata in results["metadatas"]:
                source = metadata.get("source", "Unknown")
                doc_counts[source] = doc_counts.get(source, 0) + 1
            
            return doc_counts
        except:
            return {}
        
    def check_ollama_connection(self):
        """Check if Ollama is accessible"""

        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code
        except:
            return False
    
    def get_ollama_models(self):
        """ Get llist of available Ollama models"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass
        return []

    def get_document_count(self):
        """ Get number of chunks in the vector store"""
        if not os.path.exists(CHROMA_PATH):
            return 0
        try:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            return vectorstore._collection.count()
        except:
            return 0