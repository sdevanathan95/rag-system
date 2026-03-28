"""
RAG System Web Interface - Step 1: Basic Setup
"""

import streamlit as st
import requests
import os
import tempfile
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide"
)

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
CHROMA_PATH = "./chroma_db"

def check_ollama_connection():
    """Check if Ollama is accessible"""

    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.status_code
    except:
        return False
    
def get_ollama_models():
    """ Get llist of available Ollama models"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except:
        pass
    return []

def get_document_count():
    """ Get number of chunks in the vector store"""
    if not os.path.exists(CHROMA_PATH):
        return 0
    try:
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vectorstore._collection.count()
    except:
        return 0
    
def ingest_file(uploaded_file):
    """Ingest a single uploaded file into the vector store"""
    # Get file extension
    suffix = Path(uploaded_file.name).suffix.lower()

    # Save to temp file (loaders need a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Load based on file type
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix in [".txt", ".md"]:
            loader = TextLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        documents = loader.load()

        # Add source filename to metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # Add to vector store
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

        if os.path.exists(CHROMA_PATH):
            # Add to existing store
            vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            vectorstore.add_documents(chunks)
        else:
            # create new store
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_PATH
            )

        return len(chunks)
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)

def query_rag(question: str):
    """Query the RAG system - returns (answer, source_documents)"""
    # Load vector store
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.7)

    # Create promt
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that answers questions based on the provided context.
    Use ONLY the context below to answer the question. If you cannot find the answer
    in the context, say so clearly.

    Context:
    {context}

    Question: {question}

    Answer:""")

    # Build the chain (LCEL style)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Get source documents for display
    source_docs = retriever.invoke(question)

    # Get answer
    answer = chain.invoke(question)

    return answer, source_docs


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

with st.sidebar:
    st.header("System Status")

    if check_ollama_connection():
        st.success("Ollama is connected")

        models = get_ollama_models()
        if models:
            st.subheader("Available Models")
            for model in models:
                if model.startswith(LLM_MODEL) or model.startswith(EMBEDDING_MODEL):
                    st.write(f"- `{model}`")
                else:
                    st.write(f"- `{model}`")
    else:
        st.error(f"Ollama Offline")
    
    st.divider()

    # File upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, MD"
    )

    if uploaded_files:
        if st.button("Ingest Files", use_container_width=True):
            total_chunks=0

            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}"):
                    try:
                        chunks = ingest_file(file)
                        total_chunks += chunks
                        st.success(f"{file.name}: {chunks} chunks")
                    except Exception as e:
                        st.error(f"{file.name}: {str(e)}")
            
            if total_chunks > 0:
                st.success(f"Added {total_chunks} chunks")
                st.rerun()
    
    st.divider()

    # Configuration display
    st.subheader("Configuration")
    st.write(f"**LLM:** `{LLM_MODEL}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.write(f"**Ollama URL:** `{OLLAMA_BASE_URL}`")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.title(" RAG System")
st.write("Ask Questions about your documents using local AI")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View Sources ({len(message['sources'])} chunks)"):
                for i, source in enumerate(message["sources"], 1):
                    source_name = source.metadata.get("source", "Unknown")
                    page_num = source.metadata.get("page", None)
                    
                    # Header with source info
                    if page_num is not None:
                        st.markdown(f"**Source {i}:** `{source_name}` (page {page_num + 1})")
                    else:
                        st.markdown(f"**Source {i}:** `{source_name}`")
                    
                    # Show chunk content (truncated if long)
                    content = source.page_content
                    if len(content) > 500:
                        content = content[:500] + "..."
                    st.code(content, language=None)
                    
                    if i < len(message["sources"]):
                        st.divider()

# Chat input
if prompt := st.chat_input("Ask a question..."):
    if not check_ollama_connection():
        st.error("⚠️ Ollama is not connected!")
    elif get_document_count() == 0:
        st.warning("⚠️ No documents indexed. Run ingest.py first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get RAG response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = query_rag(prompt)
                st.write(answer)
                
                # Show sources
                with st.expander(f"📚 View Sources ({len(sources)} chunks)"):
                    for i, source in enumerate(sources, 1):
                        source_name = source.metadata.get("source", "Unknown")
                        page_num = source.metadata.get("page", None)
                        
                        if page_num is not None:
                            st.markdown(f"**Source {i}:** `{source_name}` (page {page_num + 1})")
                        else:
                            st.markdown(f"**Source {i}:** `{source_name}`")
                        
                        content = source.page_content
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.code(content, language=None)
                        
                        if i < len(sources):
                            st.divider()
        
        # Add to history (including sources)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })