"""
RAG System Web Interface
"""

import streamlit as st

from config.config import OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL
from ingestion.ingestion_pipeline import IngestionPipeline
from generation.generation_pipeline import GenerationPipeline
from helper.helper import Helper

# Page Configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide"
)

ingestionPipeline = IngestionPipeline()
generationPipeline = GenerationPipeline()
helper = Helper()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]

# if st.session_state.pop("rerun_after_ingest", False):
#         st.rerun()

with st.sidebar:

    st.header("System Status")

    if helper.check_ollama_connection():
        st.success("Ollama is connected")

        models = helper.get_ollama_models()
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

    #Document management section
    st.subheader("Indexed Documents")
    indexed_documents = helper.get_indexed_documents()

    if indexed_documents:
        for doc_name, chunk_count in indexed_documents.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"{doc_name}")
                st.caption(f"{chunk_count} chunks")
            with col2:
                # Unique key for each button
                if st.button(label="🗑️", key=f"del_{doc_name}", help=f"Delete {doc_name}"):
                    try:
                        deleted = helper.delete_document(doc_name)
                        st.toast(f"Deleted {chunk_count} chunks")
                        st.rerun()
                    except Exception as e:
                        st.toast(f"Error {str(e)}")
    else:
        st.info("No docs indexed yet")

    # File upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, MD, DOCX"
    )

    if uploaded_files:
        if st.button("Ingest Files", use_container_width=True):
            total_chunks=0

            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}"):
                    try:
                        chunks = ingestionPipeline.ingest_file(file)
                        total_chunks += chunks
                        st.toast(f"{file.name}: {chunks} chunks", icon="✅")
                    except Exception as e:
                        st.toast(f"{file.name}: {str(e)}", icon="❌")
            
            if total_chunks > 0:
                st.toast(f"Added {total_chunks} chunks")
                # st.session_state["rerun_after_ingest"] = True
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
    if not helper.check_ollama_connection():
        st.error("⚠️ Ollama is not connected!")
    elif helper.get_document_count() == 0:
        st.warning("⚠️ No documents indexed. Upload documents first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get RAG response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = generationPipeline.query_rag(prompt)
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