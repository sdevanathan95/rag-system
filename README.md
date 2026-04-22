# RAG System

A local Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, Docling, and Ollama. Upload documents and ask questions — answers are grounded in your documents using a locally running LLM with no data leaving your machine.

## Features

- Upload PDF, TXT, MD, and DOCX files
- Document-aware querying — restrict questions to specific files
- Multi-query retrieval for better context coverage
- MMR search for diverse, non-redundant chunks
- Powered entirely by local models via Ollama

## Project Structure

```
rag-system/
├── src/
│   ├── app.py                        # Streamlit web interface
│   ├── config/
│   │   └── config.py                 # Model and path configuration
│   ├── ingestion/
│   │   └── ingestion_pipeline.py     # Document loading, chunking, embedding
│   ├── generation/
│   │   └── generation_pipeline.py    # Retrieval and LLM answer generation
│   └── helper/
│       └── helper.py                 # Ollama connection and vector store utils
├── storage/
│   └── chroma_db/                    # Persisted vector store
├── requirements.txt
└── run.sh
```

## Requirements

- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com/) running in Docker
- Python 3.12+

## Setup

**1. Start Ollama with GPU support:**
```bash
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  --restart unless-stopped \
  ollama/ollama
```

**2. Pull required models:**
```bash
docker exec ollama ollama pull gemma3:4b
docker exec ollama ollama pull nomic-embed-text:v1.5
```

**3. Create and activate virtual environment:**
```bash
python3 -m venv ragEnv
source ragEnv/bin/activate
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

**5. Run the app:**
```bash
sh run.sh
```

Open your browser at `http://localhost:8501` (or your WSL IP if running on WSL2).

## Configuration

Edit `src/config/config.py` to change models or settings:

```python
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"
```

## Usage

1. Upload documents using the sidebar
2. Click **Ingest Files** to process and store them
3. Optionally select specific documents under **Query Scope** to restrict answers to those files
4. Ask questions in the chat input

## How It Works

1. **Ingestion** — Documents are parsed with Docling, split into 700-character chunks with 120-character overlap, embedded with `nomic-embed-text`, and stored in ChromaDB
2. **Retrieval** — Each question is rephrased into 3 alternate queries, all are searched using MMR retrieval, and results are deduplicated
3. **Generation** — Retrieved chunks are passed as context to the LLM which answers using only the provided context
