"""
Config File
"""

from pathlib import Path

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"

_RAG_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = str(_RAG_ROOT / "storage" / "chroma_db")