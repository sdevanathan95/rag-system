"""
ingestion pipeline
"""
import os
import tempfile
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling_core.transforms.chunker import HierarchicalChunker

from config.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, CHROMA_PATH


class IngestionPipeline:

    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.chunker = HierarchicalChunker()
        pipeline_options = PdfPipelineOptions(allow_external_plugins=True)
        pipeline_options.do_ocr = True
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

    def ingest_file(self, uploaded_file):
        """Ingest a single uploaded file into the vector store"""
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if suffix in [".pdf", ".txt", ".md", ".docx"]:
                loader = DoclingLoader(
                    tmp_path,
                    converter=self.converter,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=self.chunker
                )
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            chunks = loader.load()

            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
                # ChromaDB only accepts str, int, float, bool, list, None
                chunk.metadata = {
                    k: v for k, v in chunk.metadata.items()
                    if isinstance(v, (str, int, float, bool, list, type(None)))
                }

            if os.path.exists(CHROMA_PATH):
                vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
                vectorstore.add_documents(chunks)
            else:
                os.makedirs(CHROMA_PATH, exist_ok=True)
                Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_PATH
                )

            return len(chunks)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)