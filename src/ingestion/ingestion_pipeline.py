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
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.config import OLLAMA_BASE_URL, EMBEDDING_MODEL, CHROMA_PATH


class IngestionPipeline:

    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        pipeline_options = PdfPipelineOptions(allow_external_plugins=True)
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
                loader = DoclingLoader(tmp_path, converter=self.converter)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            chunks = self.splitter.split_documents(documents)
            # Filtering complex metadata that ChromaDB cannot manage
            # TODO: change DB and include complex metadata though it had less significance.
            chunks = filter_complex_metadata(chunks)

            if os.path.exists(CHROMA_PATH):
                vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
                vectorstore.add_documents(chunks)
            else:
                Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_PATH
                )

            return len(chunks)
        finally:
            # Clean up temp file
            os.unlink(tmp_path)