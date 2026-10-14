"""AUTOSAR HLD Document Analysis Assistant RAG Package."""

from src.config import AppConfig, get_config
from src.pdf_processor import PDFProcessor, PageData, DocumentMetadata, PDFProcessingResult
from src.text_chunker import TextChunker, TextChunk
from src.vector_store import VectorStore, RetrievedChunk, FallbackOfflineEmbeddingFunction
from src.rag_pipeline import RAGPipeline, RAGResponse, Citation

__version__ = "1.0.0"

__all__ = [
    "AppConfig",
    "get_config",
    "PDFProcessor",
    "PageData",
    "DocumentMetadata",
    "PDFProcessingResult",
    "TextChunker",
    "TextChunk",
    "VectorStore",
    "RetrievedChunk",
    "FallbackOfflineEmbeddingFunction",
    "RAGPipeline",
    "RAGResponse",
    "Citation",
]

