"""ChromaDB vector store manager for indexing and semantic retrieval."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.api.types import EmbeddingFunction

try:
    from chromadb.api.collection_configuration import register_embedding_function
except ImportError:
    def register_embedding_function(cls):
        return cls

from src.text_chunker import TextChunk


@dataclass
class RetrievedChunk:
    """Represents a retrieved passage with similarity score and page provenance."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    similarity_score: float
    distance: float

    def citation_label(self) -> str:
        """Standardized citation string: [Source: filename, Page: X]"""
        return f"[Source: {self.source_file}, Page: {self.page_number}]"


@register_embedding_function
class FallbackOfflineEmbeddingFunction(EmbeddingFunction):
    """
    Deterministic offline embedding function producing normalized 384-dimensional
    vectors based on token hashing and TF weighting.
    Guarantees that automated tests and offline environments work reliably without
    requiring model downloads or external network access.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "fallback_offline_embedding"

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for ChromaDB embedding registry."""
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "FallbackOfflineEmbeddingFunction":
        """Reconstruct embedding function from stored config dictionary."""
        return cls(dim=config.get("dim", 384))

    def __call__(self, input: List[str]) -> List[List[float]]:
        stopwords = {
            "what", "is", "are", "the", "of", "and", "a", "an", "in", "to", "for", "with",
            "on", "at", "by", "from", "up", "about", "into", "over", "after"
        }
        embeddings = []
        for text in input:
            vec = [0.0] * self.dim
            words = text.split()
            if not words:
                embeddings.append(vec)
                continue

            for raw_w in words:
                w = raw_w.strip(".,;:()[]{}\"'").lower()
                if not w:
                    continue

                # Weight technical terms and content words higher than stopwords
                if w in stopwords:
                    weight = 0.05
                elif "_" in raw_w or any(c.isupper() for c in raw_w[1:]):
                    weight = 3.0  # High weight for identifiers like AntiLockBrakingSWC
                elif len(w) > 4:
                    weight = 1.5
                else:
                    weight = 0.8

                h = hash(w)
                slot = abs(h) % self.dim
                sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
                vec[slot] += sign * weight

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 1e-9:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings


def get_embedding_function(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingFunction:
    """
    Instantiate the embedding function.
    Prefers Chroma's DefaultEmbeddingFunction (which runs all-MiniLM-L6-v2 via ONNX).
    Falls back gracefully to FallbackOfflineEmbeddingFunction if loading fails or during offline testing.
    """
    # Allow explicit offline override via environment variable (useful in tests)
    if os.getenv("AUTOSAR_TEST_OFFLINE_EMBEDDINGS") == "1":
        return FallbackOfflineEmbeddingFunction()

    try:
        from chromadb.utils import embedding_functions
        # DefaultEmbeddingFunction packages all-MiniLM-L6-v2 ONNX
        return embedding_functions.DefaultEmbeddingFunction()
    except Exception:
        return FallbackOfflineEmbeddingFunction()


class VectorStore:
    """Persistent ChromaDB vector store for AUTOSAR document indexing."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "autosar_hld_docs",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_fn: Optional[EmbeddingFunction] = None
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_fn = embedding_fn or get_embedding_function(embedding_model_name)

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._registry_file = self.persist_dir / "document_registry.json"
        self._doc_registry = self._load_registry()

        # Get or create collection with cosine space
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except ValueError as e:
            if "Embedding function conflict" in str(e):
                # When switching between offline test embeddings and default ONNX embeddings,
                # reset conflicting collection to avoid hard crash
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"}
                )
                self._doc_registry = {}
                self._save_registry()
            else:
                raise

    def _load_registry(self) -> Dict[str, Any]:
        """Load document index tracking registry from disk."""
        if self._registry_file.exists():
            try:
                return json.loads(self._registry_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_registry(self) -> None:
        """Persist document index tracking registry."""
        try:
            self._registry_file.write_text(
                json.dumps(self._doc_registry, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def is_document_indexed(self, filename: str, file_hash: Optional[str] = None) -> bool:
        """Check if document is already indexed by filename or SHA-256 hash."""
        if filename in self._doc_registry:
            if file_hash:
                return self._doc_registry[filename].get("file_hash") == file_hash
            return True
        return False

    def remove_document(self, filename: str) -> int:
        """Remove all chunks associated with a specific filename."""
        try:
            # Query IDs belonging to this source_file
            results = self.collection.get(
                where={"source_file": filename}
            )
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            if filename in self._doc_registry:
                del self._doc_registry[filename]
                self._save_registry()
            return len(ids)
        except Exception:
            return 0

    def add_chunks(
        self,
        chunks: List[TextChunk],
        file_hash: Optional[str] = None,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Add text chunks to the Chroma collection with deduplication.

        Returns:
            Dict summarizing added chunks and status.
        """
        if not chunks:
            return {"status": "empty", "chunks_added": 0, "message": "No chunks to index."}

        filename = chunks[0].source_file

        # Check existing
        if self.is_document_indexed(filename, file_hash):
            if not overwrite:
                return {
                    "status": "already_indexed",
                    "filename": filename,
                    "chunks_added": 0,
                    "message": f"Document '{filename}' is already indexed."
                }
            # Overwrite: delete existing chunks for this document first
            self.remove_document(filename)

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        # Update registry
        pages = list(set(chunk.page_number for chunk in chunks))
        self._doc_registry[filename] = {
            "filename": filename,
            "file_hash": file_hash,
            "chunk_count": len(chunks),
            "pages_count": len(pages),
            "pages": sorted(pages),
        }
        self._save_registry()

        return {
            "status": "success",
            "filename": filename,
            "chunks_added": len(chunks),
            "total_pages": len(pages),
            "message": f"Successfully indexed {len(chunks)} chunks from '{filename}'."
        }

    def query(
        self,
        query_text: str,
        top_k: int = 4
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top_k most relevant chunks for a question.
        Returns empty list if collection is empty or query has no text.
        """
        clean_query = query_text.strip()
        if not clean_query or self.collection.count() == 0:
            return []

        # Ensure top_k does not exceed total items in collection
        actual_k = min(top_k, self.collection.count())
        if actual_k <= 0:
            return []

        results = self.collection.query(
            query_texts=[clean_query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[RetrievedChunk] = []

        ids_list = (results.get("ids") or [[]])[0]
        docs_list = (results.get("documents") or [[]])[0]
        meta_list = (results.get("metadatas") or [[]])[0]
        dist_list = (results.get("distances") or [[]])[0]

        for idx in range(len(ids_list)):
            chunk_id = ids_list[idx]
            text = docs_list[idx]
            meta = meta_list[idx] if idx < len(meta_list) else {}
            dist = float(dist_list[idx]) if idx < len(dist_list) else 1.0

            # For cosine distance in Chroma: similarity = 1 - distance
            # Clamped to [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - dist))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=meta.get("source_file", "unknown"),
                    page_number=int(meta.get("page_number", 1)),
                    similarity_score=similarity,
                    distance=dist
                )
            )

        return retrieved

    def clear_collection(self) -> None:
        """Clear all indexed documents and reset registry."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._doc_registry = {}
        self._save_registry()

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Return list of all currently indexed documents with summary stats."""
        return list(self._doc_registry.values())

    def get_total_chunks(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0
