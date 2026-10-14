"""Text chunker module optimized for technical AUTOSAR specifications."""

import re
from dataclasses import dataclass
from typing import List, Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_AVAILABLE = False
    _ENC = None

from src.pdf_processor import DocumentMetadata, PageData


@dataclass
class TextChunk:
    """Represents an indexed chunk with rich source provenance."""
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    token_count: int
    char_count: int

    def to_metadata(self) -> dict:
        """Convert chunk attributes to a ChromaDB-compatible metadata dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "page_number": int(self.page_number),
            "chunk_index": int(self.chunk_index),
            "token_count": int(self.token_count),
            "char_count": int(self.char_count),
        }


class TextChunker:
    """
    Splits document pages into overlapping semantic chunks
    while preserving AUTOSAR technical tokens and identifiers.
    """

    def __init__(
        self,
        target_tokens: int = 600,
        min_tokens: int = 150,
        max_tokens: int = 800,
        overlap_tokens: int = 100
    ):
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken if available, otherwise heuristic approximation."""
        if _TIKTOKEN_AVAILABLE and _ENC is not None:
            try:
                return len(_ENC.encode(text, disallowed_special=()))
            except Exception:
                pass
        # Fallback: roughly 4 characters per token for technical English text
        words = text.split()
        return max(len(words), len(text) // 4)

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines or list items without breaking lines."""
        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split on double newlines or section-like breaks
        raw_paragraphs = re.split(r"\n\s*\n+", text)
        cleaned = [p.strip() for p in raw_paragraphs if p.strip()]
        return cleaned if cleaned else ([text.strip()] if text.strip() else [])

    def _split_long_paragraph(self, paragraph: str, max_words: int = 250) -> List[str]:
        """Split very long paragraphs along sentence boundaries while preserving identifiers."""
        # Match sentence boundaries that don't split technical terms like 'Fig. 1' or 'e.g.'
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9_])", paragraph)
        sub_paras = []
        current = []
        current_len = 0

        for sent in sentences:
            sent_words = len(sent.split())
            if current_len + sent_words > max_words and current:
                sub_paras.append(" ".join(current))
                current = [sent]
                current_len = sent_words
            else:
                current.append(sent)
                current_len += sent_words

        if current:
            sub_paras.append(" ".join(current))

        return sub_paras

    def chunk_page(
        self,
        page: PageData,
        source_file: str,
        file_hash_prefix: str = "doc"
    ) -> List[TextChunk]:
        """
        Chunk a single page into overlapping segments.
        Retains exact page provenance for every chunk.
        """
        text = page.text.strip()
        if not text:
            return []

        total_tokens = self.count_tokens(text)

        # If page fits within max_tokens, treat as single clean chunk
        if total_tokens <= self.max_tokens:
            chunk_id = f"{file_hash_prefix}_p{page.page_number}_c0"
            return [
                TextChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source_file=source_file,
                    page_number=page.page_number,
                    chunk_index=0,
                    token_count=total_tokens,
                    char_count=len(text)
                )
            ]

        # Break into structured units
        raw_paragraphs = self._split_into_paragraphs(text)
        units: List[str] = []
        for p in raw_paragraphs:
            if self.count_tokens(p) > self.max_tokens:
                units.extend(self._split_long_paragraph(p))
            else:
                units.append(p)

        chunks: List[TextChunk] = []
        current_units: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)
            if current_tokens + unit_tokens > self.max_tokens and current_units:
                # Emit current chunk
                chunk_body = "\n\n".join(current_units)
                c_id = f"{file_hash_prefix}_p{page.page_number}_c{chunk_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=c_id,
                        text=chunk_body,
                        source_file=source_file,
                        page_number=page.page_number,
                        chunk_index=chunk_idx,
                        token_count=self.count_tokens(chunk_body),
                        char_count=len(chunk_body)
                    )
                )
                chunk_idx += 1

                # Calculate overlap units from tail
                overlap_units = []
                overlap_toks = 0
                for u in reversed(current_units):
                    u_tok = self.count_tokens(u)
                    if overlap_toks + u_tok <= self.overlap_tokens:
                        overlap_units.insert(0, u)
                        overlap_toks += u_tok
                    else:
                        break

                current_units = overlap_units + [unit]
                current_tokens = self.count_tokens("\n\n".join(current_units))
            else:
                current_units.append(unit)
                current_tokens += unit_tokens

        # Flush remaining units
        if current_units:
            chunk_body = "\n\n".join(current_units)
            c_id = f"{file_hash_prefix}_p{page.page_number}_c{chunk_idx}"
            chunks.append(
                TextChunk(
                    chunk_id=c_id,
                    text=chunk_body,
                    source_file=source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_idx,
                    token_count=self.count_tokens(chunk_body),
                    char_count=len(chunk_body)
                )
            )

        return chunks

    def chunk_document(self, document: DocumentMetadata) -> List[TextChunk]:
        """Chunk all pages of a processed document and assign persistent identifiers."""
        all_chunks: List[TextChunk] = []
        hash_prefix = document.file_hash[:8] if document.file_hash else "doc"

        for page in document.pages:
            page_chunks = self.chunk_page(
                page=page,
                source_file=document.filename,
                file_hash_prefix=hash_prefix
            )
            all_chunks.extend(page_chunks)

        return all_chunks
