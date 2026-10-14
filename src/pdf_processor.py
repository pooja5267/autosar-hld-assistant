"""PDF Processing module using PyMuPDF for AUTOSAR HLD documents."""

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, BinaryIO

import fitz  # PyMuPDF


@dataclass
class PageData:
    """Represents text extracted from a single PDF page."""
    page_number: int  # 1-indexed
    text: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class DocumentMetadata:
    """Metadata and content for a processed PDF document."""
    filename: str
    file_hash: str
    total_pages: int
    total_chars: int
    is_scanned: bool = False
    warning_message: Optional[str] = None
    pages: List[PageData] = field(default_factory=list)


@dataclass
class PDFProcessingResult:
    """Outcome of processing an individual PDF file."""
    filename: str
    success: bool
    metadata: Optional[DocumentMetadata] = None
    error_message: Optional[str] = None


class PDFProcessor:
    """Validates and extracts page-level text and metadata from PDF files."""

    def __init__(self, max_file_size_mb: int = 20):
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_file_size_mb = max_file_size_mb

    @staticmethod
    def compute_hash(data: bytes) -> str:
        """Compute SHA-256 hash of file content for deduplication."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize text to prevent malformed characters while preserving technical identifiers."""
        if not text:
            return ""
        # Remove null bytes and non-printable control characters except standard whitespace
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        # Normalize multiple spaces on single lines without flattening paragraph breaks
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def validate_file(self, filename: str, file_bytes: bytes) -> Optional[str]:
        """Validate filename extension, file size, and PDF magic bytes."""
        if not filename.lower().endswith(".pdf"):
            return f"Invalid file type '{filename}'. Only .pdf files are accepted."

        size_bytes = len(file_bytes)
        if size_bytes == 0:
            return f"File '{filename}' is completely empty (0 bytes)."

        if size_bytes > self.max_file_size_bytes:
            return (
                f"File '{filename}' exceeds maximum allowed size "
                f"({size_bytes / (1024 * 1024):.1f} MB > {self.max_file_size_mb} MB)."
            )

        # Check PDF magic bytes (%PDF-)
        if not file_bytes.startswith(b"%PDF-"):
            return f"File '{filename}' does not appear to be a valid PDF (missing %PDF- header)."

        return None

    def process_pdf(
        self,
        file_input: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None
    ) -> PDFProcessingResult:
        """
        Process a single PDF file from path, bytes, or file-like stream.

        Returns:
            PDFProcessingResult with DocumentMetadata or descriptive error.
        """
        # Resolve bytes and filename
        file_bytes: bytes = b""
        actual_filename: str = "document.pdf"

        try:
            if isinstance(file_input, (str, Path)):
                p = Path(file_input)
                actual_filename = filename or p.name
                if not p.exists():
                    return PDFProcessingResult(
                        filename=actual_filename,
                        success=False,
                        error_message=f"File not found: {p}"
                    )
                file_bytes = p.read_bytes()
            elif isinstance(file_input, bytes):
                actual_filename = filename or "uploaded_document.pdf"
                file_bytes = file_input
            elif hasattr(file_input, "read"):
                actual_filename = filename or getattr(file_input, "name", "uploaded_document.pdf")
                if hasattr(file_input, "getvalue"):
                    file_bytes = file_input.getvalue()
                else:
                    if hasattr(file_input, "seek"):
                        file_input.seek(0)
                    file_bytes = file_input.read()
            else:
                return PDFProcessingResult(
                    filename=actual_filename,
                    success=False,
                    error_message=f"Unsupported file input type: {type(file_input)}"
                )
        except Exception as e:
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message=f"Failed to read file input: {str(e)}"
            )

        # Validate file
        validation_error = self.validate_file(actual_filename, file_bytes)
        if validation_error:
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message=validation_error
            )

        file_hash = self.compute_hash(file_bytes)

        # Open with PyMuPDF
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message=f"Corrupted or damaged PDF file: {str(e)}"
            )
        except Exception as e:
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message=f"Could not open PDF: {str(e)}"
            )

        # Check encryption
        if doc.is_encrypted:
            doc.close()
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message="Document is password-protected or encrypted. Please decrypt it before uploading."
            )

        total_pages = doc.page_count
        if total_pages == 0:
            doc.close()
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message="Document contains zero pages."
            )

        pages_data: List[PageData] = []
        total_chars = 0

        try:
            for page_idx in range(total_pages):
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text")
                cleaned_text = self.sanitize_text(raw_text)
                p_data = PageData(
                    page_number=page_idx + 1,
                    text=cleaned_text
                )
                pages_data.append(p_data)
                total_chars += len(cleaned_text)
        except Exception as e:
            doc.close()
            return PDFProcessingResult(
                filename=actual_filename,
                success=False,
                error_message=f"Error extracting text from page: {str(e)}"
            )
        finally:
            doc.close()

        # Check for scanned / non-text PDF
        is_scanned = False
        warning = None
        if total_chars < 30 and total_pages >= 1:
            is_scanned = True
            warning = (
                f"Document '{actual_filename}' contains virtually no extractable text "
                f"({total_chars} characters across {total_pages} pages). "
                "The file appears to be a scanned image or raster PDF. "
                "Optical Character Recognition (OCR) is required to index this document."
            )

        metadata = DocumentMetadata(
            filename=actual_filename,
            file_hash=file_hash,
            total_pages=total_pages,
            total_chars=total_chars,
            is_scanned=is_scanned,
            warning_message=warning,
            pages=pages_data
        )

        return PDFProcessingResult(
            filename=actual_filename,
            success=True,
            metadata=metadata,
            error_message=warning if is_scanned else None
        )

    def process_multiple_pdfs(
        self,
        files: List[Union[str, Path, bytes, BinaryIO]],
        filenames: Optional[List[str]] = None
    ) -> List[PDFProcessingResult]:
        """Process a list of PDF files and return results for each."""
        results = []
        for idx, f in enumerate(files):
            fname = filenames[idx] if filenames and idx < len(filenames) else None
            res = self.process_pdf(f, filename=fname)
            results.append(res)
        return results
