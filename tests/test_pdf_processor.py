"""Unit tests for PDFProcessor module."""

import io
from pathlib import Path
import fitz
import pytest

from src.pdf_processor import PDFProcessor, PageData, DocumentMetadata


@pytest.fixture
def processor():
    return PDFProcessor(max_file_size_mb=20)


@pytest.fixture
def sample_pdf_path():
    p = Path(__file__).resolve().parent.parent / "data" / "sample_hld" / "Sample_AUTOSAR_BrakeSystem_HLD.pdf"
    if not p.exists():
        # Ensure sample exists
        from scripts.generate_sample_hld import create_sample_hld_pdf
        create_sample_hld_pdf(p)
    return p


def test_process_valid_sample_hld(processor, sample_pdf_path):
    """Verify end-to-end extraction from valid multi-page AUTOSAR HLD document."""
    result = processor.process_pdf(sample_pdf_path)
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.filename == "Sample_AUTOSAR_BrakeSystem_HLD.pdf"
    assert result.metadata.total_pages >= 5
    assert result.metadata.total_chars > 2000
    assert not result.metadata.is_scanned
    assert len(result.metadata.pages) == result.metadata.total_pages

    # Check page numbering is 1-indexed
    assert result.metadata.pages[0].page_number == 1
    assert "AUTOSAR" in result.metadata.pages[0].text
    assert result.metadata.file_hash is not None


def test_empty_file_rejection(processor):
    """Ensure 0-byte files are caught and return a descriptive validation error."""
    result = processor.process_pdf(b"", filename="empty.pdf")
    assert result.success is False
    assert "empty (0 bytes)" in result.error_message


def test_invalid_extension_rejection(processor):
    """Ensure non-PDF files are rejected."""
    result = processor.process_pdf(b"not a pdf", filename="spec.docx")
    assert result.success is False
    assert "Invalid file type" in result.error_message


def test_corrupted_pdf_handling(processor):
    """Ensure corrupted PDF header or structure is handled gracefully without crashing."""
    corrupted_bytes = b"%PDF-1.4\n%Malformed binary trash 9999999\x00\xFF"
    result = processor.process_pdf(corrupted_bytes, filename="damaged.pdf")
    assert result.success is False
    assert result.error_message is not None


def test_scanned_pdf_detection(processor):
    """Verify that PDFs with pages but no extractable text trigger OCR detection warning."""
    # Create in-memory blank PDF with fitz
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    pdf_bytes = doc.tobytes()
    doc.close()

    result = processor.process_pdf(pdf_bytes, filename="scanned_diagrams.pdf")
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.is_scanned is True
    assert "scanned image" in result.metadata.warning_message
    assert "OCR" in result.metadata.warning_message


def test_file_size_limit_rejection():
    """Verify that oversized files are rejected with a clear message."""
    tiny_limit_processor = PDFProcessor(max_file_size_mb=1)
    huge_bytes = b"%PDF-" + b"0" * (2 * 1024 * 1024)  # 2MB
    result = tiny_limit_processor.process_pdf(huge_bytes, filename="huge.pdf")
    assert result.success is False
    assert "exceeds maximum allowed size" in result.error_message


def test_text_sanitization(processor):
    """Verify removal of dangerous null bytes while preserving technical identifiers."""
    raw = "SWC_BrakePedalSensor\x00\x08   valid text   with  spaces \n\nSig_PedalTravel_pct"
    cleaned = processor.sanitize_text(raw)
    assert "\x00" not in cleaned
    assert "SWC_BrakePedalSensor" in cleaned
    assert "Sig_PedalTravel_pct" in cleaned


def test_process_pdf_from_bytes_io(processor, sample_pdf_path):
    """Verify processing PDF from an in-memory BytesIO stream (simulating Streamlit UploadedFile)."""
    pdf_bytes = sample_pdf_path.read_bytes()
    stream = io.BytesIO(pdf_bytes)
    result = processor.process_pdf(stream, filename="streamed_spec.pdf")
    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.filename == "streamed_spec.pdf"
    assert result.metadata.total_pages >= 5


def test_process_multiple_pdfs(processor, sample_pdf_path):
    """Verify batch processing of multiple PDF inputs."""
    pdf_bytes = sample_pdf_path.read_bytes()
    results = processor.process_multiple_pdfs(
        files=[pdf_bytes, b"not a pdf"],
        filenames=["valid_doc.pdf", "invalid_doc.pdf"]
    )
    assert len(results) == 2
    assert results[0].success is True
    assert results[0].metadata.filename == "valid_doc.pdf"
    assert results[1].success is False


def test_nonexistent_file_path(processor):
    """Verify graceful handling when given a path to a non-existent file."""
    result = processor.process_pdf(Path("non_existent_autosar_file.pdf"))
    assert result.success is False
    assert "File not found" in result.error_message
