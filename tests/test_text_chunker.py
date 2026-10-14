"""Unit tests for TextChunker module."""

import pytest
from src.pdf_processor import PageData, DocumentMetadata
from src.text_chunker import TextChunker, TextChunk


@pytest.fixture
def chunker():
    return TextChunker(target_tokens=300, min_tokens=50, max_tokens=400, overlap_tokens=60)


def test_chunk_short_page(chunker):
    """Ensure pages shorter than max_tokens form a single coherent chunk without truncation."""
    short_text = (
        "4.1 BrakePedalSensorSWC\n"
        "Responsible for reading dual potentiometer ADC signals and computing Sig_PedalTravel_pct."
    )
    page = PageData(page_number=3, text=short_text)
    chunks = chunker.chunk_page(page, source_file="EBSS_HLD.pdf", file_hash_prefix="a1b2c3d4")

    assert len(chunks) == 1
    c = chunks[0]
    assert c.chunk_id == "a1b2c3d4_p3_c0"
    assert c.source_file == "EBSS_HLD.pdf"
    assert c.page_number == 3
    assert c.chunk_index == 0
    assert "SWC" in c.text
    assert c.token_count > 0


def test_chunk_long_page_with_overlap(chunker):
    """Ensure multi-paragraph long pages are split into overlapping chunks with metadata."""
    long_paragraphs = [
        f"Paragraph {i}: The Electronic Braking and Stability System monitors wheel speeds "
        f"Sig_WheelSpeed_{i}_kmh and controls brake caliper hydraulic pressure modulation."
        for i in range(25)
    ]
    full_text = "\n\n".join(long_paragraphs)
    page = PageData(page_number=2, text=full_text)

    chunks = chunker.chunk_page(page, source_file="LongDoc.pdf", file_hash_prefix="hash1234")
    assert len(chunks) >= 2

    # Check each chunk's properties
    for idx, c in enumerate(chunks):
        assert c.source_file == "LongDoc.pdf"
        assert c.page_number == 2
        assert c.chunk_index == idx
        assert c.chunk_id == f"hash1234_p2_c{idx}"
        assert c.token_count <= 450


def test_preserve_technical_identifiers(chunker):
    """Verify that complex AUTOSAR technical identifiers are not fragmented across chunks."""
    tech_text = (
        "Interface If_DriverBrakeRequest transfers signals Sig_PedalTravel_pct and Sig_BrakeIntent_enum.\n"
        "The external CAN status frame CAN_MSG_BRAKE_STATUS has CAN ID 0x1A4 on the chassis bus.\n"
        "Component SWC_AntiLockBraking receives Sig_WheelSpeed_FL_kmh, Sig_WheelSpeed_FR_kmh, "
        "Sig_WheelSpeed_RL_kmh, and Sig_WheelSpeed_RR_kmh."
    )
    page = PageData(page_number=4, text=tech_text)
    chunks = chunker.chunk_page(page, source_file="EBSS_HLD.pdf")

    assert len(chunks) == 1
    chunk_text = chunks[0].text
    assert "If_DriverBrakeRequest" in chunk_text
    assert "Sig_PedalTravel_pct" in chunk_text
    assert "CAN_MSG_BRAKE_STATUS" in chunk_text
    assert "0x1A4" in chunk_text
    assert "SWC_AntiLockBraking" in chunk_text


def test_empty_page_returns_no_chunks(chunker):
    """Ensure completely blank pages yield 0 chunks without errors."""
    page = PageData(page_number=1, text="   \n\n  \t  ")
    chunks = chunker.chunk_page(page, source_file="Empty.pdf")
    assert chunks == []


def test_chunk_entire_document(chunker):
    """Verify processing an entire DocumentMetadata structure."""
    doc = DocumentMetadata(
        filename="TestDoc.pdf",
        file_hash="deadbeef1234",
        total_pages=3,
        total_chars=500,
        pages=[
            PageData(page_number=1, text="Page 1: Architecture Overview of AUTOSAR Classic."),
            PageData(page_number=2, text="Page 2: Software Component Specification and RTE."),
            PageData(page_number=3, text="Page 3: Signal Dictionary and Pinout Tables."),
        ]
    )

    all_chunks = chunker.chunk_document(doc)
    assert len(all_chunks) == 3
    assert all_chunks[0].page_number == 1
    assert all_chunks[1].page_number == 2
    assert all_chunks[2].page_number == 3
    assert all_chunks[0].chunk_id.startswith("deadbeef")
