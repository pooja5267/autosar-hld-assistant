"""Automated tests for Streamlit application rendering and controls."""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def sample_pdf_ensured():
    """Ensure benchmark PDF exists before Streamlit tests."""
    p = Path(__file__).resolve().parent.parent / "data" / "sample_hld" / "Sample_AUTOSAR_BrakeSystem_HLD.pdf"
    if not p.exists():
        from scripts.generate_sample_hld import create_sample_hld_pdf
        create_sample_hld_pdf(p)
    return p


def test_streamlit_app_startup(sample_pdf_ensured):
    """Verify that app.py loads, initializes all components, and renders without exceptions."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify no unhandled exceptions occurred
    assert len(at.exception) == 0, f"Streamlit app raised exceptions: {[e.value for e in at.exception]}"

    # Verify key UI components exist
    assert len(at.sidebar.selectbox) >= 2  # Provider and Model select boxes
    assert len(at.tabs) == 3  # Q&A, Document Ingestion, Specification Inspector
    assert len(at.text_area) >= 1  # Query input text area


def test_streamlit_app_query_flow(sample_pdf_ensured):
    """Verify that entering a question and triggering analysis operates cleanly."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert len(at.exception) == 0

    # Populate query input
    at.text_area[0].input("What are the responsibilities of AntiLockBrakingSWC?").run()
    assert len(at.exception) == 0
