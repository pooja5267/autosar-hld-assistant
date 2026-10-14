"""
AUTOSAR HLD Document Analysis Assistant - Streamlit Application

A specialized, evidence-grounded RAG assistant designed for automotive software engineers,
system architects, and safety analysts reviewing AUTOSAR High-Level Design (HLD) specifications.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import AppConfig, get_config
from src.pdf_processor import PDFProcessor, PDFProcessingResult
from src.text_chunker import TextChunker, TextChunk
from src.vector_store import VectorStore, RetrievedChunk
from src.rag_pipeline import RAGPipeline, RAGResponse, Citation
from scripts.generate_sample_hld import create_sample_hld_pdf, OUTPUT_FILE as SAMPLE_PDF_PATH


# =============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLING
# =============================================================================

st.set_page_config(
    page_title="AUTOSAR HLD Analysis Assistant",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern Automotive HUD Theme */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

code, pre, .mono-badge {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Main container styling */
.main {
    background: linear-gradient(135deg, #0b1120 0%, #0f172a 50%, #020617 100%);
    color: #e2e8f0;
}

/* Glassmorphic Cards */
.hud-card {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    margin-bottom: 1rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.hud-card:hover {
    border-color: rgba(56, 189, 248, 0.45);
    transform: translateY(-2px);
}

/* Header Styling */
.hud-header {
    background: linear-gradient(90deg, rgba(14, 165, 233, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}

.hud-title {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}

.hud-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    font-weight: 400;
}

/* Metric Pill Badges */
.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
}
.status-pill-blue {
    background: rgba(14, 165, 233, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
}
.status-pill-emerald {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.4);
}
.status-pill-amber {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(251, 191, 36, 0.4);
}
.status-pill-rose {
    background: rgba(244, 63, 94, 0.2);
    color: #fb7185;
    border: 1px solid rgba(251, 113, 133, 0.4);
}

/* Citation Badges */
.citation-pill {
    background: rgba(30, 58, 138, 0.5);
    border: 1px solid rgba(96, 165, 250, 0.4);
    color: #93c5fd;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    font-size: 0.8rem;
    display: inline-block;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Chunk Evidence Card */
.evidence-card {
    background: #0f172a;
    border-left: 3px solid #38bdf8;
    padding: 0.85rem 1rem;
    margin: 0.6rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.88rem;
    line-height: 1.45;
}

/* Quick prompt button styling */
div[data-testid="stButton"] button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# RESOURCE INITIALIZATION (CACHED)
# =============================================================================

@st.cache_resource(show_spinner=False)
def init_system_resources():
    """Initialize persistent VectorStore, PDFProcessor, and TextChunker."""
    config = get_config()
    vector_store = VectorStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.collection_name,
        embedding_model_name=config.embedding_model_name,
    )
    pdf_processor = PDFProcessor(max_file_size_mb=config.max_file_size_mb)
    text_chunker = TextChunker()
    return config, vector_store, pdf_processor, text_chunker


app_config, vector_store, pdf_processor, text_chunker = init_system_resources()

# Initialize session state for user queries and chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = ""


# =============================================================================
# SIDEBAR CONTROLS & MODEL CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")

    provider_options = ["offline", "groq", "openai", "gemini", "anthropic", "ollama"]
    current_provider = app_config.llm_provider if app_config.llm_provider in provider_options else "offline"
    selected_provider = st.selectbox(
        "LLM Provider",
        options=provider_options,
        index=provider_options.index(current_provider),
        help="Select offline extractive mode or connect a remote foundation model."
    )

    # Provider specific default models
    model_defaults = {
        "offline": ["extractive-synthesis"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "ollama": ["llama3", "mistral", "qwen2.5"],
    }

    available_models = model_defaults.get(selected_provider, ["default"])
    selected_model = st.selectbox(
        "Model Identifier",
        options=available_models,
        index=0
    )

    # Dynamic API Key Input
    custom_api_key = None
    if selected_provider not in ["offline", "ollama"]:
        env_has_key = bool(app_config.get_active_api_key())
        masked_current = app_config.get_masked_api_key()
        api_key_help = f"Currently loaded: {masked_current}" if env_has_key else "No key found in .env"

        custom_api_key = st.text_input(
            f"{selected_provider.upper()} API Key",
            type="password",
            placeholder=masked_current if env_has_key else "Enter API Key...",
            help=api_key_help
        )

    custom_base_url = None
    if selected_provider == "ollama":
        custom_base_url = st.text_input(
            "Ollama Base URL",
            value=app_config.llm_base_url or "http://localhost:11434/v1",
            help="OpenAI-compatible Ollama endpoint."
        )

    st.markdown("---")
    st.markdown("### 🎯 Retrieval Parameters")
    retrieval_k = st.slider("Retrieved Chunks (Top-K)", min_value=1, max_value=8, value=app_config.retrieval_top_k)
    similarity_thresh = st.slider("Similarity Threshold", min_value=0.0, max_value=0.8, value=app_config.similarity_threshold, step=0.05)

    st.markdown("---")
    st.markdown("### 🗄️ Vector Store Maintenance")
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Chunks", vector_store.get_total_chunks())
    with col_stat2:
        st.metric("Documents", len(vector_store.get_indexed_documents()))

    if st.button("🗑️ Clear Vector Database", use_container_width=True):
        vector_store.clear_collection()
        st.session_state.chat_history = []
        st.success("Vector database successfully cleared.")
        st.rerun()

    st.markdown("---")
    st.caption("AUTOSAR Classic R4.4 & Adaptive HLD Assistant • Capstone Benchmark")


# Build dynamic runtime config based on sidebar choices
runtime_config = AppConfig(
    chroma_persist_dir=app_config.chroma_persist_dir,
    collection_name=app_config.collection_name,
    embedding_model_name=app_config.embedding_model_name,
    retrieval_top_k=retrieval_k,
    similarity_threshold=similarity_thresh,
    llm_provider=selected_provider,
    llm_model=selected_model,
    openai_api_key=custom_api_key if selected_provider == "openai" and custom_api_key else app_config.openai_api_key,
    groq_api_key=custom_api_key if selected_provider == "groq" and custom_api_key else app_config.groq_api_key,
    gemini_api_key=custom_api_key if selected_provider == "gemini" and custom_api_key else app_config.gemini_api_key,
    anthropic_api_key=custom_api_key if selected_provider == "anthropic" and custom_api_key else app_config.anthropic_api_key,
    llm_base_url=custom_base_url or app_config.llm_base_url,
)

rag_pipeline = RAGPipeline(vector_store=vector_store, config=runtime_config)


# =============================================================================
# HEADER & SYSTEM STATUS HUD
# =============================================================================

st.markdown("""
<div class="hud-header">
    <div class="hud-title">🏎️ AUTOSAR HLD Document Analysis Assistant</div>
    <div class="hud-subtitle">
        Architectural reasoning and verification for automotive High-Level Design (HLD) specifications.
        Answers strictly grounded in document evidence with page-level citations and untrusted data isolation.
    </div>
</div>
""", unsafe_allow_html=True)

# Status badges
mode_pill = "status-pill-emerald" if selected_provider == "offline" else "status-pill-blue"
mode_label = "Mode: Offline Extractive (No API Key)" if selected_provider == "offline" else f"Mode: Remote LLM ({selected_provider.upper()})"

col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    st.markdown(f'<span class="status-pill {mode_pill}">{mode_label}</span>', unsafe_allow_html=True)
with col_b2:
    st.markdown(f'<span class="status-pill status-pill-blue">Model: {selected_model}</span>', unsafe_allow_html=True)
with col_b3:
    doc_count = len(vector_store.get_indexed_documents())
    st.markdown(f'<span class="status-pill status-pill-amber">Indexed Docs: {doc_count}</span>', unsafe_allow_html=True)
with col_b4:
    st.markdown('<span class="status-pill status-pill-emerald">Security: Untrusted Evidence Isolation Active</span>', unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# =============================================================================
# MAIN INTERFACE TABS
# =============================================================================

tab_qa, tab_docs, tab_specs = st.tabs([
    "💬 Architectural Q&A",
    "📂 Document Ingestion & Repository",
    "📊 Specification Inspector"
])


# -----------------------------------------------------------------------------
# TAB 1: ARCHITECTURAL Q&A
# -----------------------------------------------------------------------------
with tab_qa:
    st.markdown("### 🔍 Query AUTOSAR Technical Specification")

    # Sample architectural prompt pills
    st.markdown("**Quick Preset Queries (Click to populate):**")
    preset_cols = st.columns(4)

    prompt_presets = [
        ("ABS Runnable & Timing", "What are the responsibilities and timing cycle of AntiLockBrakingSWC?"),
        ("CAN Status Mapping", "Which CAN message ID and signals are mapped for brake status?"),
        ("Sensor Safe States", "What fail-safe actions occur if pedal stroke sensors disagree?"),
        ("Out-of-Scope Scope", "What long-range radar or parking brake features are covered in this document?"),
    ]

    for idx, (label, text) in enumerate(prompt_presets):
        with preset_cols[idx]:
            if st.button(f"📌 {label}", key=f"preset_{idx}", use_container_width=True):
                st.session_state.selected_prompt = text

    user_query = st.text_area(
        "Enter your architectural, safety, or interface question:",
        value=st.session_state.selected_prompt,
        height=85,
        placeholder="e.g. What are the cyclic runnable periods and fault thresholds for BrakePedalSensorSWC?",
        key="query_input"
    )

    col_btn_ask, col_btn_clear = st.columns([1, 5])
    with col_btn_ask:
        ask_submitted = st.button("🚀 Analyze Evidence", type="primary", use_container_width=True)
    with col_btn_clear:
        if st.button("🧹 Clear Conversation", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.selected_prompt = ""
            st.rerun()

    if ask_submitted and user_query.strip():
        if vector_store.get_total_chunks() == 0:
            st.warning("⚠️ No documents are currently indexed in the vector store. Please upload an AUTOSAR HLD PDF or load the synthetic benchmark in the 'Document Ingestion' tab.")
        else:
            with st.spinner("Retrieving evidence chunks and synthesizing verified response..."):
                response: RAGResponse = rag_pipeline.answer(user_query.strip())
                st.session_state.chat_history.insert(0, response)

    # Render Responses
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📋 Analysis Results")

        for idx, item in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"#### ❓ Question: {item.question}")

                # Abstention or verified answer alert
                if item.is_abstention:
                    st.warning(item.answer)
                else:
                    st.markdown(item.answer)

                    # Citations section
                    if item.citations:
                        st.markdown("**Validated Provenance Citations:**")
                        citation_html = " ".join([
                            f'<span class="citation-pill">📄 {c.source_file} (Page {c.page_number})</span>'
                            for c in item.citations
                        ])
                        st.markdown(citation_html, unsafe_allow_html=True)

                # Expandable Evidence Inspector
                if item.retrieved_chunks:
                    with st.expander(f"🔎 Inspect Retrieved Evidence Chunks ({len(item.retrieved_chunks)} items)", expanded=False):
                        for c_idx, chunk in enumerate(item.retrieved_chunks, 1):
                            st.markdown(
                                f"""
                                <div class="evidence-card">
                                    <b>Chunk #{c_idx}</b> | <b>Source:</b> {chunk.source_file} | <b>Page:</b> {chunk.page_number} | <b>Cosine Similarity:</b> {chunk.similarity_score:.3f}<br/>
                                    <pre style="margin-top: 6px; white-space: pre-wrap; font-size: 0.8rem; background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px;">{chunk.text}</pre>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.markdown("---")


# -----------------------------------------------------------------------------
# TAB 2: DOCUMENT INGESTION & REPOSITORY
# -----------------------------------------------------------------------------
with tab_docs:
    st.markdown("### 📥 Document Repository & Ingestion")

    col_ingest_left, col_ingest_right = st.columns([1, 1])

    with col_ingest_left:
        st.markdown("""
        <div class="hud-card">
            <h4>⚡ Quick Benchmark: Electronic Braking & Stability System (EBSS)</h4>
            <p style="font-size: 0.9rem; color: #94a3b8;">
                Load the synthetic 5-page AUTOSAR HLD specification benchmark.
                Features 4 Application SWCs, signal dictionaries, 10ms CAN mappings,
                and ASIL-D safety degradation logic.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📥 Load Synthetic Sample HLD (EBSS)", type="primary", use_container_width=True):
            with st.spinner("Generating and indexing synthetic AUTOSAR HLD benchmark document..."):
                if not SAMPLE_PDF_PATH.exists():
                    create_sample_hld_pdf(SAMPLE_PDF_PATH)

                res: PDFProcessingResult = pdf_processor.process_pdf(SAMPLE_PDF_PATH)
                if res.success and res.metadata:
                    chunks = text_chunker.chunk_document(res.metadata)
                    index_res = vector_store.add_chunks(chunks, file_hash=res.metadata.file_hash, overwrite=True)
                    st.success(f"Successfully indexed '{res.metadata.filename}': {len(chunks)} chunks across {res.metadata.total_pages} pages.")
                    st.rerun()
                else:
                    st.error(f"Failed to process sample PDF: {res.error_message}")

    with col_ingest_right:
        st.markdown("""
        <div class="hud-card">
            <h4>📤 Upload Custom AUTOSAR Specifications</h4>
            <p style="font-size: 0.9rem; color: #94a3b8;">
                Upload OEM or Tier-1 AUTOSAR Classic or Adaptive High-Level Design PDF documents.
                Max file size: 20 MB. Files are sanitized and indexed with page-level tracking.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload AUTOSAR HLD PDF Documents",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files."
        )

        if uploaded_files:
            if st.button("⚡ Index Uploaded Documents", use_container_width=True):
                with st.spinner(f"Processing and indexing {len(uploaded_files)} document(s)..."):
                    total_chunks_added = 0
                    for uploaded in uploaded_files:
                        res = pdf_processor.process_pdf(uploaded, filename=uploaded.name)
                        if res.success and res.metadata:
                            if res.metadata.is_scanned:
                                st.warning(f"⚠️ '{uploaded.name}' appears to be a scanned image PDF. Extracted text may be incomplete.")
                            chunks = text_chunker.chunk_document(res.metadata)
                            index_res = vector_store.add_chunks(chunks, file_hash=res.metadata.file_hash, overwrite=True)
                            total_chunks_added += len(chunks)
                            st.success(f"Indexed '{uploaded.name}': {len(chunks)} chunks, {res.metadata.total_pages} pages.")
                        else:
                            st.error(f"Error processing '{uploaded.name}': {res.error_message}")
                    st.rerun()

    st.markdown("---")
    st.markdown("### 📑 Currently Indexed Documents")

    indexed_docs = vector_store.get_indexed_documents()
    if indexed_docs:
        doc_table = []
        for d in indexed_docs:
            doc_table.append({
                "Document Filename": d["filename"],
                "Total Chunks": d["chunk_count"],
                "Pages Indexed": d["pages_count"],
                "SHA-256 Hash Prefix": (d.get("file_hash") or "N/A")[:12],
            })
        st.dataframe(doc_table, use_container_width=True)

        col_del_select, col_del_btn = st.columns([3, 1])
        with col_del_select:
            selected_del_doc = st.selectbox(
                "Select Document to Remove",
                options=[d["filename"] for d in indexed_docs],
                key="del_doc_select"
            )
        with col_del_btn:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("🗑️ Remove Document", use_container_width=True):
                removed = vector_store.remove_document(selected_del_doc)
                st.info(f"Removed {removed} chunks belonging to '{selected_del_doc}'.")
                st.rerun()
    else:
        st.info("No documents are currently indexed in ChromaDB.")


# -----------------------------------------------------------------------------
# TAB 3: SPECIFICATION INSPECTOR
# -----------------------------------------------------------------------------
with tab_specs:
    st.markdown("### 📊 Architecture & Signal Taxonomy Reference")

    st.markdown("""
    This reference outlines standard AUTOSAR Classic Layering and chassis control constructs
    evaluated by the benchmarking suite.
    """)

    col_spec1, col_spec2 = st.columns(2)

    with col_spec1:
        st.markdown("#### 🏛️ AUTOSAR Layer Decomposition")
        st.markdown("""
        | Layer | Scope / Responsibilities | Examples in Benchmark |
        |---|---|---|
        | **Application Layer** | Core algorithms, slip ratio, vehicle dynamics | `SWC_BrakePedalSensor`, `SWC_AntiLockBraking`, `SWC_ElectronicStabilityControl` |
        | **RTE (Runtime Env)** | Decouples SWCs from BSW; manages runnable scheduling | `Rte_EBSS_Generated_v2.1` |
        | **BSW Services** | Diagnostic Event Manager, Function Inhibition | `Dem`, `FiM`, `NvM` |
        | **BSW ComStack** | Communication routing, CAN TP, PDU router | `CanTp`, `CanIf`, `PduR`, `Com` |
        | **MCAL Drivers** | Direct microcontroller peripheral hardware access | `Adc`, `Pwm`, `Spi`, `Port`, `Dio` |
        """)

    with col_spec2:
        st.markdown("#### 📡 Signal & Interface Dictionary")
        st.markdown("""
        | Signal Identifier | Data Type | Physical Range | Update Period | Port / Interface |
        |---|---|---|---|---|
        | `Sig_PedalTravel_pct` | `float32` | 0.0 .. 100.0 % | 5 ms | `If_DriverBrakeRequest` |
        | `Sig_BrakeIntent_enum` | `uint8` | 0=None, 1=Comfort, 2=Panic | 5 ms | `If_DriverBrakeRequest` |
        | `Sig_WheelSpeed_FL_kmh` | `float32` | 0.0 .. 350.0 km/h | 10 ms | `If_WheelSpeedData` |
        | `Sig_ABS_Active_flg` | `boolean` | 0=False, 1=True | 10 ms | `If_StabilityIntervention` |
        | `CAN_MSG_BRAKE_STATUS` | Frame | CAN ID `0x1A4` | 10 ms | Powertrain / Chassis Bus |
        """)
