"""Configuration management for AUTOSAR HLD Document Analysis Assistant."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_HLD_DIR = DATA_DIR / "sample_hld"
CHROMA_DEFAULT_DIR = DATA_DIR / "chroma_db"


@dataclass
class AppConfig:
    """Application configuration parameters with environment fallback."""

    # Vector store settings
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DEFAULT_DIR))
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "autosar_hld_docs")
    )
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )

    # Retrieval parameters
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "4"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
    )

    # LLM Provider settings
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "offline").lower()
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    llm_api_key: Optional[str] = field(
        default_factory=lambda: (
            os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or None
        )
    )
    llm_base_url: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", None)
    )

    # Specific API keys
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", None)
    )
    groq_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", None)
    )
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", None)
    )
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", None)
    )

    # File constraints
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    )
    allowed_extensions: list = field(
        default_factory=lambda: [".pdf"]
    )

    def get_active_api_key(self) -> Optional[str]:
        """Return the API key corresponding to the selected provider."""
        if self.llm_provider == "openai":
            return self.openai_api_key or self.llm_api_key
        elif self.llm_provider == "groq":
            return self.groq_api_key or self.llm_api_key
        elif self.llm_provider == "gemini":
            return self.gemini_api_key or self.llm_api_key
        elif self.llm_provider == "anthropic":
            return self.anthropic_api_key or self.llm_api_key
        elif self.llm_provider == "ollama":
            return "ollama-local"  # No key needed
        return None  # offline

    def get_masked_api_key(self) -> str:
        """Return masked key for safe UI display without secret leakage."""
        key = self.get_active_api_key()
        if not key or key == "ollama-local":
            return "Not Configured (Running in Offline Mode)" if self.llm_provider == "offline" else "Local / No Key Required"
        if len(key) <= 8:
            return "********"
        return f"{key[:4]}...{key[-4:]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for telemetry or UI status."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "api_key_configured": bool(self.get_active_api_key()),
            "chroma_persist_dir": self.chroma_persist_dir,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "top_k": self.retrieval_top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_file_size_mb": self.max_file_size_mb,
        }


def get_config() -> AppConfig:
    """Retrieve application configuration instance."""
    return AppConfig()
