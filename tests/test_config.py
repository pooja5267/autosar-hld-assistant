"""Unit tests for configuration management module."""

import os
import pytest
from src.config import AppConfig, get_config


def test_default_config_initialization():
    """Verify default configuration attributes and fallback values."""
    config = AppConfig()
    assert config.llm_provider in ["offline", "openai", "groq", "gemini", "anthropic", "ollama"]
    assert config.retrieval_top_k >= 1
    assert 0.0 <= config.similarity_threshold <= 1.0
    assert config.max_file_size_mb > 0
    assert ".pdf" in config.allowed_extensions


def test_active_api_key_selection():
    """Verify provider-specific active API key resolution."""
    # Test OpenAI
    cfg_openai = AppConfig(llm_provider="openai", openai_api_key="sk-test-openai")
    assert cfg_openai.get_active_api_key() == "sk-test-openai"

    # Test Groq
    cfg_groq = AppConfig(llm_provider="groq", groq_api_key="gsk-test-groq")
    assert cfg_groq.get_active_api_key() == "gsk-test-groq"

    # Test Gemini
    cfg_gemini = AppConfig(llm_provider="gemini", gemini_api_key="gemini-key-12345")
    assert cfg_gemini.get_active_api_key() == "gemini-key-12345"

    # Test Anthropic
    cfg_anthropic = AppConfig(llm_provider="anthropic", anthropic_api_key="sk-ant-test-key")
    assert cfg_anthropic.get_active_api_key() == "sk-ant-test-key"

    # Test Ollama
    cfg_ollama = AppConfig(llm_provider="ollama")
    assert cfg_ollama.get_active_api_key() == "ollama-local"

    # Test Offline
    cfg_offline = AppConfig(llm_provider="offline")
    assert cfg_offline.get_active_api_key() is None


def test_masked_api_key_display():
    """Verify secure masking of API keys to prevent secret exposure in UI."""
    # Long key
    cfg = AppConfig(llm_provider="openai", openai_api_key="sk-1234567890abcdef")
    masked = cfg.get_masked_api_key()
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "..." in masked
    assert "1234567890" not in masked

    # Short key
    cfg_short = AppConfig(llm_provider="openai", openai_api_key="12345")
    assert cfg_short.get_masked_api_key() == "********"

    # Offline mode
    cfg_off = AppConfig(llm_provider="offline")
    assert "Offline Mode" in cfg_off.get_masked_api_key()

    # Ollama mode
    cfg_oll = AppConfig(llm_provider="ollama")
    assert "No Key Required" in cfg_oll.get_masked_api_key()


def test_config_to_dict():
    """Verify dictionary serialization for telemetry and status display."""
    cfg = AppConfig(
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        groq_api_key="gsk-secret",
        retrieval_top_k=5,
        similarity_threshold=0.4
    )
    d = cfg.to_dict()
    assert d["llm_provider"] == "groq"
    assert d["llm_model"] == "llama-3.3-70b-versatile"
    assert d["api_key_configured"] is True
    assert d["top_k"] == 5
    assert d["similarity_threshold"] == 0.4
    assert "gsk-secret" not in str(d)  # Secret key itself is not leaked in to_dict


def test_get_config_singleton_factory():
    """Verify get_config returns a valid AppConfig instance."""
    cfg = get_config()
    assert isinstance(cfg, AppConfig)
