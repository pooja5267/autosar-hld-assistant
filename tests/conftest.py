"""Pytest global configuration and fixtures."""

import os

# Guarantee fast, reliable, and completely offline execution for all unit tests
os.environ["AUTOSAR_TEST_OFFLINE_EMBEDDINGS"] = "1"
