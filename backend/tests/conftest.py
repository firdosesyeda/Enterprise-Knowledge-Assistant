"""
Shared pytest fixtures and configuration.
"""
import asyncio
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop."""
    policy = asyncio.DefaultEventLoopPolicy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mock_embedder():
    """Reusable mock sentence transformer that doesn't need network."""
    inst = MagicMock()
    inst.get_sentence_embedding_dimension.return_value = 8
    def _encode(texts, **kw):
        arr = np.random.rand(len(texts), 8).astype("float32")
        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)
    inst.encode = _encode
    return inst


@pytest.fixture
def tmp_store(tmp_path, mock_embedder):
    """A VectorStore instance backed by a temp dir with mocked embedder."""
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_embedder):
        from app.services.vector_store import VectorStore
        store = VectorStore(tmp_path)
        store.initialize()
        yield store


@pytest.fixture
def sample_chunks():
    """Reusable list of document chunks for tests."""
    return [
        {"text": "Employees are entitled to 24 paid leaves annually including 12 casual and 12 sick leaves.",
         "doc_name": "HR_Policy.txt", "doc_type": "HR Policy", "page_num": 1, "chunk_index": 0},
        {"text": "Customers may request a full refund within 30 days of initial purchase.",
         "doc_name": "Customer_FAQ.txt", "doc_type": "Customer FAQ", "page_num": 5, "chunk_index": 0},
        {"text": "Remote work is permitted up to 2 days per week with manager approval.",
         "doc_name": "HR_Policy.txt", "doc_type": "HR Policy", "page_num": 3, "chunk_index": 1},
        {"text": "API rate limits: Starter 1000/hr, Business 10000/hr, Enterprise 100000/hr.",
         "doc_name": "Technical_Guide.txt", "doc_type": "Technical Guide", "page_num": 2, "chunk_index": 0},
        {"text": "All data is encrypted at rest using AES-256 and in transit using TLS 1.3.",
         "doc_name": "Compliance_Guidelines.txt", "doc_type": "Compliance", "page_num": 4, "chunk_index": 0},
    ]
