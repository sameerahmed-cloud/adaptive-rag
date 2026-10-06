"""Shared test setup.

The real engine needs Qdrant, Gemini and models. These tests replace it with a
small fake, so they run in seconds, cost nothing, and test the API layer only:
routing, validation, auth, limits, uploads, background jobs, error handling.
"""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api import app as app_module
from adaptive_rag.api import security

API_KEY = "test-key"


def verified_result():
    return {
        "answer": "Refunds are allowed within 30 days [Context 1].",
        "status": "verified",
        "verified": True,
        "strategy": "semantic",
        "strategy_reason": "conceptual/semantic question",
        "attempts": 1,
        "sources": [{
            "context": 1, "file_name": "policy.md", "relative_path": "policy.md",
            "document_id": "abc123", "sheet_name": None, "header_path": "/Policy/",
            "chunk_index": 0, "score": 0.91, "snippet": "Refunds are allowed...",
        }],
        "evaluation": {
            "relevant": True, "faithful": True, "complete": True,
            "verdict": "PASS", "issues": [], "evaluation_error": False,
        },
        "latency_seconds": 1.2,
    }


class FakeRAG:
    """Stands in for AdaptiveRAG. Every behavior a test needs is a plain attribute."""

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ready = True
        self.manifest = {}

        self.ask_result = verified_result()
        self.ask_error = None
        self.ask_calls = []              # (question, rag_mode)

        self.sync_calls = []             # force_rebuild flag per run
        self.sync_error = None
        self.sync_gate = None            # threading.Event: when set, sync blocks until released
        self.sync_started = threading.Event()

        self.remove_result = True
        self.remove_error = None
        self.removed = []                # (relative_path, delete_file)

    def is_ready(self):
        return self.ready

    def status(self):
        return {
            "ready": self.ready, "qdrant_reachable": True,
            "documents_indexed": len(self.manifest), "documents_failed": 0,
            "failed_documents": {}, "nodes_indexed": 0, "lexical_nodes": 0,
            "last_sync": {},
        }

    def manifest_snapshot(self):
        return {path: dict(record) for path, record in self.manifest.items()}

    def ask_detailed(self, question, rag_mode="auto", max_retries=3):
        self.ask_calls.append((question, rag_mode))
        if self.ask_error:
            raise self.ask_error
        return dict(self.ask_result)

    def sync(self, force_rebuild=False):
        self.sync_calls.append(force_rebuild)
        self.sync_started.set()
        if self.sync_gate is not None:
            self.sync_gate.wait(timeout=5)
        if self.sync_error:
            raise self.sync_error
        return {
            "added": [], "updated": [], "deleted": [], "unchanged": 0,
            "failed": [], "skipped": [], "rebuild": None, "duration_seconds": 0.0,
        }

    def remove_document(self, relative_path, delete_file=True):
        if self.remove_error:
            raise self.remove_error
        self.removed.append((relative_path, delete_file))
        return self.remove_result


@pytest.fixture
def fake_rag(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return FakeRAG(data_dir)


@pytest.fixture
def client(monkeypatch, fake_rag):
    """A test client with the fake engine, auth ON, and sync-on-startup OFF."""
    monkeypatch.setattr(app_module, "AdaptiveRAG", lambda: fake_rag)
    monkeypatch.setattr(app_module, "SYNC_ON_STARTUP", False)
    monkeypatch.setattr(app_module, "MAX_CONCURRENT_QUERIES", 2)
    monkeypatch.setattr(security, "API_KEYS", [API_KEY])
    monkeypatch.setattr(security, "ask_limiter", security.RateLimiter(1000))

    # raise_server_exceptions=False: a crash becomes a real 500 response, like in production.
    with TestClient(
        app_module.app,
        raise_server_exceptions=False,
        headers={"X-API-Key": API_KEY},
    ) as test_client:
        yield test_client


@pytest.fixture
def wait_for():
    def _wait(predicate, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    return _wait
