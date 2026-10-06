"""End-to-end tests of the HTTP layer, with a fake engine behind it."""

import threading

import pytest

from adaptive_rag.api import routes_documents, security
from adaptive_rag.config import CORS_ORIGINS, MAX_QUESTION_CHARS



def upload(client, name="notes.txt", content=b"hello world", **params):
    return client.post(
        "/documents",
        files={"file": (name, content, "text/plain")},
        params=params,
    )


# --------------------------------------------------------------------------- #
# Health, readiness, request ids, CORS
# --------------------------------------------------------------------------- #

def test_health_is_open_and_ready_reflects_engine(client, fake_rag):
    assert client.get("/health", headers={"X-API-Key": ""}).json() == {"status": "ok"}

    assert client.get("/ready").status_code == 200
    fake_rag.ready = False
    response = client.get("/ready")
    assert response.status_code == 503 and response.json()["ready"] is False


def test_request_id_is_echoed_or_generated(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"

    # Unsafe values (could forge log lines) are replaced, not echoed.
    bad = "bad id with spaces"
    generated = client.get("/health", headers={"X-Request-ID": bad}).headers["x-request-id"]
    assert generated != bad and len(generated) == 16


def test_cors_allows_configured_origin_only(client):
    allowed = CORS_ORIGINS[0]
    headers = {"Origin": allowed, "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "x-api-key,content-type"}
    ok = client.options("/ask", headers=headers)
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == allowed

    blocked = client.options("/ask", headers={**headers, "Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in blocked.headers


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #

PROTECTED = [
    ("get", "/documents", None),
    ("get", "/status", None),
    ("get", "/metrics", None),
    ("get", "/sync", None),
    ("post", "/sync", {}),
    ("post", "/ask", {"question": "hi"}),
]


@pytest.mark.parametrize("method,url,body", PROTECTED)
@pytest.mark.parametrize("key", ["", "wrong-key"])
def test_protected_endpoints_reject_bad_keys(client, method, url, body, key):
    response = client.request(method, url, json=body, headers={"X-API-Key": key})
    assert response.status_code == 401


# --------------------------------------------------------------------------- #
# POST /ask
# --------------------------------------------------------------------------- #

def test_ask_returns_the_contract_the_web_app_needs(client, fake_rag):
    response = client.post("/ask", json={"question": "What is the refund window?"})
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "verified" and body["verified"] is True
    assert body["sources"][0]["context"] == 1
    assert body["evaluation"]["verdict"] == "PASS"
    assert body["request_id"] == response.headers["x-request-id"]
    assert fake_rag.ask_calls == [("What is the refund window?", "auto")]


def test_ask_passes_mode_and_trims_the_question(client, fake_rag):
    client.post("/ask", json={"question": "  spaced out  ", "mode": "hybrid"})
    assert fake_rag.ask_calls == [("spaced out", "hybrid")]


@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {"question": "x" * (MAX_QUESTION_CHARS + 1)},
    {"question": "ok", "mode": "telepathy"},
    {},
])
def test_ask_rejects_invalid_input_before_the_engine(client, fake_rag, payload):
    assert client.post("/ask", json=payload).status_code == 422
    assert fake_rag.ask_calls == []


def test_engine_validation_errors_become_422(client, fake_rag):
    fake_rag.ask_error = ValueError("Question cannot be empty.")
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


@pytest.mark.parametrize("status", ["no_evidence", "unverified"])
def test_unanswerable_outcomes_are_still_http_200(client, fake_rag, status):
    fake_rag.ask_result = {**fake_rag.ask_result, "status": status, "verified": False, "sources": []}
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 200 and response.json()["status"] == status


@pytest.mark.parametrize("status", ["not_ready", "error"])
def test_system_failures_are_http_503_with_retry_after(client, fake_rag, status):
    fake_rag.ask_result = {**fake_rag.ask_result, "status": status, "answer": "try later"}
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 503
    assert response.json()["detail"] == "try later" and "retry-after" in response.headers


def test_ask_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(security, "ask_limiter", security.RateLimiter(3))
    codes = [client.post("/ask", json={"question": "hi"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert "retry-after" in client.post("/ask", json={"question": "hi"}).headers


def test_unexpected_engine_crash_does_not_leak_details(client, fake_rag):
    fake_rag.ask_error = RuntimeError("secret detail at /srv/private/path")
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 500
    assert "secret" not in response.text and "/srv" not in response.text
    assert response.json()["request_id"]


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

def test_list_documents_shows_indexed_and_failed(client, fake_rag):
    fake_rag.manifest = {
        "bad.csv": {"status": "failed", "extension": ".csv", "size": 9, "indexed_nodes": 0,
                    "failure": {"error": "parsing failed: x", "attempts": 2}},
        "a.pdf": {"status": "indexed", "extension": ".pdf", "size": 123, "indexed_nodes": 4},
    }
    documents = client.get("/documents").json()

    assert [d["path"] for d in documents] == ["a.pdf", "bad.csv"]          # sorted
    assert documents[0]["status"] == "indexed" and documents[0]["indexed_nodes"] == 4
    assert documents[1]["status"] == "failed" and documents[1]["attempts"] == 2


def test_upload_saves_the_file_and_starts_a_sync(client, fake_rag, wait_for):
    response = upload(client)
    assert response.status_code == 202
    assert response.json()["path"] == "notes.txt" and response.json()["replaced"] is False
    assert (fake_rag.data_dir / "notes.txt").read_bytes() == b"hello world"

    assert wait_for(lambda: fake_rag.sync_calls == [False])
    assert wait_for(lambda: client.get("/sync").json()["state"] == "idle")
    assert not list(fake_rag.data_dir.glob(".upload-*"))                   # no temp leftovers


def test_upload_can_skip_the_sync(client, fake_rag):
    response = upload(client, sync="false")
    assert response.status_code == 202 and response.json()["sync"] is None
    assert fake_rag.sync_calls == []


def test_upload_overwrite_rules(client):
    assert upload(client).status_code == 202
    assert upload(client, overwrite="false").status_code == 409
    again = upload(client, content=b"new content")
    assert again.status_code == 202 and again.json()["replaced"] is True


def test_upload_rejects_unsafe_or_unsupported_files(client, fake_rag):
    assert upload(client, name="malware.exe").status_code == 415
    assert upload(client, name="README").status_code == 415
    assert upload(client, name=".env").status_code == 422                  # hidden names refused
    assert upload(client, content=b"").status_code == 422                  # empty file
    assert not list(fake_rag.data_dir.iterdir())                           # nothing was written


def test_upload_filename_cannot_escape_the_data_folder(client, fake_rag):
    response = upload(client, name="../../evil.txt", sync="false")
    assert response.status_code == 202 and response.json()["path"] == "evil.txt"
    assert (fake_rag.data_dir / "evil.txt").exists()
    assert not (fake_rag.data_dir.parent / "evil.txt").exists()
    assert not (fake_rag.data_dir.parent.parent / "evil.txt").exists()


def test_upload_too_large_is_413_and_leaves_no_temp_file(client, fake_rag, monkeypatch):
    monkeypatch.setattr(routes_documents, "MAX_UPLOAD_SIZE_BYTES", 10)
    assert upload(client, content=b"x" * 100).status_code == 413
    assert not list(fake_rag.data_dir.iterdir())


def test_delete_document(client, fake_rag):
    assert client.delete("/documents/notes.txt").status_code == 204
    assert client.delete("/documents/reports/q3.pdf").status_code == 204   # subfolder path kept intact
    assert fake_rag.removed == [("notes.txt", True), ("reports/q3.pdf", True)]


def test_delete_unknown_document_is_404_and_bad_path_is_400(client, fake_rag):
    fake_rag.remove_result = False
    assert client.delete("/documents/nope.txt").status_code == 404

    fake_rag.remove_error = ValueError("Path is outside the data directory.")
    assert client.delete("/documents/x.txt").status_code == 400


def test_delete_is_refused_while_a_sync_runs(client, fake_rag, wait_for):
    fake_rag.sync_gate = threading.Event()
    client.post("/sync", json={})
    assert wait_for(fake_rag.sync_started.is_set)

    assert client.delete("/documents/notes.txt").status_code == 409

    fake_rag.sync_gate.set()
    assert wait_for(lambda: client.get("/sync").json()["state"] == "idle")


# --------------------------------------------------------------------------- #
# Sync jobs and system endpoints
# --------------------------------------------------------------------------- #

def test_sync_requests_during_a_run_merge_into_one_follow_up(client, fake_rag, wait_for):
    fake_rag.sync_gate = threading.Event()
    assert client.post("/sync", json={}).status_code == 202
    assert wait_for(fake_rag.sync_started.is_set)

    client.post("/sync", json={})
    client.post("/sync", json={"force_rebuild": True})
    assert client.get("/sync").json()["queued"] is True

    fake_rag.sync_gate.set()
    assert wait_for(lambda: client.get("/sync").json()["state"] == "idle")
    assert fake_rag.sync_calls == [False, True]        # two runs, the second a full rebuild


def test_failed_sync_is_reported_not_raised(client, fake_rag, wait_for):
    fake_rag.sync_error = RuntimeError("boom")
    client.post("/sync", json={})
    assert wait_for(lambda: client.get("/sync").json()["state"] == "idle" and fake_rag.sync_calls)

    status = client.get("/sync").json()
    assert status["last_error"] == "boom" and status["state"] == "idle"


def test_status_and_metrics(client):
    status = client.get("/status").json()
    assert status["ready"] is True and "sync_job" in status
    assert "llm" in client.get("/metrics").json()