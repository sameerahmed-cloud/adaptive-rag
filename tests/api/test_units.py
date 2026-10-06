"""Fast tests for the pieces that do not need an HTTP client."""

import threading
import time
from threading import BoundedSemaphore
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from adaptive_rag.api import deps, security
from adaptive_rag.api.jobs import SyncJobManager


# ---- filenames --------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd.txt", "passwd.txt"),
    ("C:\\Users\\me\\report final.pdf", "report final.pdf"),
    ("data<>|?.csv", "data____.csv"),
    ("تقرير.pdf", "تقرير.pdf"),
])
def test_safe_filename_cleans_names(raw, expected):
    assert security.safe_filename(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", ".env", "..", "../.hidden"])
def test_safe_filename_refuses_bad_names(raw):
    with pytest.raises(ValueError):
        security.safe_filename(raw)


def test_safe_filename_shortens_long_names_but_keeps_extension():
    name = security.safe_filename("a" * 300 + ".pdf")
    assert len(name) <= 150 and name.endswith(".pdf")


# ---- api key ----------------------------------------------------------------

def test_api_key_check(monkeypatch):
    monkeypatch.setattr(security, "API_KEYS", ["secret"])
    security.require_api_key("secret")
    for bad in (None, "", "wrong"):
        with pytest.raises(HTTPException) as error:
            security.require_api_key(bad)
        assert error.value.status_code == 401

    monkeypatch.setattr(security, "API_KEYS", [])      # no keys configured: auth is off
    security.require_api_key(None)


# ---- rate limiter -----------------------------------------------------------

def test_rate_limiter_counts_per_client_and_window_slides():
    limiter = security.RateLimiter(3, window_seconds=0.3)
    assert [limiter.retry_after("a") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.retry_after("a") > 0
    assert limiter.retry_after("b") == 0.0             # another client is unaffected
    time.sleep(0.35)
    assert limiter.retry_after("a") == 0.0             # the window moved on


def test_rate_limiter_zero_disables_it():
    assert security.RateLimiter(0).retry_after("x") == 0.0


# ---- concurrency slots ------------------------------------------------------

def test_query_slot_rejects_when_full_and_releases_after(monkeypatch):
    monkeypatch.setattr(deps, "QUERY_QUEUE_TIMEOUT_SECONDS", 0.05)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(query_slots=BoundedSemaphore(1))))

    with deps.query_slot(request):
        with pytest.raises(HTTPException) as error:
            with deps.query_slot(request):
                pass
        assert error.value.status_code == 429

    with deps.query_slot(request):                      # the slot was released
        pass


# ---- sync job manager -------------------------------------------------------

class FakeEngine:
    def __init__(self):
        self.calls, self.gate, self.fail = [], threading.Event(), False

    def sync(self, force_rebuild=False):
        self.calls.append(force_rebuild)
        self.gate.wait(2)
        if self.fail:
            raise RuntimeError("boom")
        return {"added": ["a"]}


def _wait_idle(manager):
    for _ in range(100):
        if not manager.is_running():
            return
        time.sleep(0.02)
    raise AssertionError("sync did not finish")


def test_job_manager_runs_one_at_a_time_and_merges_requests():
    engine = FakeEngine()
    manager = SyncJobManager(engine)

    assert manager.request()["state"] == "running"
    time.sleep(0.05)
    manager.request()
    manager.request(force_rebuild=True)
    assert manager.status()["queued"] is True

    engine.gate.set()
    _wait_idle(manager)
    assert engine.calls == [False, True]
    assert manager.status()["last_report"] == {"added": ["a"]}


def test_job_manager_records_errors():
    engine = FakeEngine()
    engine.gate.set()
    engine.fail = True
    manager = SyncJobManager(engine)
    manager.request()
    _wait_idle(manager)
    assert manager.status()["last_error"] == "boom" and manager.status()["state"] == "idle"
