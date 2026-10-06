import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SyncJobManager:
    """At most ONE sync runs at a time.

    A request that arrives while a sync is running does not start a second one:
    it marks one follow-up run as queued (merged into a single run, with a full
    rebuild if any of the merged requests asked for it). That is what you want
    when several uploads arrive back to back.
    """

    def __init__(self, rag):
        self._rag = rag
        self._lock = threading.Lock()
        self._running = False
        self._queued = False
        self._queued_force = False
        self._force = False
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None

    def request(self, force_rebuild: bool = False) -> Dict[str, Any]:
        with self._lock:
            if self._running:
                self._queued = True
                self._queued_force = self._queued_force or force_rebuild
            else:
                self._running = True
                self._force = force_rebuild
                self._started_at = _now()
                threading.Thread(
                    target=self._run,
                    args=(force_rebuild,),
                    name="sync-worker",
                    daemon=True,
                ).start()
            return self._status_locked()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> Dict[str, Any]:
        return {
            "state": "running" if self._running else "idle",
            "queued": self._queued,
            "force_rebuild": self._force,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "last_report": self._last_report,
            "last_error": self._last_error,
        }

    def _run(self, force_rebuild: bool) -> None:
        while True:
            try:
                report = self._rag.sync(force_rebuild=force_rebuild)
                error = None
            except Exception as exc:  # the engine already recovers from most failures
                logger.exception("Background sync failed")
                report, error = None, str(exc)

            with self._lock:
                if report is not None:
                    self._last_report = report
                self._last_error = error
                self._finished_at = _now()

                if self._queued:
                    force_rebuild = self._queued_force
                    self._queued = self._queued_force = False
                    self._force = force_rebuild
                    self._started_at = _now()
                    continue

                self._running = False
                return