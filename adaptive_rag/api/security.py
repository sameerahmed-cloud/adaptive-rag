import re
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional

from fastapi import Header, HTTPException, Request

from ..config import API_KEYS, ASK_RATE_LIMIT_PER_MINUTE


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests without a valid key. With no keys configured, auth is
    off (development); app startup logs a warning in that case."""
    if not API_KEYS:
        return
    if not x_api_key or not any(
        secrets.compare_digest(x_api_key, key) for key in API_KEYS
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class RateLimiter:
    """In-memory sliding-window limiter, one window per client.

    Good enough for ONE server process. With several processes or machines,
    move this to Redis (each process would otherwise allow its own quota).
    """

    def __init__(self, max_calls: int, window_seconds: float = 60.0):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float:
        """0.0 if the call is allowed (and counted), else seconds until it would be."""
        if self.max_calls <= 0:
            return 0.0
        now = time.monotonic()
        with self._lock:
            calls = self._calls[key]
            while calls and now - calls[0] >= self.window:
                calls.popleft()
            if len(calls) >= self.max_calls:
                return max(0.1, self.window - (now - calls[0]))
            calls.append(now)
            if len(self._calls) > 10_000:
                self._purge(now)
            return 0.0

    def _purge(self, now: float) -> None:
        for key in list(self._calls):
            calls = self._calls[key]
            while calls and now - calls[0] >= self.window:
                calls.popleft()
            if not calls:
                del self._calls[key]


ask_limiter = RateLimiter(ASK_RATE_LIMIT_PER_MINUTE)


def client_identity(request: Request) -> str:
    # Behind a reverse proxy, start uvicorn with --proxy-headers so
    # request.client reflects the real caller instead of the proxy.
    return (
        request.headers.get("X-API-Key")
        or (request.client.host if request.client else "unknown")
    )


def enforce_ask_rate_limit(request: Request) -> None:
    wait = ask_limiter.retry_after(client_identity(request))
    if wait > 0:
        raise HTTPException(
            status_code=429,
            detail="Too many questions. Please slow down.",
            headers={"Retry-After": str(int(wait) + 1)},
        )


_UNSAFE_CHARS = re.compile(r"[^\w.\- ()]", re.UNICODE)


def safe_filename(raw: str, max_length: int = 150) -> str:
    """Reduce an uploaded filename to a harmless flat name, or raise ValueError.

    Removes any directory part (so ../../x cannot escape the data folder),
    replaces odd characters, and refuses hidden names (the sync ignores them).
    """
    name = Path(raw.replace("\\", "/")).name.strip()
    name = _UNSAFE_CHARS.sub("_", name)
    if not name or name.startswith(".") or name in {".", ".."}:
        raise ValueError("Invalid file name.")

    path = Path(name)
    if len(name) > max_length:
        name = path.stem[: max_length - len(path.suffix)] + path.suffix
    return name