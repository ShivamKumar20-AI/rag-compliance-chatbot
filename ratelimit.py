"""Shared rate limit for all visitors of the running app (a new browser tab does not reset it)."""
import threading
import time
from collections import deque


class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._lock:
            while self._times and now - self._times[0] >= self.window_seconds:
                self._times.popleft()
            if len(self._times) >= self.max_requests:
                return False
            self._times.append(now)
            return True
