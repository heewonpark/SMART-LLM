# resources/bus.py
from threading import Event, Lock
from typing import Any, Dict, Optional

class _Signal:
    def __init__(self):
        self.event = Event()
        self.payload: Optional[Dict[str, Any]] = None

class Bus:
    _instance = None
    _lock = Lock()
    def __init__(self):
        self._signals: Dict[str, _Signal] = {}
        self._lock = Lock()

    @classmethod
    def get(cls) -> "Bus":
        with cls._lock:
            if cls._instance is None:
                cls._instance = Bus()
            return cls._instance

    def _get_sig(self, name: str) -> _Signal:
        with self._lock:
            if name not in self._signals:
                self._signals[name] = _Signal()
            return self._signals[name]

    # Non-blocking notify (optionally with payload)
    def publish(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        sig = self._get_sig(name)
        sig.payload = payload
        sig.event.set()

    # Block until someone publishes this signal (returns payload)
    def wait(self, name: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        sig = self._get_sig(name)
        ok = sig.event.wait(timeout=timeout)
        return sig.payload if ok else None

    # Reset a signal so it can be used again
    def clear(self, name: str) -> None:
        sig = self._get_sig(name)
        sig.event.clear()
        sig.payload = None

