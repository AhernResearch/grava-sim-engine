"""Thread-safe dataset registry: name → metric_cache_dir mapping."""

from __future__ import annotations

import threading
from typing import Dict


class DatasetRegistry:
    """Maps dataset names to metric_cache_dir paths.

    Thread-safe for use with multi-worker uvicorn.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._datasets: Dict[str, str] = {}

    def register(self, name: str, path: str) -> None:
        """Register a dataset name → path mapping."""
        with self._lock:
            self._datasets[name] = path

    def unregister(self, name: str) -> None:
        """Remove a dataset. Raises KeyError if not found."""
        with self._lock:
            del self._datasets[name]

    def resolve(self, name: str) -> str:
        """Resolve dataset name to path. Raises KeyError if not found."""
        with self._lock:
            return self._datasets[name]

    def list_all(self) -> Dict[str, str]:
        """Return a snapshot of all registered datasets."""
        with self._lock:
            return dict(self._datasets)

    def __len__(self) -> int:
        with self._lock:
            return len(self._datasets)

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._datasets
