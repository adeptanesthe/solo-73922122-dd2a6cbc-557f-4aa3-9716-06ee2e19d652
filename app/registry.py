"""Thread-safe in-memory registry for the two reference samples."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Sample:
    slot: int
    image: np.ndarray  # (H, W, 3) uint8, read-only
    background: Tuple[int, int, int]
    version: int
    filename: str
    updated_at: str


@dataclass(frozen=True)
class PairSnapshot:
    first: Optional[Sample]
    second: Optional[Sample]
    version_token: Tuple[Optional[int], Optional[int]]


class SampleRegistry:
    """Holds at most two samples.

    Updates are accepted at any time, including while a composition is running.
    A composition snapshots the two slots, computes outside the lock, and then
    checks the version token again before answering.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: Dict[int, Sample] = {}
        self._revision = 0

    def put(
        self,
        slot: int,
        image: np.ndarray,
        background: Tuple[int, int, int],
        filename: str,
    ) -> Sample:
        if slot not in (1, 2):
            raise ValueError("slot must be 1 or 2")
        frozen = np.ascontiguousarray(image, dtype=np.uint8)
        frozen.setflags(write=False)
        with self._lock:
            self._revision += 1
            sample = Sample(
                slot=slot,
                image=frozen,
                background=tuple(int(v) for v in background),
                version=self._revision,
                filename=filename,
                updated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
            self._samples[slot] = sample
            return sample

    def delete(self, slot: int) -> bool:
        with self._lock:
            if slot not in self._samples:
                return False
            del self._samples[slot]
            self._revision += 1
            return True

    def get(self, slot: int) -> Optional[Sample]:
        with self._lock:
            return self._samples.get(slot)

    def snapshot(self) -> PairSnapshot:
        with self._lock:
            first = self._samples.get(1)
            second = self._samples.get(2)
            token = (
                first.version if first else None,
                second.version if second else None,
            )
            return PairSnapshot(first, second, token)

    def is_current(self, token: Tuple[Optional[int], Optional[int]]) -> bool:
        with self._lock:
            first = self._samples.get(1)
            second = self._samples.get(2)
            current = (
                first.version if first else None,
                second.version if second else None,
            )
            return current == token

    def revision(self) -> int:
        with self._lock:
            return self._revision

    def pairs_match(self) -> bool:
        """True when both slots are filled and share one image size."""
        with self._lock:
            first = self._samples.get(1)
            second = self._samples.get(2)
            if first is None or second is None:
                return False
            return first.image.shape == second.image.shape
