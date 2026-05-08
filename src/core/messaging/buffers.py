from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BufferedValue:
    value: Any
    timestamp: float
    sequence: int


class LatestFrameBuffer:
    """Thread-safe container that always exposes the newest BGR frame."""

    def __init__(self) -> None:
        self.frame_bgr = None
        self.timestamp = 0.0
        self.sequence = 0
        self.lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def write(self, frame) -> None:
        if frame is None:
            return
        with self.lock:
            self.frame_bgr = frame
            self.timestamp = time.time()
            self.sequence += 1

    def read_latest(self, copy_frame: bool = True):
        with self.lock:
            if self.frame_bgr is None:
                return None, 0.0, 0
            frame = self.frame_bgr.copy() if copy_frame else self.frame_bgr
            return frame, self.timestamp, self.sequence


class LatestValueBuffer:
    """Thread-safe latest-value buffer with timestamp and sequence tracking."""

    def __init__(self, *, copy_on_write: bool = False) -> None:
        self._value = None
        self._timestamp = 0.0
        self._sequence = 0
        self._copy_on_write = bool(copy_on_write)
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def write(self, value: Any, *, timestamp: float | None = None) -> BufferedValue:
        now = time.time() if timestamp is None else float(timestamp)
        stored_value = copy.deepcopy(value) if self._copy_on_write else value
        with self._lock:
            self._value = stored_value
            self._timestamp = now
            self._sequence += 1
            return BufferedValue(self._value, self._timestamp, self._sequence)

    def read_latest(self, *, copy_value: bool = False, with_metadata: bool = False):
        with self._lock:
            value = copy.deepcopy(self._value) if copy_value else self._value
            if with_metadata:
                return value, self._timestamp, self._sequence
            return value

    def read_buffered(self, *, copy_value: bool = False) -> BufferedValue:
        value, timestamp, sequence = self.read_latest(copy_value=copy_value, with_metadata=True)
        return BufferedValue(value=value, timestamp=timestamp, sequence=sequence)

    def read_value(self, *, copy_value: bool = False) -> Any:
        return self.read_latest(copy_value=copy_value, with_metadata=False)
