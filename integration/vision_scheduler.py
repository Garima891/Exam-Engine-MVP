from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Condition, Lock
from time import perf_counter
from typing import Any, Callable, Literal


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    image: Any
    capture_ms: float = 0.0


@dataclass(frozen=True)
class TimedResult:
    value: Any
    started_at: float
    completed_at: float

    @property
    def duration_ms(self) -> float:
        return (self.completed_at - self.started_at) * 1000


@dataclass(frozen=True)
class ProcessedFrame:
    frame: CapturedFrame
    mode: Literal["sequential", "concurrent"]
    dispatched_at: float
    completed_at: float
    phone: TimedResult
    face: TimedResult

    @property
    def queue_delay_ms(self) -> float:
        return (self.dispatched_at - self.frame.captured_at) * 1000

    @property
    def processing_ms(self) -> float:
        return (self.completed_at - self.dispatched_at) * 1000

    @property
    def frame_age_ms(self) -> float:
        return (self.completed_at - self.frame.captured_at) * 1000


class LatestFrameMailbox:
    """A one-frame slot that replaces pending stale frames with newer ones."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._pending: CapturedFrame | None = None
        self._closed = False
        self._replaced_count = 0

    def publish(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._closed:
                raise RuntimeError("Cannot publish to a closed frame mailbox.")
            replaced = self._pending is not None
            if replaced:
                self._replaced_count += 1
            self._pending = frame
            self._condition.notify()
            return replaced

    def take(self, timeout: float | None = None) -> CapturedFrame | None:
        with self._condition:
            deadline = None if timeout is None else perf_counter() + timeout
            while self._pending is None and not self._closed:
                remaining = None if deadline is None else deadline - perf_counter()
                if remaining is not None and remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._pending is not None:
                frame, self._pending = self._pending, None
                return frame
            return None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def replaced_count(self) -> int:
        with self._condition:
            return self._replaced_count


class VisionFrameProcessor:
    """Runs one phone and one face inference for a captured frame."""

    def __init__(
        self,
        phone_adapter: Any,
        face_adapter: Any,
        mode: Literal["sequential", "concurrent"],
    ) -> None:
        if mode not in ("sequential", "concurrent"):
            raise ValueError(f"Unsupported execution mode: {mode}")

        self._phone_adapter = phone_adapter
        self._face_adapter = face_adapter
        self.mode = mode
        self._process_lock = Lock()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="vision-adapter",
            )
            if mode == "concurrent"
            else None
        )

    @staticmethod
    def _timed_call(
        function: Callable[[Any], Any],
        frame: Any,
    ) -> TimedResult:
        started_at = perf_counter()
        value = function(frame)
        return TimedResult(value, started_at, perf_counter())

    def process(self, frame: CapturedFrame) -> ProcessedFrame:
        if not self._process_lock.acquire(blocking=False):
            raise RuntimeError("A frame is already being processed.")

        try:
            dispatched_at = perf_counter()
            if self.mode == "sequential":
                phone_result = self._timed_call(
                    self._phone_adapter.detect,
                    frame.image,
                )
                face_result = self._timed_call(
                    self._face_adapter.extract_all_face_embeddings,
                    frame.image,
                )
            else:
                if self._executor is None:
                    raise RuntimeError("Concurrent executor is unavailable.")
                phone_future = self._executor.submit(
                    self._timed_call,
                    self._phone_adapter.detect,
                    frame.image,
                )
                face_future = self._executor.submit(
                    self._timed_call,
                    self._face_adapter.extract_all_face_embeddings,
                    frame.image,
                )
                phone_result, face_result = self._get_both_results(
                    phone_future,
                    face_future,
                )

            completed_at = perf_counter()
            return ProcessedFrame(
                frame=frame,
                mode=self.mode,
                dispatched_at=dispatched_at,
                completed_at=completed_at,
                phone=phone_result,
                face=face_result,
            )
        finally:
            self._process_lock.release()

    @staticmethod
    def _get_both_results(
        phone_future: Future[TimedResult],
        face_future: Future[TimedResult],
    ) -> tuple[TimedResult, TimedResult]:
        try:
            return phone_future.result(), face_future.result()
        except BaseException:
            wait((phone_future, face_future))
            raise

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "VisionFrameProcessor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False
