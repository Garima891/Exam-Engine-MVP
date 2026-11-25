from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock
import time
from typing import Any
from uuid import uuid4

from integration.config import (
    EVENT_CONFIRMATION_WINDOW_SECONDS,
    EVENT_MINIMUM_OBSERVATION_INTERVAL_SECONDS,
)


EVENT_MODULES = {
    "PHONE_DETECTED": "phone_detection",
    "IDENTITY_UNMATCHED": "face_identity",
    "NO_FACE_DETECTED": "face_identity",
    "UNKNOWN_HUMAN_VOICE": "audio_detection",
    "SUSPICIOUS_AUDIO_SOUND": "audio_detection",
}


class ProctoringEventEngine:
    """Track independent, deduplicated observations and confirmed event logs."""

    def __init__(
        self,
        confirmation_observations: int = 5,
        session_id: str | None = None,
        history_path: str | Path | None = None,
        event_persistence_callback: Any | None = None,
        confirmation_window_seconds: float = EVENT_CONFIRMATION_WINDOW_SECONDS,
        minimum_observation_intervals: dict[str, float] | None = None,
    ) -> None:
        if (
            isinstance(confirmation_observations, bool)
            or not isinstance(confirmation_observations, int)
            or confirmation_observations <= 0
        ):
            raise ValueError("Confirmation observations must be a positive integer.")
        if (
            isinstance(confirmation_window_seconds, bool)
            or not isinstance(confirmation_window_seconds, (int, float))
            or confirmation_window_seconds <= 0
        ):
            raise ValueError(
                "Confirmation window must be a positive number of seconds."
            )
        observation_intervals = (
            EVENT_MINIMUM_OBSERVATION_INTERVAL_SECONDS
            if minimum_observation_intervals is None
            else minimum_observation_intervals
        )
        if not isinstance(observation_intervals, dict):
            raise ValueError("Minimum observation intervals must be a dictionary.")
        for event_type, interval in observation_intervals.items():
            if (
                event_type not in EVENT_MODULES
                or isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not math.isfinite(float(interval))
                or interval < 0
            ):
                raise ValueError(
                    "Minimum observation intervals must map supported event "
                    "types to finite, non-negative seconds."
                )

        self.confirmation_observations = confirmation_observations
        self.confirmation_window_seconds = float(confirmation_window_seconds)
        self.minimum_observation_intervals = {
            event_type: float(interval)
            for event_type, interval in observation_intervals.items()
        }
        self.session_id = session_id or str(uuid4())
        self.history_path = None if history_path is None else Path(history_path)
        self.event_persistence_callback = event_persistence_callback
        self._counts = {event_type: 0 for event_type in EVENT_MODULES}
        self._seen: dict[str, set[str]] = defaultdict(set)
        self._seen_times: dict[str, dict[str, float]] = {
            event_type: {} for event_type in EVENT_MODULES
        }
        self._observation_times: dict[str, deque[float]] = {
            event_type: deque() for event_type in EVENT_MODULES
        }
        self._events: list[dict[str, Any]] = []
        self._history_error: str | None = None
        self._persistence_status = "NOT_CONFIGURED"
        self._lock = RLock()
        self._history_lock = Lock()
        self._historical_events = self._load_historical_events()

    @property
    def counters(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    @property
    def confirmed_events(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                event_type
                for event_type, count in self._counts.items()
                if count >= self.confirmation_observations
            )

    @property
    def event_log(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._copy_events(self._events)

    @property
    def historical_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._copy_events(self._historical_events)

    @property
    def historical_confirmed_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._copy_events(
                [
                    event for event in self._historical_events
                    if event["status"] == "CONFIRMED"
                ]
            )

    @property
    def history_error(self) -> str | None:
        with self._lock:
            return self._history_error

    @property
    def persistence_status(self) -> str:
        with self._lock:
            return self._persistence_status

    def _expire_observations(
        self,
        event_type: str,
        now_monotonic: float,
    ) -> None:
        observations = self._observation_times[event_type]
        cutoff = now_monotonic - self.confirmation_window_seconds
        while observations and observations[0] < cutoff:
            observations.popleft()
        self._counts[event_type] = len(observations)
        seen_times = self._seen_times[event_type]
        expired_ids = [
            observation_id
            for observation_id, observed_at in seen_times.items()
            if observed_at < cutoff
        ]
        for observation_id in expired_ids:
            seen_times.pop(observation_id, None)
            self._seen[event_type].discard(observation_id)

    def observe(
        self,
        event_type: str,
        observation_id: str,
        model_output: dict[str, Any],
    ) -> dict[str, Any] | None:
        if event_type not in EVENT_MODULES:
            raise ValueError(f"Unsupported event type: {event_type}")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("Observation ID must be a non-empty string.")
        if not isinstance(model_output, dict):
            raise ValueError("Model output must be a dictionary.")

        safe_output = json.loads(json.dumps(model_output, allow_nan=False))
        now_monotonic = time.monotonic()
        now = datetime.now().astimezone()
        with self._lock:
            self._expire_observations(event_type, now_monotonic)
            seen = self._seen[event_type]
            if observation_id in seen:
                return None
            minimum_interval = self.minimum_observation_intervals.get(
                event_type,
                0.0,
            )
            observations = self._observation_times[event_type]
            if (
                minimum_interval > 0.0
                and observations
                and now_monotonic - observations[-1] < minimum_interval
            ):
                return None
            if self._counts[event_type] >= self.confirmation_observations:
                return None
            seen.add(observation_id)
            self._seen_times[event_type][observation_id] = now_monotonic
            observations.append(now_monotonic)
            self._counts[event_type] += 1
            count = self._counts[event_type]
            event = {
                "session_id": self.session_id,
                "module": EVENT_MODULES[event_type],
                "event_type": event_type,
                "observation_id": observation_id,
                "observation_count": count,
                "timestamp": now.isoformat(
                    timespec="milliseconds"
                ),
                "model_output": safe_output,
                "status": (
                    "CONFIRMED"
                    if count >= self.confirmation_observations
                    else "OBSERVED"
                ),
            }
            self._events.append(event)
            returned_event = self._copy_events([event])[0]

        history_write_succeeded = True
        database_write_succeeded = True
        has_destination = False
        if self.history_path is not None:
            has_destination = True
            try:
                with self._history_lock:
                    self._append_history(event)
            except OSError as exc:
                history_write_succeeded = False
                with self._lock:
                    self._history_error = (
                        f"Could not persist event history to {self.history_path}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if self.event_persistence_callback is not None:
            has_destination = True
            try:
                self.event_persistence_callback(returned_event)
            except Exception as exc:
                database_write_succeeded = False
                with self._lock:
                    self._history_error = (
                        f"Could not persist event history to the database: "
                        f"{type(exc).__name__}: {exc}"
                    )
        with self._lock:
            if not has_destination:
                self._persistence_status = "NOT_CONFIGURED"
            elif history_write_succeeded and database_write_succeeded:
                self._persistence_status = "PERSISTED"
            else:
                self._persistence_status = "FAILED"
        return returned_event

    def _load_historical_events(self) -> list[dict[str, Any]]:
        if self.history_path is None or not self.history_path.is_file():
            return []

        events = []
        with self.history_path.open("r", encoding="utf-8") as history_file:
            for line_number, line in enumerate(history_file, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid event history at {self.history_path}:"
                        f"{line_number}: {exc}"
                    ) from exc
                self._validate_history_event(event, line_number)
                if event["session_id"] != self.session_id:
                    events.append(event)
        return events

    @staticmethod
    def _validate_history_event(event: Any, line_number: int) -> None:
        if not isinstance(event, dict):
            raise ValueError(
                f"Invalid event history record on line {line_number}: "
                "expected an object."
            )
        event_type = event.get("event_type")
        if (
            event_type not in EVENT_MODULES
            or event.get("module") != EVENT_MODULES[event_type]
            or not isinstance(event.get("session_id"), str)
            or not isinstance(event.get("timestamp"), str)
            or isinstance(event.get("observation_count"), bool)
            or not isinstance(event.get("observation_count"), int)
            or event["observation_count"] <= 0
            or not isinstance(event.get("model_output"), dict)
            or event.get("status") not in {"OBSERVED", "CONFIRMED"}
        ):
            raise ValueError(
                f"Invalid event history record on line {line_number}."
            )

    def _append_history(self, event: dict[str, Any]) -> None:
        if self.history_path is None:
            return
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf-8", newline="\n") as history_file:
            history_file.write(
                json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n"
            )

    @staticmethod
    def _copy_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return json.loads(json.dumps(events, allow_nan=False))
