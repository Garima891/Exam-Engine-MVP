from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from uuid import uuid4


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
    ) -> None:
        if (
            isinstance(confirmation_observations, bool)
            or not isinstance(confirmation_observations, int)
            or confirmation_observations <= 0
        ):
            raise ValueError("Confirmation observations must be a positive integer.")

        self.confirmation_observations = confirmation_observations
        self.session_id = session_id or str(uuid4())
        self.history_path = None if history_path is None else Path(history_path)
        self.event_persistence_callback = event_persistence_callback
        self._counts = {event_type: 0 for event_type in EVENT_MODULES}
        self._seen: dict[str, set[str]] = defaultdict(set)
        self._events: list[dict[str, Any]] = []
        self._history_error: str | None = None
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
        with self._lock:
            seen = self._seen[event_type]
            if observation_id in seen:
                return None
            if self._counts[event_type] >= self.confirmation_observations:
                return None
            seen.add(observation_id)
            self._counts[event_type] += 1
            count = self._counts[event_type]
            event = {
                "session_id": self.session_id,
                "module": EVENT_MODULES[event_type],
                "event_type": event_type,
                "observation_id": observation_id,
                "observation_count": count,
                "timestamp": datetime.now().astimezone().isoformat(
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

        if self.history_path is not None:
            try:
                with self._history_lock:
                    self._append_history(event)
            except OSError as exc:
                with self._lock:
                    self._history_error = (
                        f"Could not persist event history to {self.history_path}: "
                        f"{exc}"
                    )
        if self.event_persistence_callback is not None:
            try:
                self.event_persistence_callback(returned_event)
            except Exception as exc:
                with self._lock:
                    self._history_error = (
                        f"Could not persist event history to the database: {exc}"
                    )
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
