from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from integration.audio_live_test import load_cached_detector
from integration.audio_monitor import AudioMonitor, InsufficientAudioError
from integration.config import AUDIO_ANALYSIS_WINDOW_SAMPLES
from integration.microphone_manager import (
    MicrophoneError,
    MicrophoneManager,
)
from integration.module_adapters.audio import AudioAdapter


MESSAGE_PREFIX = "PROCTOR_JSON:"
MICROPHONE_QUEUE_CAPACITY = 4
RECOVERY_DELAYS_SECONDS = (0.5, 1.0, 2.0)
MICROPHONE_CALLBACK_STALL_TIMEOUT_SECONDS = 2.0
_output_lock = threading.Lock()


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _emit(message_type: str, **payload: Any) -> None:
    message = {"type": message_type, "timestamp": _timestamp(), **payload}
    encoded = json.dumps(message, allow_nan=False, separators=(",", ":"))
    with _output_lock:
        print(f"{MESSAGE_PREFIX}{encoded}", flush=True)


def _watch_parent(stop_event: threading.Event) -> None:
    for line in sys.stdin:
        if line.strip().upper() == "STOP":
            stop_event.set()
            return
    stop_event.set()


def _send_telemetry(
    microphone: MicrophoneManager,
    monitor: AudioMonitor,
    stop_event: threading.Event,
) -> None:
    reported_discontinuities = 0
    while not stop_event.wait(0.25):
        if monitor.discontinuities > reported_discontinuities:
            reported_discontinuities = monitor.discontinuities
            _emit(
                "warning",
                status="AUDIO GAP RECOVERED",
                discontinuities=reported_discontinuities,
                dropped_blocks=monitor.dropped_blocks,
            )
        _emit(
            "telemetry",
            status="MICROPHONE ON",
            input_rms=microphone.last_input_rms,
            input_min=microphone.last_input_min,
            input_max=microphone.last_input_max,
            callback_count=microphone.callback_count,
            callback_age_seconds=microphone.seconds_since_last_callback,
            selected_device_index=microphone.selected_device_index,
            selected_device_name=microphone.selected_device_name,
            queue_size=microphone.queue_size,
            queue_capacity=microphone.queue_capacity,
            blocks_received=monitor.blocks_received,
            samples_received=monitor.samples_received,
            pending_samples=monitor.pending_samples,
            discontinuities=monitor.discontinuities,
            dropped_blocks=monitor.dropped_blocks,
            last_window_rms=monitor.last_window_rms,
            last_window_peak=monitor.last_window_peak,
            last_window_min=monitor.last_window_min,
            last_window_max=monitor.last_window_max,
            last_window_age_seconds=(
                None
                if monitor.last_window_completed_monotonic is None
                else max(
                    0.0,
                    time.monotonic()
                    - monitor.last_window_completed_monotonic,
                )
            ),
        )


def _capture_audio(
    adapter: AudioAdapter,
    stop_event: threading.Event,
    *,
    initial_window_id: int = 0,
) -> int:
    window_id = initial_window_id
    recovery_attempts = 0

    while not stop_event.is_set():
        microphone = MicrophoneManager(
            queue_capacity=MICROPHONE_QUEUE_CAPACITY
        )
        monitor = AudioMonitor(microphone, adapter)
        try:
            with monitor:
                _emit(
                    "status",
                    status="MICROPHONE ON",
                    selected_device_index=microphone.selected_device_index,
                    selected_device_name=microphone.selected_device_name,
                    sample_rate_hz=microphone.sample_rate_hz,
                    block_size=microphone.block_size,
                    queue_capacity=microphone.queue_capacity,
                )
                telemetry_stop = threading.Event()
                telemetry_thread = threading.Thread(
                    target=_send_telemetry,
                    args=(microphone, monitor, telemetry_stop),
                    name="audio-worker-telemetry",
                    daemon=True,
                )
                telemetry_thread.start()
                try:
                    while not stop_event.is_set():
                        try:
                            result = monitor.analyze_next(timeout=0.25)
                        except InsufficientAudioError:
                            callback_age = (
                                microphone.seconds_since_last_callback
                            )
                            if (
                                microphone.callback_count > 0
                                and callback_age is not None
                                and callback_age
                                > MICROPHONE_CALLBACK_STALL_TIMEOUT_SECONDS
                            ):
                                raise MicrophoneError(
                                    "Microphone callbacks stopped for "
                                    f"{callback_age:.1f} seconds."
                                )
                            continue
                        window_id += 1
                        recovery_attempts = 0
                        _emit(
                            "audio_result",
                            window_id=window_id,
                            window_start_sample=monitor.last_window_start_sample,
                            input_rms=microphone.last_input_rms,
                            input_min=microphone.last_input_min,
                            input_max=microphone.last_input_max,
                            callback_count=microphone.callback_count,
                            callback_age_seconds=(
                                microphone.seconds_since_last_callback
                            ),
                            selected_device_index=microphone.selected_device_index,
                            selected_device_name=microphone.selected_device_name,
                            queue_size=microphone.queue_size,
                            queue_capacity=microphone.queue_capacity,
                            discontinuities=monitor.discontinuities,
                            dropped_blocks=monitor.dropped_blocks,
                            window_rms=monitor.last_window_rms,
                            window_peak=monitor.last_window_peak,
                            window_min=monitor.last_window_min,
                            window_max=monitor.last_window_max,
                            last_window_age_seconds=0.0,
                            result=result,
                        )
                finally:
                    telemetry_stop.set()
                    telemetry_thread.join(timeout=1.0)
        except MicrophoneError as exc:
            if stop_event.is_set():
                break
            if recovery_attempts >= len(RECOVERY_DELAYS_SECONDS):
                raise
            delay = RECOVERY_DELAYS_SECONDS[recovery_attempts]
            recovery_attempts += 1
            _emit(
                "status",
                status="MICROPHONE RECOVERING",
                attempt=recovery_attempts,
                error=f"{type(exc).__name__}: {exc}",
            )
            if stop_event.wait(delay):
                break
            continue
        break

    return window_id


def run_worker(
    audio_reference: Path | None = None,
    speaker_reference_base64: str | None = None,
) -> int:
    stop_event = threading.Event()

    try:
        _emit("status", status="LOADING CACHED AUDIO MODELS")
        detector = load_cached_detector()
        reference_embedding = None
        if audio_reference is not None:
            if not audio_reference.is_file():
                raise FileNotFoundError(
                    f"Audio reference WAV not found: {audio_reference}"
                )
            reference_embedding = detector.enroll(str(audio_reference))
        elif speaker_reference_base64 is not None:
            reference_embedding = np.frombuffer(
                base64.b64decode(speaker_reference_base64, validate=True),
                dtype=np.float32,
            ).copy()
            if reference_embedding.size == 0:
                raise ValueError("Speaker reference embedding is empty.")
            norm = float(np.linalg.norm(reference_embedding))
            if norm == 0.0:
                raise ValueError("Speaker reference embedding is zero-valued.")
            reference_embedding = reference_embedding / norm
        adapter = AudioAdapter(
            speaker_reference_embedding=reference_embedding,
            detector_loader=lambda: detector,
        )
        adapter.initialize()
        warmup = adapter.analyze(
            np.zeros(AUDIO_ANALYSIS_WINDOW_SAMPLES, dtype=np.float32)
        )
        if warmup.get("module") != "audio_detection":
            raise RuntimeError("Audio model warm-up returned an invalid result.")
        adapter.reset()
        _emit(
            "status",
            status="AUDIO MODELS READY",
            speaker_reference=(
                "candidate-linked WAV loaded in memory"
                if audio_reference is not None
                else "candidate-linked embedding loaded in memory"
                if speaker_reference_base64 is not None
                else "not configured; speaker verification unavailable"
            ),
            warmup="silence window discarded; no event was counted",
        )

        threading.Thread(
            target=_watch_parent,
            args=(stop_event,),
            name="audio-worker-control",
            daemon=True,
        ).start()

        if stop_event.is_set():
            _emit("status", status="MICROPHONE OFF")
            return 0

        _capture_audio(adapter, stop_event)

        _emit("status", status="MICROPHONE OFF")
        return 0
    except Exception as exc:
        status = (
            "MICROPHONE ERROR"
            if isinstance(exc, MicrophoneError)
            else "AUDIO WORKER ERROR"
        )
        _emit(
            "error",
            status=status,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live demo audio worker.")
    parser.add_argument("--audio-reference", type=Path)
    parser.add_argument("--speaker-reference-base64")
    args = parser.parse_args(argv)
    if args.audio_reference is not None and not args.audio_reference.is_file():
        parser.error(f"Audio reference does not exist: {args.audio_reference}")
    return run_worker(
        None if args.audio_reference is None else args.audio_reference.resolve(),
        args.speaker_reference_base64,
    )


if __name__ == "__main__":
    raise SystemExit(main())
