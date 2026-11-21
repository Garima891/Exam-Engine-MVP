from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable

import numpy as np

from integration.config import (
    MICROPHONE_SAMPLE_RATE_HZ,
    MICROPHONE_STARTUP_TIMEOUT_SECONDS,
)
from integration.microphone_manager import (
    MicrophoneManager,
    MicrophoneStartupTimeoutError,
)


MIN_SPEECH_SECONDS = 8.0
DEFAULT_CAPTURE_SECONDS = 15.0


def _emit(message_type: str, **payload: object) -> None:
    message = {
        "type": message_type,
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        **payload,
    }
    print(
        "VOICE_REG_JSON:" + json.dumps(
            message,
            allow_nan=False,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _watch_stop(stop_event: threading.Event) -> None:
    for line in sys.stdin:
        if line.strip().upper() in {"STOP", "Q", "A"}:
            stop_event.set()
            return
    stop_event.set()


def _embedding_to_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    embedding = np.asarray(value, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise ValueError("ECAPA returned an empty or non-finite embedding.")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("ECAPA returned a zero or invalid embedding.")
    return (embedding / norm).astype(np.float32, copy=False)


def _resample_waveform(
    waveform: np.ndarray,
    source_rate_hz: int,
    target_rate_hz: int,
) -> np.ndarray:
    if source_rate_hz == target_rate_hz:
        return waveform
    divisor = math.gcd(source_rate_hz, target_rate_hz)
    from scipy.signal import resample_poly

    return resample_poly(
        waveform,
        target_rate_hz // divisor,
        source_rate_hz // divisor,
    ).astype(np.float32, copy=False)


def capture_voice_embedding(
    duration_seconds: float = DEFAULT_CAPTURE_SECONDS,
    *,
    microphone_factory: Callable[..., Any] = MicrophoneManager,
    detector_loader: Callable[[], Any] | None = None,
    stop_event: threading.Event | None = None,
    watch_stdin: bool = True,
    startup_timeout_seconds: float = MICROPHONE_STARTUP_TIMEOUT_SECONDS,
    emit: Callable[..., None] = _emit,
) -> int:
    """Capture mono 16 kHz audio in memory, then enroll it before releasing the stream."""
    if duration_seconds <= 0.0:
        raise ValueError("Capture duration must be positive.")
    if startup_timeout_seconds <= 0.0:
        raise ValueError("Startup timeout must be positive.")
    startup_started = time.monotonic()
    stop_event = stop_event or threading.Event()
    emit("stage", stage="AUDIO_WORKER_STARTED")

    emit("stage", stage="AUDIO_MODELS_LOADING")
    try:
        if detector_loader is None:
            from integration.voice_registration_models import (
                load_cached_voice_registration_models,
            )

            detector = load_cached_voice_registration_models(
                stage_callback=lambda stage: emit("stage", stage=stage)
            )
        else:
            detector = detector_loader()
    except Exception as exc:
        emit(
            "error",
            status="AUDIO WORKER ERROR",
            error=f"{type(exc).__name__}: {exc}",
            diagnostic=traceback.format_exc(),
        )
        return 1

    if time.monotonic() - startup_started >= startup_timeout_seconds:
        emit(
            "error",
            status="MICROPHONE STARTUP TIMEOUT",
            reason=(
                "Audio models did not finish loading within "
                f"{startup_timeout_seconds:.1f} seconds."
            ),
            diagnostic="last_stage=AUDIO_MODELS_LOADING",
        )
        return 1

    emit("stage", stage="AUDIO_MODELS_READY")
    emit("status", status="AUDIO MODELS READY")
    if watch_stdin:
        threading.Thread(
            target=_watch_stop,
            args=(stop_event,),
            name="voice-registration-control",
            daemon=True,
        ).start()
    if stop_event.is_set():
        emit("status", status="VOICE CAPTURE CANCELLED")
        return 0

    microphone: Any | None = None
    microphone_open = False
    captured_blocks: list[np.ndarray] = []
    result: dict[str, Any] | None = None
    capture_error: tuple[str, str, str] | None = None

    try:
        microphone = microphone_factory(
            stage_callback=lambda stage, details: emit(
                "stage",
                stage=stage,
                **details,
            )
        )
        microphone.open()
        microphone_open = True
        if time.monotonic() - startup_started >= startup_timeout_seconds:
            raise MicrophoneStartupTimeoutError(
                "Microphone startup exceeded "
                f"{startup_timeout_seconds:.1f} seconds."
            )
        if (
            microphone.sample_rate_hz != MICROPHONE_SAMPLE_RATE_HZ
            or microphone.channels != 1
        ):
            raise ValueError(
                "Microphone configuration must be mono at "
                f"{MICROPHONE_SAMPLE_RATE_HZ} Hz."
            )
        emit(
            "status",
            status="MICROPHONE CONNECTED",
            selected_device_name=microphone.selected_device_name,
            selected_device_index=microphone.selected_device_index,
            sample_rate_hz=microphone.sample_rate_hz,
            input_sample_rate_hz=microphone.input_sample_rate_hz,
            channels=microphone.channels,
            callback_count=microphone.callback_count,
        )

        started = time.monotonic()
        captured_samples = 0
        input_sample_rate_hz = microphone.input_sample_rate_hz
        recording_started = False

        def append_audio_block(value: Any) -> None:
            nonlocal captured_samples
            block = np.asarray(value)
            if (
                block.dtype != np.float32
                or block.ndim != 1
                or block.size == 0
                or not np.all(np.isfinite(block))
            ):
                raise ValueError(
                    "Microphone returned an invalid mono float32 audio block."
                )
            captured_blocks.append(block.copy())
            captured_samples += block.size

        while not stop_event.is_set():
            elapsed = time.monotonic() - started
            if elapsed >= duration_seconds:
                break
            try:
                block = microphone.read(timeout=0.25)
            except TimeoutError:
                continue
            append_audio_block(block)
            actual_duration = captured_samples / input_sample_rate_hz
            if not recording_started:
                emit(
                    "stage",
                    stage="RECORDING_STARTED",
                    callback_count=microphone.callback_count,
                    input_sample_rate_hz=input_sample_rate_hz,
                    selected_device_name=microphone.selected_device_name,
                    selected_device_index=microphone.selected_device_index,
                )
                emit(
                    "stage",
                    stage="AUDIO_WORKER_READY",
                    callback_count=microphone.callback_count,
                    input_sample_rate_hz=input_sample_rate_hz,
                    selected_device_name=microphone.selected_device_name,
                    selected_device_index=microphone.selected_device_index,
                )
                emit(
                    "status",
                    status="AUDIO_WORKER_READY",
                    callback_count=microphone.callback_count,
                )
                recording_started = True
            emit(
                "progress",
                status="RECORDING",
                elapsed_seconds=round(actual_duration, 2),
                target_seconds=duration_seconds,
                input_rms=float(
                    np.sqrt(np.mean(block.astype(np.float64) ** 2))
                ),
            )

        microphone.stop_capture()
        while True:
            try:
                append_audio_block(microphone.read(timeout=0))
            except TimeoutError:
                break
        if not captured_blocks:
            emit(
                "error",
                status="INSUFFICIENT SPEECH",
                reason="No microphone audio blocks were captured.",
                diagnostic=(
                    f"sample_rate_hz={microphone.sample_rate_hz}, "
                    f"callbacks={microphone.callback_count}"
                ),
            )
        else:
            input_waveform = np.concatenate(captured_blocks).astype(
                np.float32,
                copy=False,
            )
            duration = input_waveform.size / input_sample_rate_hz
            if duration < MIN_SPEECH_SECONDS:
                emit(
                    "error",
                    status="INSUFFICIENT SPEECH",
                    reason=(
                        f"Captured {duration:.2f}s; at least "
                        f"{MIN_SPEECH_SECONDS:.0f}s is required."
                    ),
                    duration_seconds=round(duration, 2),
                )
            else:
                waveform = _resample_waveform(
                    input_waveform,
                    input_sample_rate_hz,
                    MICROPHONE_SAMPLE_RATE_HZ,
                )
                emit(
                    "status",
                    status="PROCESSING AUDIO",
                    duration_seconds=round(duration, 2),
                    input_sample_rate_hz=input_sample_rate_hz,
                    sample_rate_hz=MICROPHONE_SAMPLE_RATE_HZ,
                )
                try:
                    reference = detector.enroll_array(waveform)
                    embedding = _embedding_to_array(reference)
                    result = {
                        "status": "VOICE CAPTURE SUCCESSFUL",
                        "duration_seconds": round(duration, 2),
                        "validation_status": "VALID SPEECH AND ECAPA EMBEDDING",
                        "embedding_base64": base64.b64encode(
                            embedding.tobytes()
                        ).decode("ascii"),
                        "embedding_dimension": int(embedding.size),
                    }
                except ValueError as exc:
                    message = str(exc)
                    if "less than 8 seconds of detected speech" in message:
                        emit(
                            "error",
                            status="INSUFFICIENT SPEECH",
                            reason=message,
                            duration_seconds=round(duration, 2),
                        )
                    else:
                        emit(
                            "error",
                            status="VOICE CAPTURE ERROR",
                            reason=f"{type(exc).__name__}: {exc}",
                            diagnostic=traceback.format_exc(),
                        )
                except Exception as exc:
                    emit(
                        "error",
                        status="VOICE CAPTURE ERROR",
                        reason=f"{type(exc).__name__}: {exc}",
                        diagnostic=traceback.format_exc(),
                    )
    except MicrophoneStartupTimeoutError as exc:
        capture_error = (
            f"{type(exc).__name__}: {exc}",
            traceback.format_exc(),
            "MICROPHONE STARTUP TIMEOUT",
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        dropped_blocks = getattr(exc, "dropped_blocks", None)
        if isinstance(dropped_blocks, int):
            detail = f"{detail} Dropped microphone blocks: {dropped_blocks}."
        capture_error = (
            detail,
            traceback.format_exc(),
            "MICROPHONE ERROR",
        )
    finally:
        if microphone_open and microphone is not None:
            try:
                microphone.release()
            except Exception as exc:
                capture_error = (
                    f"{type(exc).__name__}: {exc}",
                    traceback.format_exc(),
                    "MICROPHONE ERROR",
                )

    if capture_error is not None:
        error, diagnostic, status = capture_error
        emit("error", status=status, error=error, diagnostic=diagnostic)
        return 1
    if result is not None:
        emit(
            "complete",
            status=result["status"],
            duration_seconds=result["duration_seconds"],
            validation_status=result["validation_status"],
            embedding_base64=result["embedding_base64"],
            embedding_dimension=result["embedding_dimension"],
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="In-memory voice registration worker.")
    parser.add_argument("--duration", type=float, default=DEFAULT_CAPTURE_SECONDS)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=MICROPHONE_STARTUP_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    if args.duration < MIN_SPEECH_SECONDS:
        parser.error(
            f"--duration must be at least {MIN_SPEECH_SECONDS:.0f} seconds."
        )
    return capture_voice_embedding(
        args.duration,
        startup_timeout_seconds=args.startup_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
