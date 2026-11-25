from __future__ import annotations

import importlib
import math
import queue
from collections.abc import Callable
from threading import Event, Lock
from time import monotonic
from typing import Any

import numpy as np

from integration.config import (
    MICROPHONE_BLOCK_SIZE,
    MICROPHONE_CHANNELS,
    MICROPHONE_CALLBACK_TIMEOUT_SECONDS,
    MICROPHONE_DEVICE_INDEX,
    MICROPHONE_SAMPLE_RATE_HZ,
)


_WAKE_UP = object()


class MicrophoneError(RuntimeError):
    """Base error for microphone capture failures."""


class MicrophoneStreamError(MicrophoneError):
    """The input stream reported an error or stopped unexpectedly."""


class MicrophoneStartupTimeoutError(MicrophoneError):
    """The input stream did not deliver a valid block during startup."""


class MicrophoneQueueOverflowError(MicrophoneStreamError):
    """A queue overrun created an audio discontinuity."""

    def __init__(
        self,
        message: str = "Microphone queue overflowed; stale audio was discarded.",
        *,
        dropped_blocks: int = 0,
    ) -> None:
        super().__init__(message)
        self.dropped_blocks = dropped_blocks


class MicrophoneManager:
    """Owns a single sounddevice input stream and queues copied audio blocks."""

    def __init__(
        self,
        sample_rate_hz: int = MICROPHONE_SAMPLE_RATE_HZ,
        channels: int = MICROPHONE_CHANNELS,
        block_size: int = MICROPHONE_BLOCK_SIZE,
        device_index: int | None = MICROPHONE_DEVICE_INDEX,
        queue_capacity: int = 4,
        callback_timeout_seconds: float = MICROPHONE_CALLBACK_TIMEOUT_SECONDS,
        *,
        input_stream_factory: Callable[..., Any] | None = None,
        sounddevice_module: Any | None = None,
        stage_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        if channels != 1:
            raise ValueError("Microphone capture supports mono audio only.")
        if block_size < 0:
            raise ValueError("Block size must not be negative.")
        if queue_capacity <= 0:
            raise ValueError("Queue capacity must be a positive integer.")
        if callback_timeout_seconds <= 0.0:
            raise ValueError("Callback timeout must be a positive number.")
        if device_index is not None and device_index < 0:
            raise ValueError("Device index must not be negative.")

        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.block_size = block_size
        self.device_index = device_index
        self.queue_capacity = queue_capacity
        self.callback_timeout_seconds = callback_timeout_seconds
        self._input_stream_factory = input_stream_factory
        self._sounddevice_module = sounddevice_module
        self._stage_callback = stage_callback
        self._stream: Any | None = None
        self._blocks: queue.Queue[Any] = queue.Queue(maxsize=queue_capacity)
        self._state_lock = Lock()
        self._callback_received = Event()
        self._stream_error: MicrophoneStreamError | None = None
        self._closing = False
        self._capture_stopped = False
        self._last_input_rms = 0.0
        self._last_input_min: float | None = None
        self._last_input_max: float | None = None
        self._callback_count = 0
        self._last_callback_monotonic: float | None = None
        self._selected_device_index: int | None = self.device_index
        self._selected_device_name: str | None = None
        self._selected_device_info: dict[str, Any] = {}
        self.input_sample_rate_hz = self.sample_rate_hz
        self.input_block_size = self.block_size
        self._generation = 0
        self._read_generation = 0
        self._dropped_blocks = 0
        self._reported_dropped_blocks = 0
        self._pending_block: np.ndarray | None = None

    def open(self) -> "MicrophoneManager":
        if self._stream is not None:
            raise RuntimeError("Microphone is already open.")

        with self._state_lock:
            self._stream_error = None
            self._closing = False
            self._capture_stopped = False
            self._last_input_rms = 0.0
            self._last_input_min = None
            self._last_input_max = None
            self._callback_count = 0
            self._last_callback_monotonic = None
            self._selected_device_index = self.device_index
            self._selected_device_name = None
            self._selected_device_info = {}
            self.input_sample_rate_hz = self.sample_rate_hz
            self.input_block_size = self.block_size
            self._generation = 0
            self._read_generation = 0
            self._dropped_blocks = 0
            self._reported_dropped_blocks = 0
            self._pending_block = None
            self._callback_received.clear()
        self._blocks = queue.Queue(maxsize=self.queue_capacity)

        try:
            self._notify_stage("AUDIO_DEVICE_SELECTING")
            self._resolve_selected_device()
            self._select_input_sample_rate()
            self._notify_stage(
                "AUDIO_STREAM_OPENING",
                input_sample_rate_hz=self.input_sample_rate_hz,
                target_sample_rate_hz=self.sample_rate_hz,
                channels=self.channels,
            )
            stream = self._get_input_stream_factory()(
                samplerate=self.input_sample_rate_hz,
                channels=self.channels,
                blocksize=self.input_block_size,
                device=self._selected_device_index,
                dtype="float32",
                callback=self._audio_callback,
                finished_callback=self._finished_callback,
            )
            self._stream = stream
            self._notify_stage("AUDIO_STREAM_CREATED")
            self._notify_stage("AUDIO_STREAM_STARTING")
            stream.start()
            self._notify_stage("AUDIO_STREAM_OPENED")
            if not self._callback_received.wait(self.callback_timeout_seconds):
                stream_error = self._get_stream_error()
                if stream_error is not None:
                    raise stream_error
                raise MicrophoneStartupTimeoutError(
                    "Input stream start returned, but no valid audio callback "
                    f"arrived within {self.callback_timeout_seconds:.1f} seconds."
                )
            stream_error = self._get_stream_error()
            if stream_error is not None:
                raise stream_error
            self._notify_stage(
                "AUDIO_CALLBACK_STARTED",
                callback_count=self.callback_count,
                input_sample_rate_hz=self.input_sample_rate_hz,
                device_name=self.selected_device_name,
                device_index=self.selected_device_index,
            )
        except MicrophoneError as exc:
            try:
                self.release()
            except Exception as cleanup_error:
                if isinstance(exc, MicrophoneStartupTimeoutError):
                    raise MicrophoneStartupTimeoutError(
                        f"{exc}; microphone cleanup failed: {cleanup_error}"
                    ) from exc
                raise MicrophoneError(
                    f"{exc}; microphone cleanup failed: {cleanup_error}"
                ) from exc
            raise
        except Exception as exc:
            try:
                self.release()
            except Exception as cleanup_error:
                raise MicrophoneError(
                    f"Could not open microphone at device "
                    f"{self.device_index}; cleanup failed: {cleanup_error}"
                ) from exc
            raise MicrophoneError(
                f"Could not open microphone at device "
                f"{self._selected_device_index}: {type(exc).__name__}: {exc}"
            ) from exc

        return self

    def read(self, timeout: float | None = None) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Microphone is not open.")
        if timeout is not None and timeout < 0:
            raise ValueError("Read timeout must not be negative.")

        error = self._get_stream_error()
        if error is not None:
            self._raise_after_release(error)

        with self._state_lock:
            pending_block = self._pending_block
            self._pending_block = None
        if pending_block is not None:
            return pending_block

        try:
            queued_block = self._blocks.get(timeout=timeout)
        except queue.Empty:
            error = self._get_stream_error()
            if error is not None:
                self._raise_after_release(error)
            if self._stream is None:
                raise RuntimeError("Microphone is not open.")
            raise TimeoutError("Timed out waiting for a microphone audio block.")

        if queued_block is _WAKE_UP:
            error = self._get_stream_error()
            if error is not None:
                self._raise_after_release(error)
            raise RuntimeError("Microphone was closed while waiting for audio.")

        generation, block = queued_block
        with self._state_lock:
            if generation != self._read_generation:
                self._read_generation = generation
                dropped_blocks = (
                    self._dropped_blocks - self._reported_dropped_blocks
                )
                self._reported_dropped_blocks = self._dropped_blocks
                self._pending_block = block
                raise MicrophoneQueueOverflowError(
                    dropped_blocks=dropped_blocks
                )

        return block

    def release(self) -> None:
        with self._state_lock:
            self._closing = True
            stream, self._stream = self._stream, None
            self._last_input_rms = 0.0
            self._last_input_min = None
            self._last_input_max = None
            self._callback_count = 0
            self._last_callback_monotonic = None
            self._pending_block = None

        self._wake_reader()
        if stream is None:
            return

        stop_error: Exception | None = None
        if getattr(stream, "active", False):
            try:
                stream.stop()
            except Exception as exc:
                stop_error = exc

        try:
            stream.close()
        except Exception as close_error:
            if stop_error is not None:
                raise MicrophoneError(
                    f"Stopping and closing the microphone stream failed: "
                    f"{stop_error}; {close_error}"
                ) from close_error
            raise

        if stop_error is not None:
            raise MicrophoneError(
                f"Could not stop the microphone stream: {stop_error}"
            ) from stop_error

    def stop_capture(self) -> None:
        """Stop callbacks while retaining the stream until buffered audio is processed."""
        with self._state_lock:
            stream = self._stream
            if stream is None:
                raise RuntimeError("Microphone is not open.")
            if self._capture_stopped:
                return
            self._capture_stopped = True
            self._closing = True

        if not getattr(stream, "active", False):
            return
        try:
            stream.stop()
        except Exception as exc:
            raise MicrophoneStreamError(
                f"Could not stop microphone capture: {exc}"
            ) from exc

    close = release

    def __enter__(self) -> "MicrophoneManager":
        return self.open()

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        self.release()
        return False

    def _get_input_stream_factory(self) -> Callable[..., Any]:
        if self._input_stream_factory is not None:
            return self._input_stream_factory
        if self._sounddevice_module is None:
            self._sounddevice_module = importlib.import_module("sounddevice")
        return self._sounddevice_module.InputStream

    def _resolve_selected_device(self) -> None:
        if self._sounddevice_module is None:
            if self._input_stream_factory is not None:
                return
            self._sounddevice_module = importlib.import_module("sounddevice")

        module = self._sounddevice_module
        selected_index = self.device_index
        if selected_index is None:
            default_device = getattr(
                getattr(module, "default", None),
                "device",
                None,
            )
            if isinstance(default_device, (tuple, list)) and default_device:
                value = default_device[0]
                if isinstance(value, int) and not isinstance(value, bool):
                    if value >= 0:
                        selected_index = value
        if selected_index is not None and selected_index < 0:
            selected_index = None
        if selected_index is None:
            device_info = module.query_devices(kind="input")
            all_devices = module.query_devices()
            if isinstance(all_devices, (list, tuple)):
                for index, candidate in enumerate(all_devices):
                    channels = (
                        candidate.get("max_input_channels")
                        if isinstance(candidate, dict)
                        else None
                    )
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("name") == device_info.get("name")
                        and candidate.get("hostapi") == device_info.get("hostapi")
                        and isinstance(channels, (int, float))
                        and channels > 0
                    ):
                        selected_index = index
                        break
        else:
            device_info = module.query_devices(selected_index, "input")
        if not isinstance(device_info, dict) or not isinstance(
            device_info.get("name"),
            str,
        ):
            raise MicrophoneError("Could not identify the selected input device.")
        input_channels = device_info.get("max_input_channels")
        if input_channels is not None and (
            not isinstance(input_channels, (int, float))
            or input_channels < 1
        ):
            raise MicrophoneError(
                f"Selected input device {device_info['name']!r} has no input channels."
            )
        with self._state_lock:
            self._selected_device_index = selected_index
            self._selected_device_name = device_info["name"]
            self._selected_device_info = dict(device_info)
        self._notify_stage(
            "AUDIO_DEVICE_SELECTED",
            device_index=selected_index,
            device_name=device_info["name"],
            max_input_channels=input_channels,
            default_sample_rate_hz=device_info.get("default_samplerate"),
        )

    def _select_input_sample_rate(self) -> None:
        module = self._sounddevice_module
        check_settings = getattr(module, "check_input_settings", None)
        if not callable(check_settings):
            return

        device_index = self.selected_device_index
        try:
            check_settings(
                device=device_index,
                channels=self.channels,
                dtype="float32",
                samplerate=self.sample_rate_hz,
            )
            self.input_sample_rate_hz = self.sample_rate_hz
        except Exception as preferred_error:
            default_sample_rate = self._selected_device_info.get(
                "default_samplerate"
            )
            if (
                not isinstance(default_sample_rate, (int, float))
                or not math.isfinite(float(default_sample_rate))
                or default_sample_rate <= 0
            ):
                raise MicrophoneError(
                    f"Device {self.selected_device_name!r} rejected "
                    f"{self.sample_rate_hz} Hz mono float32 input and did not "
                    "report a usable default sample rate. "
                    f"Device error: {type(preferred_error).__name__}: "
                    f"{preferred_error}"
                ) from preferred_error

            native_sample_rate = int(round(float(default_sample_rate)))
            if native_sample_rate == self.sample_rate_hz:
                raise MicrophoneError(
                    f"Device {self.selected_device_name!r} rejected "
                    f"{self.sample_rate_hz} Hz mono float32 input. "
                    f"Device error: {type(preferred_error).__name__}: "
                    f"{preferred_error}"
                ) from preferred_error
            try:
                check_settings(
                    device=device_index,
                    channels=self.channels,
                    dtype="float32",
                    samplerate=native_sample_rate,
                )
            except Exception as native_error:
                raise MicrophoneError(
                    f"Device {self.selected_device_name!r} rejected both "
                    f"{self.sample_rate_hz} Hz and its reported native rate "
                    f"{native_sample_rate} Hz for mono float32 input. "
                    f"Preferred-rate error: {type(preferred_error).__name__}: "
                    f"{preferred_error}; native-rate error: "
                    f"{type(native_error).__name__}: {native_error}"
                ) from native_error
            self.input_sample_rate_hz = native_sample_rate

        if self.block_size:
            self.input_block_size = max(
                1,
                round(
                    self.block_size
                    * self.input_sample_rate_hz
                    / self.sample_rate_hz
                ),
            )
        self._notify_stage(
            "AUDIO_INPUT_FORMAT_SELECTED",
            input_sample_rate_hz=self.input_sample_rate_hz,
            output_sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            resampling_required=(
                self.input_sample_rate_hz != self.sample_rate_hz
            ),
        )

    def _notify_stage(self, stage: str, **details: Any) -> None:
        if self._stage_callback is not None:
            self._stage_callback(stage, details)

    def _audio_callback(
        self,
        indata: Any,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        del time_info
        with self._state_lock:
            if self._closing or self._stream_error is not None:
                return

        if status:
            self._record_error(
                MicrophoneStreamError(
                    f"Microphone input stream reported status: {status}"
                )
            )
            return

        try:
            block = self._validate_block(indata, frames)
        except ValueError as exc:
            self._record_error(
                MicrophoneStreamError(f"Invalid microphone audio block: {exc}")
            )
            return

        try:
            block_rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            block_min = float(np.min(block))
            block_max = float(np.max(block))
        except Exception as exc:
            self._record_error(
                MicrophoneStreamError(
                    f"Microphone callback processing failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
            return
        with self._state_lock:
            if self._closing:
                return
            self._last_input_rms = block_rms
            self._last_input_min = block_min
            self._last_input_max = block_max
            self._callback_count += 1
            self._last_callback_monotonic = monotonic()
            queued_block = (self._generation, block)
            try:
                self._blocks.put_nowait(queued_block)
                self._callback_received.set()
                return
            except queue.Full:
                dropped_blocks = 0
                while True:
                    try:
                        stale_block = self._blocks.get_nowait()
                    except queue.Empty:
                        break
                    if stale_block is not _WAKE_UP:
                        dropped_blocks += 1

                self._generation += 1
                self._dropped_blocks += dropped_blocks
                self._blocks.put_nowait((self._generation, block))
                self._callback_received.set()

    def _finished_callback(self) -> None:
        with self._state_lock:
            closing = self._closing
        if not closing:
            self._record_error(
                MicrophoneStreamError(
                    "Microphone input stream stopped unexpectedly."
                )
            )

    def _record_error(self, error: MicrophoneStreamError) -> None:
        with self._state_lock:
            if self._closing or self._stream_error is not None:
                return
            self._stream_error = error
            self._last_input_rms = 0.0
            self._last_input_min = None
            self._last_input_max = None
            self._callback_count = 0
            self._last_callback_monotonic = None
        self._callback_received.set()
        self._wake_reader()

    @property
    def last_input_rms(self) -> float:
        with self._state_lock:
            return self._last_input_rms

    @property
    def dropped_blocks(self) -> int:
        with self._state_lock:
            return self._dropped_blocks

    @property
    def callback_count(self) -> int:
        with self._state_lock:
            return self._callback_count

    @property
    def queue_size(self) -> int:
        return self._blocks.qsize()

    @property
    def last_input_min(self) -> float | None:
        with self._state_lock:
            return self._last_input_min

    @property
    def last_input_max(self) -> float | None:
        with self._state_lock:
            return self._last_input_max

    @property
    def seconds_since_last_callback(self) -> float | None:
        with self._state_lock:
            last_callback = self._last_callback_monotonic
        if last_callback is None:
            return None
        return max(0.0, monotonic() - last_callback)

    @property
    def selected_device_index(self) -> int | None:
        with self._state_lock:
            return self._selected_device_index

    @property
    def selected_device_name(self) -> str | None:
        with self._state_lock:
            return self._selected_device_name

    @property
    def selected_device_info(self) -> dict[str, Any]:
        with self._state_lock:
            return dict(self._selected_device_info)

    def _get_stream_error(self) -> MicrophoneStreamError | None:
        with self._state_lock:
            return self._stream_error

    def _wake_reader(self) -> None:
        try:
            self._blocks.put_nowait(_WAKE_UP)
        except queue.Full:
            pass

    def _raise_after_release(self, error: MicrophoneStreamError) -> None:
        try:
            self.release()
        except Exception as cleanup_error:
            raise MicrophoneError(
                f"{error}; microphone cleanup failed: {cleanup_error}"
            ) from error
        raise error

    @staticmethod
    def _validate_block(indata: Any, frames: int) -> np.ndarray:
        if not isinstance(indata, np.ndarray):
            raise ValueError("expected a NumPy array.")
        if indata.dtype != np.float32:
            raise ValueError("expected float32 samples.")
        if (
            indata.ndim != 2
            or indata.shape[1] != 1
            or indata.shape[0] != frames
            or frames <= 0
        ):
            raise ValueError("expected a non-empty mono block matching frame count.")
        if not np.all(np.isfinite(indata)):
            raise ValueError("samples must be finite.")
        return indata[:, 0].copy()
