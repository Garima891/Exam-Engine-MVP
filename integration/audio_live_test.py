"""Run with ``python -m integration.audio_live_test`` for manual audio testing."""

from __future__ import annotations

import importlib
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from integration.audio_monitor import AudioMonitor, InsufficientAudioError
from integration.config import AUDIO_ANALYSIS_WINDOW_SAMPLES
from integration.microphone_manager import (
    MicrophoneManager,
    MicrophoneStreamError,
)
from integration.module_adapters.audio import AudioAdapter


YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
MICROPHONE_OFF = "MICROPHONE OFF"
MICROPHONE_STARTING = "MICROPHONE STARTING"
MICROPHONE_ON = "MICROPHONE ON – LISTENING"
MICROPHONE_ERROR = "MICROPHONE ERROR"
MICROPHONE_STOPPED = "MICROPHONE STOPPED"


@dataclass(frozen=True)
class AudioTestScenario:
    name: str
    instruction: str
    kind: str


SCENARIOS = (
    AudioTestScenario(
        "Initial silence",
        "Remain quiet until NEXT.",
        "silence",
    ),
    AudioTestScenario(
        "Normal speech",
        "Speak a sentence naturally for at least 4 seconds.",
        "speech",
    ),
    AudioTestScenario(
        "Soft speech",
        "Speak quietly for at least 4 seconds.",
        "speech",
    ),
    AudioTestScenario(
        "Loud speech",
        "Speak loudly without shouting for at least 4 seconds.",
        "speech",
    ),
    AudioTestScenario(
        "Brief utterance",
        "Say one brief utterance of about 0.4 seconds, then wait.",
        "brief",
    ),
    AudioTestScenario(
        "Continuous speech",
        "Speak continuously for at least 5 seconds.",
        "continuous",
    ),
    AudioTestScenario(
        "Post-speech silence",
        "Stop speaking and remain quiet.",
        "silence",
    ),
    AudioTestScenario(
        "Background human voice",
        "Have another person speak for at least 4 seconds.",
        "speech",
    ),
    AudioTestScenario(
        "Multiple speakers",
        "Have two people alternate short turns for at least 6 seconds.",
        "multiple_speakers",
    ),
    AudioTestScenario(
        "Phone notification or ringing",
        "Play a phone notification or ringing sound near the microphone.",
        "phone",
    ),
    AudioTestScenario(
        "Fan or environmental noise",
        "Introduce ordinary fan or environmental noise for several seconds.",
        "environmental_noise",
    ),
    AudioTestScenario(
        "Microphone interruption",
        "Disable or disconnect the input device; wait for MICROPHONE ERROR.",
        "interruption",
    ),
)


def load_cached_detector() -> Any:
    """Load the standalone detector only when every required model is cached."""
    from voice_vyor_Ai.ecapa_model_loader import (
        ECAPA_MODEL_FILES,
        ECAPA_MODEL_ID,
        ECAPA_MODEL_REVISION,
    )

    hf_constants = importlib.import_module("huggingface_hub.constants")
    repository_dir = (
        Path(hf_constants.HF_HUB_CACHE)
        / f"models--{ECAPA_MODEL_ID.replace('/', '--')}"
        / "snapshots"
    )
    ecapa_snapshot = repository_dir / ECAPA_MODEL_REVISION
    missing = [
        filename
        for filename in ECAPA_MODEL_FILES
        if not (ecapa_snapshot / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cached ECAPA snapshot is incomplete; no model download was attempted. "
            f"Missing: {', '.join(missing)}"
        )

    tfhub_cache = Path(
        os.environ.get(
            "TFHUB_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "tfhub_modules"),
        )
    )
    yamnet_candidates = sorted(
        path.parent.parent
        for path in tfhub_cache.glob("*/assets/yamnet_class_map.csv")
        if (path.parent.parent / "saved_model.pb").is_file()
    )
    if not yamnet_candidates:
        raise FileNotFoundError(
            f"Cached YAMNet model was not found under {tfhub_cache}; "
            "no model download was attempted."
        )

    huggingface_hub = importlib.import_module("huggingface_hub")
    tensorflow_hub = importlib.import_module("tensorflow_hub")
    original_snapshot_download = huggingface_hub.snapshot_download
    original_hf_hub_download = huggingface_hub.hf_hub_download
    original_hub_load = tensorflow_hub.load

    def cached_snapshot_download(
        repo_id: str,
        revision: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if repo_id != ECAPA_MODEL_ID or revision != ECAPA_MODEL_REVISION:
            raise RuntimeError(
                "The live test permits only the pinned cached ECAPA snapshot."
            )
        return str(ecapa_snapshot)

    def cached_hub_load(handle: str, *args: Any, **kwargs: Any) -> Any:
        if handle != YAMNET_URL:
            raise RuntimeError(
                "The live test permits only the cached YAMNet model."
            )
        return original_hub_load(str(yamnet_candidates[0]), *args, **kwargs)

    def reject_hf_download(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("The live test does not download Hugging Face files.")

    huggingface_hub.snapshot_download = cached_snapshot_download
    huggingface_hub.hf_hub_download = reject_hf_download
    tensorflow_hub.load = cached_hub_load
    try:
        return importlib.import_module("voice_vyor_Ai.detector")
    finally:
        huggingface_hub.snapshot_download = original_snapshot_download
        huggingface_hub.hf_hub_download = original_hf_hub_download
        tensorflow_hub.load = original_hub_load


def evaluate_scenario(
    scenario: AudioTestScenario,
    observations: list[dict[str, Any]],
    *,
    speaker_reference_configured: bool,
) -> tuple[str, str]:
    if scenario.kind == "interruption":
        return (
            "INCONCLUSIVE",
            "No stream interruption was observed during this scenario.",
        )
    if not observations:
        return "INCONCLUSIVE", "No complete eligible analysis windows were observed."

    speech_detected = [
        bool(item["vad"]["speech_detected"])
        for item in observations
    ]
    phone_detected = [
        item["phone_sound"].get("detected") is True
        for item in observations
    ]

    if scenario.kind == "silence":
        if any(speech_detected):
            return "FAIL", "VAD reported speech during the silence scenario."
        return "PASS", "No speech was reported in the evaluated windows."

    if scenario.kind == "speech":
        if any(speech_detected):
            return "PASS", "VAD reported speech in at least one evaluated window."
        return "FAIL", "VAD did not report speech in the evaluated windows."

    if scenario.kind == "brief":
        speech_observations = [
            item
            for item in observations
            if item["vad"]["speech_detected"]
        ]
        if not speech_observations:
            return "INCONCLUSIVE", "No speech was reported for the brief utterance."
        if all(
            item["speaker"]["status"] == "insufficient_speech"
            for item in speech_observations
        ):
            return (
                "PASS",
                "Speech was detected but remained below the speaker-comparison minimum.",
            )
        return (
            "FAIL",
            "At least one brief-utterance window was eligible for speaker comparison.",
        )

    if scenario.kind == "continuous":
        speech_windows = sum(speech_detected)
        if speech_windows == 0:
            return "FAIL", "VAD did not report speech during continuous speech."
        if not speaker_reference_configured:
            return (
                "INCONCLUSIVE",
                "Speech was reported, but no in-memory speaker reference is available.",
            )
        if speech_windows < 2:
            return (
                "INCONCLUSIVE",
                "Speech was reported in fewer than two complete windows.",
            )
        if any(item["speaker"]["status"] == "compared" for item in observations):
            return "PASS", "Speech and speaker comparisons were produced."
        return (
            "INCONCLUSIVE",
            "Speech was reported, but no speaker comparison was produced.",
        )

    if scenario.kind == "multiple_speakers":
        if not any(speech_detected):
            return "FAIL", "VAD did not report speech during alternating turns."
        return (
            "INCONCLUSIVE",
            "VAD reported speech, but this interface cannot verify speaker alternation.",
        )

    if scenario.kind == "phone":
        if any(phone_detected):
            return "PASS", "YAMNet reported a phone-sound score above its existing threshold."
        return "FAIL", "No phone-sound detection was reported."

    if scenario.kind == "environmental_noise":
        if any(speech_detected) or any(phone_detected):
            return (
                "FAIL",
                "Speech or phone-sound detection occurred during the noise scenario.",
            )
        return (
            "PASS",
            "No speech or phone-sound detection was reported in the evaluated windows.",
        )

    return "INCONCLUSIVE", "No evaluation rule is defined for this scenario."


def _speaker_label(speaker_result: dict[str, Any]) -> str:
    status = speaker_result.get("status")
    comparison = speaker_result.get("comparison")
    if status in {"no_speech", "insufficient_speech"}:
        return "INSUFFICIENT SPEECH"
    if status == "reference_not_configured":
        return "REVIEW REQUIRED"
    if comparison == "high_similarity":
        return "MATCHED"
    if comparison == "low_similarity":
        return "UNMATCHED"
    if comparison == "ambiguous_similarity":
        return "REVIEW REQUIRED"
    return "REVIEW REQUIRED"


class InteractiveAudioTest:
    """Manual CLI test flow; microphone creation is deferred until START TEST."""

    def __init__(
        self,
        *,
        speaker_reference_embedding: Any | None = None,
        adapter_factory: Callable[[], Any] | None = None,
        microphone_factory: Callable[[], Any] = MicrophoneManager,
        monitor_factory: Callable[[Any, Any], Any] = AudioMonitor,
        output: Callable[[str], None] = print,
    ) -> None:
        self._speaker_reference_embedding = speaker_reference_embedding
        self._adapter_factory = adapter_factory or self._default_adapter_factory
        self._microphone_factory = microphone_factory
        self._monitor_factory = monitor_factory
        self._output = output
        self._status = MICROPHONE_OFF
        self._adapter: Any | None = None
        self._monitor: Any | None = None
        self._scenario_index = 0
        self._scenario_start_sample = 0
        self._scenario_ready = True
        self._observations: list[list[dict[str, Any]]] = []
        self._outcomes: dict[int, tuple[str, str]] = {}
        self._last_wait_display = 0.0
        self._last_displayed_blocks = 0

    @property
    def status(self) -> str:
        return self._status

    @property
    def active(self) -> bool:
        return self._status == MICROPHONE_ON

    def start_test(self) -> bool:
        if self.active or self._status == MICROPHONE_STARTING:
            self._output("A microphone test is already active or starting.")
            return False

        self._set_status(MICROPHONE_STARTING)
        monitor: Any | None = None
        try:
            self._adapter = self._adapter_factory()
            self._adapter.initialize()
            microphone = self._microphone_factory()
            monitor = self._monitor_factory(microphone, self._adapter)
            set_processing_callback = getattr(
                monitor,
                "set_processing_callback",
                None,
            )
            if callable(set_processing_callback):
                set_processing_callback(self._display_processing)
            monitor.open()
        except Exception as exc:
            if monitor is not None:
                try:
                    monitor.close()
                except Exception as cleanup_error:
                    self._output(f"Microphone cleanup error: {cleanup_error}")
            self._monitor = None
            self._set_status(MICROPHONE_ERROR)
            self._output(f"{MICROPHONE_ERROR}: {type(exc).__name__}: {exc}")
            return False

        self._monitor = monitor
        self._scenario_index = 0
        self._scenario_start_sample = 0
        self._scenario_ready = True
        self._observations = [[] for _ in SCENARIOS]
        self._outcomes = {}
        self._last_wait_display = 0.0
        self._last_displayed_blocks = 0
        self._set_status(MICROPHONE_ON)
        device_index = getattr(microphone, "device_index", None)
        device_text = "default" if device_index is None else str(device_index)
        self._output(
            "Audio format: "
            f"{getattr(microphone, 'sample_rate_hz', 16_000)} Hz mono float32; "
            f"device index: {device_text}"
        )
        if not self._adapter.speaker_reference_configured:
            self._output(
                "Speaker reference unavailable: speaker results will show "
                "REVIEW REQUIRED; no enrollment will be performed."
            )
        self._show_scenario()
        return True

    def process_once(self, timeout: float = 0.1) -> dict[str, Any] | None:
        if not self.active or self._monitor is None:
            return None

        try:
            result = self._monitor.analyze_next(timeout=timeout)
        except InsufficientAudioError:
            self._display_waiting()
            return None
        except Exception as exc:
            self._record_stream_error(exc)
            return None

        window_start = self._monitor.last_window_start_sample
        if window_start is None:
            window_start = 0
        scenario = SCENARIOS[self._scenario_index]
        if self._scenario_ready:
            self._observations[self._scenario_index].append(result)
        elif window_start >= self._scenario_start_sample:
            self._scenario_ready = True
            self._output(
                f"SCENARIO READY: {scenario.name}. Perform it now; "
                "press NEXT when finished."
            )
        self._display_result(result)
        return result

    def next_scenario(self) -> bool:
        if not self.active or self._monitor is None:
            self._output("Start a test before using NEXT.")
            return False

        scenario = SCENARIOS[self._scenario_index]
        outcome, reason = evaluate_scenario(
            scenario,
            self._observations[self._scenario_index],
            speaker_reference_configured=(
                self._adapter.speaker_reference_configured
            ),
        )
        self._outcomes[self._scenario_index] = (outcome, reason)
        self._output(f"{scenario.name}: {outcome} — {reason}")

        if self._scenario_index + 1 == len(SCENARIOS):
            self._print_summary()
            self.stop_test(report_current=False)
            return True

        self._scenario_index += 1
        self._scenario_start_sample = self._monitor.samples_received
        self._scenario_ready = False
        self._show_scenario()
        return True

    def stop_test(self, *, report_current: bool = True) -> None:
        if (
            report_current
            and self.active
            and self._scenario_index not in self._outcomes
        ):
            scenario = SCENARIOS[self._scenario_index]
            outcome, reason = evaluate_scenario(
                scenario,
                self._observations[self._scenario_index],
                speaker_reference_configured=(
                    self._adapter.speaker_reference_configured
                ),
            )
            self._outcomes[self._scenario_index] = (outcome, reason)
            self._output(f"{scenario.name}: {outcome} — {reason}")
            self._print_summary()

        monitor, self._monitor = self._monitor, None
        release_error: Exception | None = None
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                release_error = exc
        if release_error is not None:
            self._set_status(MICROPHONE_ERROR)
            self._output(
                f"{MICROPHONE_ERROR}: microphone release failed: {release_error}"
            )
            return
        self._set_status(MICROPHONE_STOPPED)

    def handle_command(self, command: str) -> bool:
        normalized = command.strip().upper()
        if normalized == "START TEST":
            self.start_test()
        elif normalized == "NEXT":
            self.next_scenario()
        elif normalized == "STOP TEST":
            self.stop_test()
        elif normalized == "QUIT":
            if self.active or self._monitor is not None:
                self.stop_test()
            return False
        else:
            self._output("Use START TEST, NEXT, STOP TEST, or QUIT.")
        return True

    def _default_adapter_factory(self) -> AudioAdapter:
        return AudioAdapter(
            speaker_reference_embedding=self._speaker_reference_embedding,
            detector_loader=load_cached_detector,
        )

    def _show_scenario(self) -> None:
        scenario = SCENARIOS[self._scenario_index]
        self._output(
            f"CURRENT SCENARIO {self._scenario_index + 1}/{len(SCENARIOS)} — "
            f"{scenario.name} — {scenario.instruction}"
        )
        self._output("When finished, type NEXT. To stop immediately, type STOP TEST.")
        if not self._scenario_ready:
            self._output(
                "Wait for SCENARIO READY before performing it. "
                "Transition windows are not counted for its outcome."
            )

    def _scenario_prefix(self) -> str:
        scenario = SCENARIOS[self._scenario_index]
        return (
            f"SCENARIO {self._scenario_index + 1}/{len(SCENARIOS)} "
            f"— {scenario.name}"
        )

    def _display_waiting(self) -> None:
        now = time.monotonic()
        if now - self._last_wait_display < 0.5:
            return
        self._last_wait_display = now
        monitor = self._monitor
        if monitor is None:
            return
        received = monitor.blocks_received > self._last_displayed_blocks
        self._last_displayed_blocks = monitor.blocks_received
        audio_status = "RECEIVING" if received else "NO NEW SAMPLES"
        self._output(
            f"{self._scenario_prefix()} | {MICROPHONE_ON} | "
            f"audio={audio_status} | "
            f"RMS={monitor.last_input_rms:.5f} | "
            f"window=WAITING {monitor.pending_samples}/"
            f"{AUDIO_ANALYSIS_WINDOW_SAMPLES} samples | "
            "VAD=WAITING FOR COMPLETE WINDOW"
        )

    def _display_processing(self) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        self._output(
            f"{self._scenario_prefix()} | {MICROPHONE_ON} | "
            "audio=RECEIVING | "
            f"RMS={monitor.last_input_rms:.5f} | window=PROCESSING"
        )

    def _display_result(self, result: dict[str, Any]) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        vad = result["vad"]
        speaker = result["speaker"]
        phone = result["phone_sound"]
        vad_label = (
            "SPEECH DETECTED"
            if vad["speech_detected"]
            else "NO SPEECH"
        )
        phone_score = phone.get("score")
        phone_text = "n/a" if phone_score is None else f"{phone_score:.4f}"
        similarity = speaker.get("similarity")
        similarity_text = "n/a" if similarity is None else f"{similarity:.4f}"
        self._output(
            f"{self._scenario_prefix()} | {MICROPHONE_ON} | "
            "audio=RECEIVING | "
            f"RMS={monitor.last_input_rms:.5f} | "
            f"VAD={vad_label} | "
            f"speech={vad['speech_duration_seconds']:.3f}s | "
            "window=COMPLETE | "
            f"speaker similarity={similarity_text} | "
            f"speaker result={_speaker_label(speaker)} | "
            f"YAMNet phone score={phone_text} | "
            f"latency={monitor.last_inference_seconds * 1000:.1f}ms"
        )

    def _record_stream_error(self, error: Exception) -> None:
        scenario = SCENARIOS[self._scenario_index]
        monitor, self._monitor = self._monitor, None
        cleanup_error: Exception | None = None
        if monitor is not None:
            try:
                monitor.close()
            except Exception as exc:
                cleanup_error = exc
                self._output(f"Microphone cleanup error: {exc}")

        if (
            scenario.kind == "interruption"
            and isinstance(error, MicrophoneStreamError)
            and cleanup_error is None
        ):
            self._outcomes[self._scenario_index] = (
                "PASS",
                "The stream error was surfaced and the monitor closed cleanly.",
            )
            self._output("Microphone interruption was reported and handled.")
        else:
            detail = f"{type(error).__name__}: {error}"
            if cleanup_error is not None:
                detail += f"; cleanup failed: {cleanup_error}"
            self._outcomes[self._scenario_index] = ("FAIL", detail)
        self._print_summary()
        self._set_status(MICROPHONE_ERROR)
        self._output(f"{MICROPHONE_ERROR}: {type(error).__name__}: {error}")

    def _print_summary(self) -> None:
        self._output("MANUAL AUDIO TEST SUMMARY")
        for index, scenario in enumerate(SCENARIOS):
            outcome, reason = self._outcomes.get(
                index,
                ("INCONCLUSIVE", "Scenario was not completed."),
            )
            self._output(f"{scenario.name}: {outcome} — {reason}")

    def _set_status(self, status: str) -> None:
        self._status = status
        self._output(status)


def run_cli(speaker_reference_embedding: Any | None = None) -> None:
    command_queue: queue.Queue[str] = queue.Queue()
    tester = InteractiveAudioTest(
        speaker_reference_embedding=speaker_reference_embedding
    )
    print(MICROPHONE_OFF)

    def read_commands() -> None:
        while True:
            try:
                command = input(
                    "Command (START TEST/NEXT/STOP TEST/QUIT): "
                )
            except EOFError:
                command_queue.put("QUIT")
                return
            command_queue.put(command)
            if command.strip().upper() == "QUIT":
                return

    threading.Thread(target=read_commands, daemon=True).start()
    try:
        while True:
            try:
                command = command_queue.get(timeout=0.05)
            except queue.Empty:
                if tester.active:
                    tester.process_once(timeout=0.1)
                continue
            if not tester.handle_command(command):
                return
    except KeyboardInterrupt:
        print("Stopping the manual audio test.")
    finally:
        if tester.active:
            tester.stop_test()


if __name__ == "__main__":
    run_cli()
