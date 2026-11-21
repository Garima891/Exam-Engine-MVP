import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integration.audio_live_test import (
    MICROPHONE_ERROR,
    MICROPHONE_OFF,
    MICROPHONE_ON,
    MICROPHONE_STOPPED,
    SCENARIOS,
    AudioTestScenario,
    InteractiveAudioTest,
    evaluate_scenario,
    load_cached_detector,
    _speaker_label,
)
from integration.audio_monitor import InsufficientAudioError
from integration.microphone_manager import MicrophoneStreamError
from voice_vyor_Ai.ecapa_model_loader import (
    ECAPA_MODEL_FILES,
    ECAPA_MODEL_ID,
    ECAPA_MODEL_REVISION,
)


def audio_result(
    *,
    speech=False,
    duration=0.0,
    speaker_status="no_speech",
    comparison=None,
    similarity=None,
    phone_detected=False,
    phone_score=0.0,
):
    return {
        "module": "audio_detection",
        "vad": {
            "speech_detected": speech,
            "speech_duration_seconds": duration,
        },
        "speaker": {
            "status": speaker_status,
            "comparison": comparison,
            "similarity": similarity,
        },
        "phone_sound": {
            "status": "scored",
            "detected": phone_detected,
            "score": phone_score,
        },
    }


class FakeAdapter:
    def __init__(self, *, reference_configured=False, calls=None):
        self.speaker_reference_configured = reference_configured
        self.calls = calls if calls is not None else []

    def initialize(self):
        self.calls.append("initialize")


class FakeMicrophone:
    def __init__(self, calls):
        self.calls = calls


class FakeMonitor:
    def __init__(
        self,
        microphone,
        adapter,
        calls,
        results=None,
        close_error=None,
    ):
        self.microphone = microphone
        self.adapter = adapter
        self.calls = calls
        self.results = list(results or [])
        self.blocks_received = 0
        self.samples_received = 0
        self.pending_samples = 0
        self.last_input_rms = 0.0
        self.last_window_start_sample = 0
        self.last_inference_seconds = 0.01
        self.opened = False
        self.close_error = close_error
        self.processing_callback = None

    def set_processing_callback(self, callback):
        self.processing_callback = callback

    def open(self):
        self.calls.append("open")
        self.opened = True

    def analyze_next(self, timeout=None):
        del timeout
        if not self.results:
            raise InsufficientAudioError("waiting")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if self.processing_callback is not None:
            self.processing_callback()
        self.blocks_received += 1
        self.samples_received += 8_000
        self.last_input_rms = 0.25
        self.last_window_start_sample += 8_000
        return result

    def close(self):
        self.calls.append("close")
        self.opened = False
        if self.close_error is not None:
            raise self.close_error


class InteractiveAudioTestTests(unittest.TestCase):
    def test_starts_closed_without_constructing_capture_resources(self):
        calls = []
        tester = InteractiveAudioTest(
            adapter_factory=lambda: calls.append("adapter"),
            microphone_factory=lambda: calls.append("microphone"),
            monitor_factory=lambda *_: calls.append("monitor"),
            output=lambda _message: None,
        )

        self.assertEqual(tester.status, MICROPHONE_OFF)
        self.assertFalse(tester.active)
        self.assertEqual(calls, [])

    def test_start_command_initializes_models_before_opening_microphone(self):
        calls = []
        adapter = FakeAdapter(calls=calls)
        tester = InteractiveAudioTest(
            adapter_factory=lambda: calls.append("adapter") or adapter,
            microphone_factory=lambda: calls.append("microphone")
            or FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: (
                calls.append("monitor")
                or FakeMonitor(
                    microphone,
                    current_adapter,
                    calls,
                )
            ),
            output=lambda _message: None,
        )

        self.assertTrue(tester.handle_command("START TEST"))
        self.assertEqual(tester.status, MICROPHONE_ON)
        self.assertEqual(
            calls,
            ["adapter", "initialize", "microphone", "monitor", "open"],
        )
        tester.stop_test()

    def test_stop_test_closes_capture_and_reports_stopped(self):
        calls = []
        adapter = FakeAdapter(calls=calls)
        tester = InteractiveAudioTest(
            adapter_factory=lambda: adapter,
            microphone_factory=lambda: FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: FakeMonitor(
                microphone,
                current_adapter,
                calls,
            ),
            output=lambda _message: None,
        )

        tester.start_test()
        tester.handle_command("STOP TEST")

        self.assertEqual(tester.status, MICROPHONE_STOPPED)
        self.assertEqual(calls.count("close"), 1)

    def test_next_advances_only_when_requested_and_waits_for_clean_window(self):
        output = []
        calls = []
        adapter = FakeAdapter(calls=calls)
        monitor = FakeMonitor(None, adapter, calls)
        tester = InteractiveAudioTest(
            adapter_factory=lambda: adapter,
            microphone_factory=lambda: FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: monitor,
            output=output.append,
        )

        tester.start_test()
        self.assertTrue(
            any(
                "CURRENT SCENARIO 1/12 — Initial silence" in line
                for line in output
            )
        )
        self.assertTrue(tester.handle_command("NEXT"))
        self.assertTrue(
            any(
                "CURRENT SCENARIO 2/12 — Normal speech" in line
                for line in output
            )
        )
        self.assertTrue(
            any("Transition windows are not counted" in line for line in output)
        )

    def test_live_feedback_repeats_current_scenario_name(self):
        output = []
        calls = []
        adapter = FakeAdapter(calls=calls)
        monitor = FakeMonitor(
            None,
            adapter,
            calls,
            [audio_result()],
        )
        tester = InteractiveAudioTest(
            adapter_factory=lambda: adapter,
            microphone_factory=lambda: FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: monitor,
            output=output.append,
        )

        tester.start_test()
        tester.process_once()

        self.assertTrue(
            any(
                "SCENARIO 1/12 — Initial silence | MICROPHONE ON"
                in line
                for line in output
            )
        )

    def test_silence_during_window_fill_is_not_reported_as_speech(self):
        output = []
        calls = []
        adapter = FakeAdapter(calls=calls)
        monitor = FakeMonitor(None, adapter, calls, [InsufficientAudioError()])
        tester = InteractiveAudioTest(
            adapter_factory=lambda: adapter,
            microphone_factory=lambda: FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: monitor,
            output=output.append,
        )
        tester.start_test()

        tester.process_once(timeout=0)

        self.assertTrue(any("VAD=WAITING FOR COMPLETE WINDOW" in line for line in output))
        self.assertFalse(any("VAD=SPEECH DETECTED" in line for line in output))

    def test_structured_live_feedback_shows_review_without_reference(self):
        output = []
        calls = []
        adapter = FakeAdapter(reference_configured=False, calls=calls)
        monitor = FakeMonitor(
            None,
            adapter,
            calls,
            [
                audio_result(
                    speech=True,
                    duration=0.8,
                    speaker_status="reference_not_configured",
                )
            ],
        )
        tester = InteractiveAudioTest(
            adapter_factory=lambda: adapter,
            microphone_factory=lambda: FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: monitor,
            output=output.append,
        )
        tester.start_test()

        tester.process_once()

        self.assertTrue(any("VAD=SPEECH DETECTED" in line for line in output))
        self.assertTrue(any("speaker result=REVIEW REQUIRED" in line for line in output))
        self.assertTrue(any("YAMNet phone score=0.0000" in line for line in output))
        self.assertTrue(any("window=PROCESSING" in line for line in output))

    def test_stream_error_sets_error_status_and_closes_capture(self):
        calls = []
        adapter = FakeAdapter(calls=calls)
        monitor = FakeMonitor(
            None,
            adapter,
            calls,
            [MicrophoneStreamError("device disconnected")],
        )
        output = []
        tester = InteractiveAudioTest(
            adapter_factory=lambda: adapter,
            microphone_factory=lambda: FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: monitor,
            output=output.append,
        )
        tester.start_test()

        tester.process_once()

        self.assertEqual(tester.status, MICROPHONE_ERROR)
        self.assertFalse(monitor.opened)
        self.assertIn("close", calls)
        self.assertTrue(any(MICROPHONE_ERROR in line for line in output))

    def test_model_initialization_error_does_not_open_microphone(self):
        calls = []

        class FailingAdapter:
            speaker_reference_configured = False

            def initialize(self):
                raise RuntimeError("cached model missing")

        tester = InteractiveAudioTest(
            adapter_factory=FailingAdapter,
            microphone_factory=lambda: calls.append("microphone"),
            monitor_factory=lambda *_: calls.append("monitor"),
            output=lambda _message: None,
        )

        self.assertFalse(tester.start_test())
        self.assertEqual(tester.status, MICROPHONE_ERROR)
        self.assertEqual(calls, [])

    def test_cached_detector_loader_uses_local_models_and_rejects_downloads(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            hf_cache = root / "hf"
            snapshot = (
                hf_cache
                / f"models--{ECAPA_MODEL_ID.replace('/', '--')}"
                / "snapshots"
                / ECAPA_MODEL_REVISION
            )
            snapshot.mkdir(parents=True)
            for filename in ECAPA_MODEL_FILES:
                (snapshot / filename).write_bytes(b"mock")

            tfhub_cache = root / "tfhub"
            yamnet = tfhub_cache / "mock-yamnet"
            (yamnet / "assets").mkdir(parents=True)
            (yamnet / "assets" / "yamnet_class_map.csv").write_text(
                "index,mid,display_name\n0,/m/test,Test\n",
                encoding="utf-8",
            )
            (yamnet / "saved_model.pb").write_bytes(b"mock")

            def unexpected_download(*_args, **_kwargs):
                raise AssertionError("A download function was called.")

            def load_local_model(path, *_args, **_kwargs):
                if Path(path) != yamnet:
                    raise AssertionError("YAMNet was not loaded from the cache.")
                return path

            huggingface_hub = SimpleNamespace(
                snapshot_download=unexpected_download,
                hf_hub_download=unexpected_download,
            )
            tensorflow_hub = SimpleNamespace(load=load_local_model)
            detector = object()
            import_calls = []

            def mocked_import(name):
                import_calls.append(name)
                if name == "huggingface_hub.constants":
                    return SimpleNamespace(HF_HUB_CACHE=hf_cache)
                if name == "huggingface_hub":
                    return huggingface_hub
                if name == "tensorflow_hub":
                    return tensorflow_hub
                if name == "voice_vyor_Ai.detector":
                    downloaded_path = huggingface_hub.snapshot_download(
                        repo_id=ECAPA_MODEL_ID,
                        revision=ECAPA_MODEL_REVISION,
                        allow_patterns=list(ECAPA_MODEL_FILES),
                    )
                    self.assertEqual(Path(downloaded_path), snapshot)
                    with self.assertRaisesRegex(RuntimeError, "does not download"):
                        huggingface_hub.hf_hub_download("unused")
                    self.assertEqual(
                        Path(tensorflow_hub.load("https://tfhub.dev/google/yamnet/1")),
                        yamnet,
                    )
                    return detector
                raise AssertionError(f"Unexpected import: {name}")

            with patch.dict(os.environ, {"TFHUB_CACHE_DIR": str(tfhub_cache)}):
                with patch(
                    "integration.audio_live_test.importlib.import_module",
                    side_effect=mocked_import,
                ):
                    loaded = load_cached_detector()

            self.assertIs(loaded, detector)
            self.assertIs(huggingface_hub.snapshot_download, unexpected_download)
            self.assertIs(huggingface_hub.hf_hub_download, unexpected_download)
            self.assertIs(tensorflow_hub.load, load_local_model)
            self.assertIn("voice_vyor_Ai.detector", import_calls)

    def test_interrupt_scenario_passes_only_after_stream_error(self):
        scenario = SCENARIOS[-1]
        result, reason = evaluate_scenario(
            scenario,
            [],
            speaker_reference_configured=False,
        )

        self.assertEqual(result, "INCONCLUSIVE")
        self.assertIn("No stream interruption", reason)

    def test_interruption_pass_requires_error_and_successful_cleanup(self):
        for close_error, expected in (
            (None, "PASS"),
            (RuntimeError("release failed"), "FAIL"),
        ):
            with self.subTest(expected=expected):
                calls = []
                adapter = FakeAdapter(calls=calls)
                monitor = FakeMonitor(
                    None,
                    adapter,
                    calls,
                    [MicrophoneStreamError("device disconnected")],
                    close_error=close_error,
                )
                output = []
                tester = InteractiveAudioTest(
                    adapter_factory=lambda: adapter,
                    microphone_factory=lambda: FakeMicrophone(calls),
                    monitor_factory=lambda *_: monitor,
                    output=output.append,
                )
                tester.start_test()
                tester._scenario_index = len(SCENARIOS) - 1

                tester.process_once()

                self.assertEqual(tester.status, MICROPHONE_ERROR)
                self.assertTrue(
                    any(
                        f"Microphone interruption: {expected}" in line
                        for line in output
                    )
                )

    def test_speaker_feedback_labels_match_unmatched_and_review(self):
        self.assertEqual(
            _speaker_label(
                {"status": "compared", "comparison": "high_similarity"}
            ),
            "MATCHED",
        )
        self.assertEqual(
            _speaker_label(
                {"status": "compared", "comparison": "low_similarity"}
            ),
            "UNMATCHED",
        )
        self.assertEqual(
            _speaker_label(
                {"status": "compared", "comparison": "ambiguous_similarity"}
            ),
            "REVIEW REQUIRED",
        )
        self.assertEqual(
            _speaker_label({"status": "insufficient_speech"}),
            "INSUFFICIENT SPEECH",
        )

    def test_scenario_evaluation_uses_observed_outputs(self):
        silence = AudioTestScenario("Silence", "", "silence")
        speech = AudioTestScenario("Speech", "", "speech")
        brief = AudioTestScenario("Brief", "", "brief")
        continuous = AudioTestScenario("Continuous", "", "continuous")
        phone = AudioTestScenario("Phone", "", "phone")

        self.assertEqual(
            evaluate_scenario(
                silence,
                [audio_result()],
                speaker_reference_configured=False,
            )[0],
            "PASS",
        )
        self.assertEqual(
            evaluate_scenario(
                speech,
                [audio_result()],
                speaker_reference_configured=False,
            )[0],
            "FAIL",
        )
        self.assertEqual(
            evaluate_scenario(
                brief,
                [
                    audio_result(
                        speech=True,
                        duration=0.4,
                        speaker_status="insufficient_speech",
                    )
                ],
                speaker_reference_configured=False,
            )[0],
            "PASS",
        )
        self.assertEqual(
            evaluate_scenario(
                continuous,
                [audio_result(speech=True)],
                speaker_reference_configured=False,
            )[0],
            "INCONCLUSIVE",
        )
        self.assertEqual(
            evaluate_scenario(
                phone,
                [audio_result(phone_detected=True, phone_score=0.7)],
                speaker_reference_configured=False,
            )[0],
            "PASS",
        )

    def test_command_names_are_explicit_and_unknown_commands_do_not_start_capture(self):
        calls = []
        adapter = FakeAdapter(calls=calls)
        tester = InteractiveAudioTest(
            adapter_factory=lambda: calls.append("adapter") or adapter,
            microphone_factory=lambda: calls.append("microphone")
            or FakeMicrophone(calls),
            monitor_factory=lambda microphone, current_adapter: (
                calls.append("monitor")
                or FakeMonitor(microphone, current_adapter, calls)
            ),
            output=lambda _message: None,
        )

        self.assertTrue(tester.handle_command("start"))
        self.assertEqual(calls, [])
        self.assertTrue(tester.handle_command("START TEST"))
        self.assertEqual(
            calls,
            ["adapter", "initialize", "microphone", "monitor", "open"],
        )
        self.assertTrue(tester.active)
        tester.stop_test()


if __name__ == "__main__":
    unittest.main()
