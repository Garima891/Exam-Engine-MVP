import unittest

import numpy as np

from integration.module_adapters.audio import AudioAdapter


class FakeDetector:
    MIN_SPEECH_DUR = 0.6
    SR = 16_000
    HOP = 8_000
    SIM_LOW = 0.22
    SIM_HIGH = 0.38
    PHONE_TH = 0.20
    PHONE_HIGH_CONF = 0.50

    def __init__(
        self,
        speech_duration=0.8,
        phone_score=0.1,
        embedding=None,
        speech_durations=None,
        extract_error=None,
        embedding_error=None,
        phone_error=None,
    ):
        self.speech_duration = speech_duration
        self.score = phone_score
        self.embedding = (
            np.array([0.0, 1.0], dtype=np.float32)
            if embedding is None
            else np.asarray(embedding, dtype=np.float32)
        )
        self.speech_durations = list(speech_durations or [])
        self.extract_error = extract_error
        self.embedding_error = embedding_error
        self.phone_error = phone_error
        self.calls = []

    def get_speech_segments(self, audio):
        self.calls.append(("vad", audio))
        duration = (
            self.speech_durations.pop(0)
            if self.speech_durations
            else self.speech_duration
        )
        if duration <= 0:
            return np.zeros(0, dtype=np.float32), 0.0
        return np.ones(int(duration * self.SR), dtype=np.float32), duration

    def extract_evaluable_speech(self, audio, speech_buffer, min_dur=None):
        if self.extract_error is not None:
            raise self.extract_error
        speech_parts, duration = self.get_speech_segments(audio)
        min_dur = self.MIN_SPEECH_DUR if min_dur is None else min_dur
        if duration >= min_dur:
            return speech_parts, np.zeros(0, dtype=np.float32), duration
        if duration >= 0.15:
            combined = (
                np.concatenate((speech_buffer, speech_parts))
                if speech_buffer.size
                else speech_parts
            )
            if len(combined) >= int(min_dur * self.SR):
                return (
                    combined[: int(min_dur * self.SR)],
                    combined[self.HOP :],
                    duration,
                )
            return None, combined, duration
        new_buffer = (
            speech_buffer[self.HOP :]
            if len(speech_buffer) > self.HOP
            else np.zeros(0, dtype=np.float32)
        )
        return None, new_buffer, duration

    @staticmethod
    def safe_normalize(audio):
        return audio

    def embed(self, _audio):
        self.calls.append(("embed", _audio))
        if self.embedding_error is not None:
            raise self.embedding_error
        return self.embedding

    def phone_score(self, audio):
        self.calls.append(("phone", audio))
        if self.phone_error is not None:
            raise self.phone_error
        return self.score


class AudioAdapterTests(unittest.TestCase):
    def test_initialization_does_not_load_detector(self):
        calls = []
        adapter = AudioAdapter(detector_loader=lambda: calls.append("load"))

        self.assertEqual(calls, [])
        self.assertIsNone(adapter._detector)

    def test_detector_is_loaded_lazily_once_on_valid_analysis(self):
        detector = FakeDetector(speech_duration=0.0)
        calls = []
        adapter = AudioAdapter(
            detector_loader=lambda: calls.append("load") or detector
        )
        audio = np.zeros(16_000, dtype=np.float32)

        adapter.analyze(audio)
        adapter.analyze(audio)

        self.assertEqual(calls, ["load"])

    def test_explicit_initialization_loads_detector_once(self):
        detector = FakeDetector(speech_duration=0.0)
        calls = []
        adapter = AudioAdapter(
            detector_loader=lambda: calls.append("load") or detector
        )

        adapter.initialize()
        adapter.initialize()

        self.assertEqual(calls, ["load"])
        self.assertFalse(adapter.speaker_reference_configured)

    def test_rejects_invalid_audio_before_loading_detector(self):
        calls = []
        adapter = AudioAdapter(detector_loader=lambda: calls.append("load"))
        invalid_inputs = (
            (np.zeros(1, dtype=np.float64), 16_000),
            (np.zeros((10, 1), dtype=np.float32), 16_000),
            (np.zeros(0, dtype=np.float32), 16_000),
            (np.array([np.nan], dtype=np.float32), 16_000),
            (np.zeros(1, dtype=np.float32), 8_000),
        )

        for audio, sample_rate in invalid_inputs:
            with self.subTest(audio=audio.shape, sample_rate=sample_rate):
                with self.assertRaises(ValueError):
                    adapter.analyze(audio, sample_rate)

        self.assertEqual(calls, [])

    def test_no_speech_skips_speaker_embedding_but_scores_phone_independently(self):
        detector = FakeDetector(speech_duration=0.0, phone_score=0.7)
        adapter = AudioAdapter(
            speaker_reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            detector_loader=lambda: detector,
        )

        result = adapter.analyze(np.zeros(16_000, dtype=np.float32))

        self.assertFalse(result["vad"]["speech_detected"])
        self.assertEqual(result["speaker"]["status"], "no_speech")
        self.assertIsNone(result["speaker"]["similarity"])
        self.assertEqual(result["phone_sound"]["status"], "scored")
        self.assertTrue(result["phone_sound"]["detected"])
        self.assertTrue(result["phone_sound"]["high_confidence"])
        self.assertEqual([name for name, _ in detector.calls], ["vad", "phone"])

    def test_short_speech_is_accumulated_across_overlapping_windows(self):
        detector = FakeDetector(speech_durations=[0.3, 0.3])
        adapter = AudioAdapter(
            speaker_reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            detector_loader=lambda: detector,
        )
        audio = np.zeros(48_000, dtype=np.float32)

        first = adapter.analyze(audio)
        second = adapter.analyze(audio)

        self.assertEqual(first["speaker"]["status"], "insufficient_speech")
        self.assertEqual(second["speaker"]["status"], "compared")
        self.assertAlmostEqual(
            second["speaker"]["evaluated_speech_duration_seconds"],
            0.6,
        )
        self.assertAlmostEqual(second["speaker"]["similarity"], 0.0)

    def test_insufficient_speech_skips_embedding_and_keeps_phone_independent(self):
        detector = FakeDetector(speech_duration=0.3, phone_score=0.7)
        adapter = AudioAdapter(
            speaker_reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            detector_loader=lambda: detector,
        )

        result = adapter.analyze(np.zeros(48_000, dtype=np.float32))

        self.assertTrue(result["vad"]["speech_detected"])
        self.assertEqual(result["speaker"]["status"], "insufficient_speech")
        self.assertTrue(result["phone_sound"]["detected"])
        self.assertFalse(any(name == "embed" for name, _ in detector.calls))

    def test_missing_reference_reports_status_without_embedding(self):
        detector = FakeDetector(speech_duration=0.8)
        adapter = AudioAdapter(detector_loader=lambda: detector)

        result = adapter.analyze(np.zeros(48_000, dtype=np.float32))

        self.assertEqual(
            result["speaker"]["status"],
            "reference_not_configured",
        )
        self.assertIsNone(result["speaker"]["similarity"])
        self.assertFalse(any(name == "embed" for name, _ in detector.calls))

    def test_returns_independent_structured_scores(self):
        detector = FakeDetector(speech_duration=0.8, phone_score=0.3)
        adapter = AudioAdapter(
            speaker_reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            detector_loader=lambda: detector,
        )

        result = adapter.analyze(np.zeros(32_000, dtype=np.float32))

        self.assertEqual(result["module"], "audio_detection")
        self.assertEqual(result["sample_rate_hz"], 16_000)
        self.assertEqual(result["window_samples"], 32_000)
        self.assertEqual(result["vad"]["speech_duration_seconds"], 0.8)
        self.assertEqual(result["speaker"]["status"], "compared")
        self.assertEqual(result["speaker"]["comparison"], "low_similarity")
        self.assertAlmostEqual(result["speaker"]["similarity"], 0.0)
        self.assertTrue(result["phone_sound"]["detected"])
        self.assertFalse(result["phone_sound"]["high_confidence"])
        self.assertNotIn("cheating", result)

    def test_high_and_ambiguous_similarities_are_distinguished(self):
        reference = np.array([1.0, 0.0], dtype=np.float32)
        cases = (
            (np.array([1.0, 0.0], dtype=np.float32), "high_similarity"),
            (
                np.array([0.3, np.sqrt(0.91)], dtype=np.float32),
                "ambiguous_similarity",
            ),
        )

        for embedding, expected in cases:
            with self.subTest(expected=expected):
                adapter = AudioAdapter(
                    speaker_reference_embedding=reference,
                    detector_loader=lambda: FakeDetector(embedding=embedding),
                )

                result = adapter.analyze(np.zeros(48_000, dtype=np.float32))

                self.assertEqual(result["speaker"]["comparison"], expected)

    def test_embedding_dimension_mismatch_is_reported(self):
        adapter = AudioAdapter(
            speaker_reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            detector_loader=lambda: FakeDetector(
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            ),
        )

        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            adapter.analyze(np.zeros(48_000, dtype=np.float32))

    def test_detector_and_inference_exceptions_propagate(self):
        detector_error = RuntimeError("detector load failed")
        def failing_detector_loader():
            raise detector_error

        adapter = AudioAdapter(detector_loader=failing_detector_loader)
        with self.assertRaisesRegex(RuntimeError, "detector load failed"):
            adapter.analyze(np.zeros(48_000, dtype=np.float32))

        cases = (
            (FakeDetector(extract_error=RuntimeError("VAD failed")), "VAD failed"),
            (
                FakeDetector(embedding_error=RuntimeError("ECAPA failed")),
                "ECAPA failed",
            ),
            (
                FakeDetector(phone_error=RuntimeError("YAMNet failed")),
                "YAMNet failed",
            ),
        )
        for detector, expected in cases:
            with self.subTest(expected=expected):
                adapter = AudioAdapter(
                    speaker_reference_embedding=np.array(
                        [1.0, 0.0],
                        dtype=np.float32,
                    ),
                    detector_loader=lambda: detector,
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    adapter.analyze(np.zeros(48_000, dtype=np.float32))

    def test_reset_clears_accumulated_speech(self):
        detector = FakeDetector(speech_durations=[0.3, 0.3, 0.3])
        adapter = AudioAdapter(
            speaker_reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
            detector_loader=lambda: detector,
        )
        audio = np.zeros(48_000, dtype=np.float32)

        adapter.analyze(audio)
        adapter.reset()
        result = adapter.analyze(audio)

        self.assertEqual(result["speaker"]["status"], "insufficient_speech")

    def test_short_window_reports_phone_insufficient_audio(self):
        detector = FakeDetector(speech_duration=0.0)
        adapter = AudioAdapter(detector_loader=lambda: detector)

        result = adapter.analyze(np.zeros(8_000, dtype=np.float32))

        self.assertEqual(
            result["phone_sound"]["status"],
            "insufficient_audio",
        )
        self.assertIsNone(result["phone_sound"]["score"])
        self.assertEqual([name for name, _ in detector.calls], ["vad"])


if __name__ == "__main__":
    unittest.main()
