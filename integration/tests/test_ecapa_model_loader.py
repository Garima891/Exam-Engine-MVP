import unittest
from pathlib import Path

from voice_vyor_Ai.ecapa_model_loader import (
    ECAPA_MODEL_FILES,
    ECAPA_MODEL_ID,
    ECAPA_MODEL_REVISION,
    load_ecapa_classifier,
)


class FakeEncoderClassifier:
    def __init__(self):
        self.calls = []
        self.result = object()

    def from_hparams(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class EcapaModelLoaderTests(unittest.TestCase):
    def test_downloads_pinned_model_files_and_loads_from_local_snapshot(self):
        snapshot_path = Path("C:/models/ecapa-snapshot")
        downloads = []

        def fake_snapshot_download(**kwargs):
            downloads.append(kwargs)
            return snapshot_path

        classifier = FakeEncoderClassifier()
        result = load_ecapa_classifier(classifier, fake_snapshot_download)

        self.assertIs(result, classifier.result)
        self.assertEqual(
            downloads,
            [
                {
                    "repo_id": ECAPA_MODEL_ID,
                    "revision": ECAPA_MODEL_REVISION,
                    "allow_patterns": list(ECAPA_MODEL_FILES),
                }
            ],
        )
        self.assertEqual(
            classifier.calls,
            [
                {
                    "source": str(snapshot_path),
                    "overrides": {"pretrained_path": str(snapshot_path)},
                }
            ],
        )

    def test_snapshot_download_errors_propagate_before_model_loading(self):
        classifier = FakeEncoderClassifier()

        def failing_snapshot_download(**_kwargs):
            raise OSError("snapshot download failed")

        with self.assertRaisesRegex(OSError, "snapshot download failed"):
            load_ecapa_classifier(classifier, failing_snapshot_download)

        self.assertEqual(classifier.calls, [])


if __name__ == "__main__":
    unittest.main()
