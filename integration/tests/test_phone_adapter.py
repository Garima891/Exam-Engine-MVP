import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integration.config import PHONE_MODEL_PATH
from integration.module_adapters.phone import PhoneDetectionAdapter


class FakeBox:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)
        self.cls = np.array([class_id], dtype=np.float32)


class FakeModel:
    names = {0: "phone"}

    def __init__(self, boxes):
        self.boxes = boxes
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=self.boxes, names=self.names)]


class PhoneAdapterTests(unittest.TestCase):
    def test_loads_verified_v2_model_path_once(self):
        loaded_paths = []
        model = FakeModel([])

        adapter = PhoneDetectionAdapter(
            model_loader=lambda path: loaded_paths.append(path) or model
        )

        self.assertEqual(loaded_paths, [str(PHONE_MODEL_PATH)])
        self.assertTrue(Path(loaded_paths[0]).is_file())
        self.assertIs(adapter.model, model)

    def test_returns_phone_boxes_and_uses_configured_inference(self):
        model = FakeModel([FakeBox([10, 20, 40, 60], 0.87, 0)])
        adapter = PhoneDetectionAdapter(model=model)
        frame = np.zeros((80, 100, 3), dtype=np.uint8)

        output = adapter.detect(frame)

        self.assertEqual(model.kwargs["source"] is frame, True)
        self.assertEqual(model.kwargs["conf"], 0.50)
        self.assertEqual(model.kwargs["imgsz"], 416)
        self.assertFalse(model.kwargs["verbose"])
        self.assertEqual(output["module"], "phone_detection")
        self.assertTrue(output["detected"])
        self.assertEqual(output["detection_count"], 1)
        self.assertEqual(
            output["detections"][0],
            {
                "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
                "class_id": 0,
                "class_name": "phone",
                "confidence": np.float32(0.87).item(),
            },
        )

    def test_no_boxes_returns_empty_detection_result(self):
        adapter = PhoneDetectionAdapter(model=FakeModel([]))
        output = adapter.detect(np.zeros((8, 8, 3), dtype=np.uint8))

        self.assertFalse(output["detected"])
        self.assertEqual(output["detection_count"], 0)
        self.assertEqual(output["detections"], [])

    def test_rejects_non_color_frame(self):
        adapter = PhoneDetectionAdapter(model=FakeModel([]))

        with self.assertRaisesRegex(ValueError, "three channels"):
            adapter.detect(np.zeros((8, 8), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
