import unittest
from types import SimpleNamespace

import numpy as np

from integration.camera_manager import CameraManager
from integration.module_adapters.face_identity import FaceIdentityAdapter
from integration.module_adapters.phone import PhoneDetectionAdapter


class FakeCapture:
    def __init__(self, frame, opened=True):
        self.frame = frame
        self.opened = opened
        self.released = False
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.settings.append((property_id, value))
        return True

    def read(self):
        return True, self.frame

    def release(self):
        self.released = True


class FakePhoneModel:
    names = {0: "phone"}

    def __init__(self):
        self.frame = None

    def predict(self, **kwargs):
        self.frame = kwargs["source"]
        return [SimpleNamespace(names=self.names, boxes=[])]


class FakeFaceLandmarker:
    def __init__(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
        points[0] = SimpleNamespace(x=0.2, y=0.2)
        points[1] = SimpleNamespace(x=0.8, y=0.8)
        self.result = SimpleNamespace(face_landmarks=[points])
        self.image = None

    def detect(self, image):
        self.image = image
        return self.result


class FakeRecognizer:
    def __init__(self):
        self.image = None

    def alignCrop(self, image, face_data):
        self.image = image
        return object()

    def feature(self, aligned_face):
        vector = np.zeros(128, dtype=np.float32)
        vector[0] = 1.0
        return vector


class FakeCV2:
    COLOR_BGR2RGB = 1

    def __init__(self):
        self.input_frame = None

    def cvtColor(self, frame, _conversion):
        self.input_frame = frame
        return frame


class FakeMediaPipe:
    ImageFormat = SimpleNamespace(SRGB=1)

    @staticmethod
    def Image(**kwargs):
        return SimpleNamespace(**kwargs)


class CameraManagerTests(unittest.TestCase):
    def test_context_releases_camera(self):
        capture = FakeCapture(np.zeros((8, 8, 3), dtype=np.uint8))
        manager = CameraManager(capture_factory=lambda _index: capture)

        with manager:
            self.assertIs(manager.read(), capture.frame)

        self.assertTrue(capture.released)
        with self.assertRaisesRegex(RuntimeError, "Camera is not open"):
            manager.read()

    def test_open_failure_releases_camera(self):
        capture = FakeCapture(None, opened=False)
        manager = CameraManager(capture_factory=lambda _index: capture)

        with self.assertRaisesRegex(RuntimeError, "Could not open webcam"):
            manager.__enter__()

        self.assertTrue(capture.released)

    def test_same_camera_frame_reaches_both_adapters(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        capture = FakeCapture(frame)
        phone_model = FakePhoneModel()
        landmarker = FakeFaceLandmarker()
        recognizer = FakeRecognizer()
        cv2_module = FakeCV2()
        phone = PhoneDetectionAdapter(model=phone_model)
        face = FaceIdentityAdapter(
            landmarker=landmarker,
            recognizer=recognizer,
            cv2_module=cv2_module,
            mediapipe_module=FakeMediaPipe,
        )
        candidate = {
            "candidate_id": "candidate-1",
            "embedding": np.array(
                [1.0] + [0.0] * 127,
                dtype=np.float32,
            ),
        }

        with CameraManager(capture_factory=lambda _index: capture) as camera:
            shared_frame = camera.read()
            phone.detect(shared_frame)
            face.verify_registered_identity(candidate, shared_frame)

        self.assertIs(phone_model.frame, frame)
        self.assertIs(cv2_module.input_frame, frame)
        self.assertIs(recognizer.image, frame)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
