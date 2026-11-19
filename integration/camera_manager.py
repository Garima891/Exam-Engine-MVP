import sys
from collections.abc import Iterator
from typing import Any, Callable

from integration.config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


class CameraManager:
    """Owns the single webcam capture shared by monitoring adapters."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        width: int | None = CAMERA_WIDTH,
        height: int | None = CAMERA_HEIGHT,
        capture_api: int | None = None,
        capture_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture_api = capture_api
        self._capture_factory = capture_factory
        self._capture: Any | None = None

    def __enter__(self) -> "CameraManager":
        if self._capture is not None:
            raise RuntimeError("Camera is already open.")

        factory = self._capture_factory
        api = self.capture_api
        if factory is None:
            import cv2

            factory = cv2.VideoCapture
            if api is None and sys.platform == "win32":
                api = cv2.CAP_DSHOW

        capture = (
            factory(self.camera_index)
            if api is None
            else factory(self.camera_index, api)
        )
        self._capture = capture

        try:
            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open webcam at index {self.camera_index}."
                )
            if self.width is not None:
                capture.set(3, self.width)
            if self.height is not None:
                capture.set(4, self.height)
        except Exception:
            self.release()
            raise

        return self

    def read(self) -> Any:
        if self._capture is None:
            raise RuntimeError("Camera is not open.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Unable to capture a webcam frame.")
        return frame

    def frames(self) -> Iterator[Any]:
        if self._capture is None:
            raise RuntimeError("Camera is not open.")

        while self._capture.isOpened():
            yield self.read()

    def release(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.release()
        return False
