from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from integration.config import (
    PHONE_CONFIDENCE_THRESHOLD,
    PHONE_IMAGE_SIZE,
    PHONE_MODEL_PATH,
)


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def _coordinates(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [float(coordinate) for coordinate in value]


class PhoneDetectionAdapter:
    """Runs the trained V2 phone detector on a caller-provided webcam frame."""

    def __init__(
        self,
        model_path: str | Path = PHONE_MODEL_PATH,
        confidence_threshold: float = PHONE_CONFIDENCE_THRESHOLD,
        image_size: int = PHONE_IMAGE_SIZE,
        model: Any | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Phone detector weights not found: {self.model_path}"
                )
            if model_loader is None:
                from ultralytics import YOLO

                model_loader = YOLO
            model = model_loader(str(self.model_path))

        self.model = model

    def detect(self, frame: Any) -> dict[str, Any]:
        shape = getattr(frame, "shape", None)
        if shape is None or len(shape) != 3 or shape[2] != 3:
            raise ValueError("Expected a BGR webcam frame with three channels.")

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results:
            raise RuntimeError("Phone detector returned no inference result.")

        result = results[0]
        names = getattr(result, "names", None)
        if names is None:
            names = getattr(self.model, "names", {})

        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box in boxes:
                class_id = int(_scalar(box.cls))
                detections.append(
                    {
                        "bbox_xyxy": _coordinates(box.xyxy),
                        "class_id": class_id,
                        "class_name": self._class_name(names, class_id),
                        "confidence": float(_scalar(box.conf)),
                    }
                )

        return {
            "module": "phone_detection",
            "detected": bool(detections),
            "detection_count": len(detections),
            "detections": detections,
        }

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, Mapping):
            return str(names.get(class_id, class_id))
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            if 0 <= class_id < len(names):
                return str(names[class_id])
        return str(class_id)
