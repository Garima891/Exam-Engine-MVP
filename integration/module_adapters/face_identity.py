from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from integration.config import (
    FACE_IDENTITY_THRESHOLD,
    FACE_LANDMARKER_MODEL_PATH,
    FACE_MAX_FACES,
    FACE_RECOGNITION_MODEL_PATH,
)


class FaceIdentityAdapter:
    """Detects faces and verifies them against a registered reference image."""

    EMBEDDING_DIMENSION = 128

    @classmethod
    def validate_embedding(cls, embedding: Any) -> np.ndarray:
        """Validate and L2-normalize a 128-dimensional face embedding."""
        vector = np.asarray(embedding, dtype=np.float32).flatten()
        if vector.shape != (cls.EMBEDDING_DIMENSION,):
            raise ValueError(
                f"Invalid embedding dimension: expected "
                f"{cls.EMBEDDING_DIMENSION}, got {vector.shape[0]}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("Invalid embedding: values must be finite.")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Invalid embedding: zero vector.")
        normalized = vector / norm
        if not np.all(np.isfinite(normalized)):
            raise ValueError("Invalid embedding: normalization failed.")
        return normalized.astype(np.float32, copy=False)

    def __init__(
        self,
        landmarker_model_path: str | Path = FACE_LANDMARKER_MODEL_PATH,
        recognition_model_path: str | Path = FACE_RECOGNITION_MODEL_PATH,
        identity_threshold: float = FACE_IDENTITY_THRESHOLD,
        max_faces: int = FACE_MAX_FACES,
        *,
        landmarker: Any | None = None,
        recognizer: Any | None = None,
        cv2_module: Any | None = None,
        mediapipe_module: Any | None = None,
    ) -> None:
        self.landmarker_model_path = Path(landmarker_model_path)
        self.recognition_model_path = Path(recognition_model_path)
        self.identity_threshold = identity_threshold

        if cv2_module is None:
            import cv2

            cv2_module = cv2
        if mediapipe_module is None:
            import mediapipe as mp

            mediapipe_module = mp
        self._cv2 = cv2_module
        self._mp = mediapipe_module
        self.last_timings_ms: dict[str, float] = {}

        if landmarker is None:
            if not self.landmarker_model_path.is_file():
                raise FileNotFoundError(
                    f"Face Landmarker model not found: "
                    f"{self.landmarker_model_path}"
                )
            base_options = self._mp.tasks.BaseOptions(
                model_asset_path=str(self.landmarker_model_path)
            )
            options = self._mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=self._mp.tasks.vision.RunningMode.IMAGE,
                num_faces=max_faces,
            )
            landmarker = self._mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )

        if recognizer is None:
            if not self.recognition_model_path.is_file():
                raise FileNotFoundError(
                    f"SFace model not found: {self.recognition_model_path}"
                )
            recognizer = self._cv2.FaceRecognizerSF.create(
                str(self.recognition_model_path),
                "",
            )

        self.landmarker = landmarker
        self.recognizer = recognizer

    def extract_all_face_embeddings(self, image: Any) -> list[dict[str, Any]]:
        """Return normalized SFace embeddings and boxes for all detected faces."""
        extraction_started = perf_counter()
        self._validate_frame(image)
        preprocessing_started = perf_counter()
        image_rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        preprocessing_ms = (perf_counter() - preprocessing_started) * 1000.0
        detection_started = perf_counter()
        detection_result = self.landmarker.detect(mp_image)
        face_detection_ms = (perf_counter() - detection_started) * 1000.0

        height, width = image.shape[:2]
        face_results = []
        embedding_extraction_ms = 0.0

        for face_number, landmarks in enumerate(
            detection_result.face_landmarks,
            start=1,
        ):
            if not landmarks:
                face_results.append(
                    {
                        "face_number": face_number,
                        "embedding": None,
                        "box": None,
                        "landmarks_available": False,
                        "review_required_reason": "Landmarks unavailable.",
                    }
                )
                continue

            x_coordinates = [int(point.x * width) for point in landmarks]
            y_coordinates = [int(point.y * height) for point in landmarks]

            x_min = max(0, min(x_coordinates))
            y_min = max(0, min(y_coordinates))
            x_max = min(width - 1, max(x_coordinates))
            y_max = min(height - 1, max(y_coordinates))

            box = [x_min, y_min, x_max, y_max]
            if len(landmarks) <= 362:
                face_results.append(
                    {
                        "face_number": face_number,
                        "embedding": None,
                        "box": box,
                        "landmarks_available": True,
                        "review_required_reason": (
                            "Insufficient landmarks for SFace alignment."
                        ),
                    }
                )
                continue
            if x_max <= x_min or y_max <= y_min:
                face_results.append(
                    {
                        "face_number": face_number,
                        "embedding": None,
                        "box": box,
                        "landmarks_available": True,
                        "review_required_reason": "Degenerate face bounding box.",
                    }
                )
                continue

            def get_point(index: int) -> np.ndarray:
                return np.array(
                    [
                        landmarks[index].x * width,
                        landmarks[index].y * height,
                    ],
                    dtype=np.float32,
                )

            embedding_started = perf_counter()
            right_eye = (get_point(33) + get_point(133)) / 2
            left_eye = (get_point(263) + get_point(362)) / 2
            nose = get_point(1)
            right_mouth = get_point(61)
            left_mouth = get_point(291)
            five_points = np.array(
                [right_eye, left_eye, nose, right_mouth, left_mouth],
                dtype=np.float32,
            )

            face_data = np.array(
                [
                    x_min,
                    y_min,
                    x_max - x_min,
                    y_max - y_min,
                    *five_points.flatten(),
                    1.0,
                ],
                dtype=np.float32,
            )
            aligned_face = self.recognizer.alignCrop(
                image,
                face_data.reshape(1, -1),
            )
            embedding = self.recognizer.feature(aligned_face).flatten()
            embedding_extraction_ms += (
                perf_counter() - embedding_started
            ) * 1000.0
            try:
                embedding = self.validate_embedding(embedding)
            except ValueError as exc:
                face_results.append(
                    {
                        "face_number": face_number,
                        "embedding": None,
                        "box": box,
                        "landmarks_available": True,
                        "review_required_reason": str(exc),
                    }
                )
                continue

            face_results.append({
                "face_number": face_number,
                "embedding": embedding,
                "box": box,
                "landmarks_available": True,
            })

        self.last_timings_ms = {
            "preprocessing_ms": preprocessing_ms,
            "face_detection_ms": face_detection_ms,
            "embedding_extraction_ms": embedding_extraction_ms,
            "extraction_total_ms": (perf_counter() - extraction_started) * 1000.0,
        }
        return face_results

    def register_candidate(
        self,
        reference_image_path: str | Path,
        candidate_id: str,
    ) -> dict[str, Any]:
        reference_image_path = Path(reference_image_path)
        image = self._cv2.imread(str(reference_image_path))
        if image is None:
            raise ValueError(
                f"Unable to load reference image: {reference_image_path}"
            )

        face_results = self.extract_all_face_embeddings(image)
        if not face_results:
            raise ValueError("No face detected in reference image.")
        if len(face_results) > 1:
            raise ValueError(
                "Multiple faces detected in reference image. "
                "Please provide a clear single-face reference."
            )

        return {
            "candidate_id": candidate_id,
            "embedding": face_results[0]["embedding"],
            "reference_image": str(reference_image_path),
        }

    def verify_registered_identity(
        self,
        registered_candidate: dict[str, Any],
        probe_frame: Any,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        verification_started = perf_counter()
        threshold = self.identity_threshold if threshold is None else threshold
        candidate_id = registered_candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("Registered candidate ID is required.")
        try:
            reference_embedding = self.validate_embedding(
                registered_candidate.get("embedding")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid registered candidate embedding: {exc}"
            ) from exc

        face_results = self.extract_all_face_embeddings(probe_frame)

        verification_results = []
        similarity_started = perf_counter()
        for face in face_results:
            try:
                live_embedding = self.validate_embedding(face.get("embedding"))
                if not face.get("landmarks_available", False):
                    raise ValueError("Landmarks unavailable.")
            except (TypeError, ValueError) as exc:
                verification_results.append(
                    {
                        "face_number": face["face_number"],
                        "similarity": None,
                        "predicted": "REVIEW_REQUIRED",
                        "box": face.get("box"),
                        "reason": face.get("review_required_reason", str(exc)),
                    }
                )
                continue

            similarity = float(np.dot(reference_embedding, live_embedding))
            verification_results.append(
                {
                    "face_number": face["face_number"],
                    "similarity": similarity,
                    "predicted": (
                        "MATCH" if similarity >= threshold else "MISMATCH"
                    ),
                    "box": face["box"],
                }
            )
        similarity_comparison_ms = (
            perf_counter() - similarity_started
        ) * 1000.0

        matching_faces = [
            face
            for face in verification_results
            if face["predicted"] == "MATCH"
        ]
        best_match = (
            max(matching_faces, key=lambda face: face["similarity"])
            if matching_faces
            else None
        )

        total_faces = len(face_results)
        if total_faces == 0:
            monitoring_status = "NO_FACE_DETECTED"
        elif total_faces == 1:
            monitoring_status = "SINGLE_FACE_DETECTED"
        else:
            monitoring_status = "MULTIPLE_FACES_DETECTED"

        if total_faces == 0:
            identity_status = "NOT_VERIFIED"
        elif any(
            face["predicted"] == "REVIEW_REQUIRED"
            for face in verification_results
        ) and not matching_faces:
            identity_status = "REVIEW_REQUIRED"
        elif not matching_faces:
            identity_status = "CANDIDATE_NOT_FOUND"
        elif len(matching_faces) == 1:
            identity_status = "CANDIDATE_IDENTIFIED"
        else:
            identity_status = "MULTIPLE_MATCHES_FOUND"

        self.last_timings_ms.update(
            {
                "similarity_comparison_ms": similarity_comparison_ms,
                "verification_total_ms": (
                    perf_counter() - verification_started
                ) * 1000.0,
            }
        )
        return {
            "candidate_id": candidate_id,
            "monitoring_status": monitoring_status,
            "identity_status": identity_status,
            "total_faces": total_faces,
            "matching_faces_count": len(matching_faces),
            "best_match": best_match,
            "face_results": verification_results,
            "image": probe_frame,
            "timings_ms": dict(self.last_timings_ms),
        }

    @staticmethod
    def _validate_frame(image: Any) -> None:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] == 0
            or image.shape[1] == 0
        ):
            raise ValueError(
                "Expected a non-empty BGR webcam frame with three channels."
            )
        if not np.issubdtype(image.dtype, np.number):
            raise ValueError("Webcam frame must contain numeric pixel values.")
        if np.issubdtype(image.dtype, np.complexfloating):
            raise ValueError("Webcam frame must contain real pixel values.")
        if np.issubdtype(image.dtype, np.floating) and not np.all(
            np.isfinite(image)
        ):
            raise ValueError("Webcam frame must contain finite pixel values.")
