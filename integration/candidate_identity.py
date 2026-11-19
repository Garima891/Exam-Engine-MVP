from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from integration.candidate_storage import CandidateStorage
from integration.config import FACE_IDENTITY_THRESHOLD
from integration.module_adapters.face_identity import FaceIdentityAdapter


class CandidateIdentityService:
    """Candidate enrollment and timestamped live identity verification."""

    def __init__(
        self,
        face_adapter: FaceIdentityAdapter,
        storage: CandidateStorage,
        threshold: float = FACE_IDENTITY_THRESHOLD,
    ) -> None:
        self.face_adapter = face_adapter
        self.storage = storage
        self.threshold = threshold

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        reference_frames: Iterable[Any],
        *,
        overwrite: bool = False,
        expected_sample_count: int | None = None,
    ) -> dict:
        if expected_sample_count is not None and expected_sample_count <= 0:
            raise ValueError("Expected sample count must be positive.")

        valid_embeddings = []
        rejected_frames = []
        for index, frame in enumerate(reference_frames):
            face_results = self.face_adapter.extract_all_face_embeddings(frame)
            if len(face_results) != 1:
                rejected_frames.append({
                    "frame_index": index,
                    "reason": (
                        "No face detected."
                        if not face_results
                        else "Multiple faces detected."
                    ),
                })
                continue

            face = face_results[0]
            try:
                embedding = FaceIdentityAdapter.validate_embedding(
                    face["embedding"]
                )
            except ValueError as exc:
                rejected_frames.append({
                    "frame_index": index,
                    "reason": str(exc),
                })
                continue

            if not face.get("landmarks_available", True):
                rejected_frames.append({
                    "frame_index": index,
                    "reason": "Landmarks unavailable.",
                })
                continue
            valid_embeddings.append(embedding)

        if (
            expected_sample_count is not None
            and len(valid_embeddings) != expected_sample_count
        ):
            raise ValueError(
                f"Expected {expected_sample_count} valid samples; "
                f"received {len(valid_embeddings)}."
            )
        if not valid_embeddings:
            raise ValueError(
                "Candidate registration failed; no valid single-face samples.",
            )

        mean_embedding = FaceIdentityAdapter.validate_embedding(
            np.mean(valid_embeddings, axis=0)
        )
        timestamp = self._timestamp()
        record = self.storage.register_candidate(
            candidate_id=candidate_id,
            candidate_name=candidate_name,
            embeddings=[mean_embedding],
            created_at=timestamp,
            overwrite=overwrite,
        )
        record["registration_status"] = "REGISTERED"
        record["valid_sample_count"] = len(valid_embeddings)
        record["rejected_samples"] = rejected_frames
        record["timestamp"] = timestamp
        record.pop("embedding", None)
        return record

    def load_candidate(self, candidate_id: str) -> dict:
        candidate = self.storage.load_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate record not found: {candidate_id}")
        return candidate

    def verify_frame(self, candidate_id: str, frame: Any) -> dict:
        timestamp = self._timestamp()
        candidate = self.storage.load_candidate(candidate_id)
        if candidate is None:
            return {
                "timestamp": timestamp,
                "candidate_id": candidate_id,
                "identity_status": "CANDIDATE_NOT_FOUND",
                "reason": "No registered candidate record exists.",
                "total_faces": 0,
                "matched_faces": [],
                "unmatched_faces": [],
                "unexpected_faces": [],
            }

        try:
            face_results = self.face_adapter.extract_all_face_embeddings(frame)
        except ValueError as exc:
            return {
                "timestamp": timestamp,
                "candidate_id": candidate_id,
                "identity_status": "REVIEW_REQUIRED",
                "reason": str(exc),
                "total_faces": 0,
                "matched_faces": [],
                "unmatched_faces": [],
                "unexpected_faces": [],
            }

        if not face_results:
            return {
                "timestamp": timestamp,
                "candidate_id": candidate_id,
                "identity_status": "CANDIDATE_NOT_FOUND",
                "reason": "No valid face detected.",
                "total_faces": 0,
                "matched_faces": [],
                "unmatched_faces": [],
                "unexpected_faces": [],
            }

        reference_embedding = candidate["embedding"]
        all_candidates = self.storage.list_candidates()
        loaded_candidates = []
        for candidate_summary in all_candidates:
            loaded = self.storage.load_candidate(
                candidate_summary["candidate_id"]
            )
            if loaded is not None:
                loaded_candidates.append(loaded)

        matched_faces = []
        unmatched_faces = []
        multiple_matches = []

        for face in face_results:
            try:
                live_embedding = FaceIdentityAdapter.validate_embedding(
                    face["embedding"]
                )
                if not face.get("landmarks_available", True):
                    raise ValueError("Landmarks unavailable.")
            except (KeyError, TypeError, ValueError) as exc:
                unmatched_faces.append({
                    "face_number": face.get("face_number"),
                    "box": face.get("box"),
                    "status": "REVIEW_REQUIRED",
                    "reason": str(exc),
                })
                continue

            identity_matches = []
            similarities = {}
            for known_candidate in loaded_candidates:
                similarity = float(
                    np.dot(
                        known_candidate["embedding"],
                        live_embedding,
                    )
                )
                similarities[known_candidate["candidate_id"]] = similarity
                if similarity >= self.threshold:
                    identity_matches.append(known_candidate["candidate_id"])

            similarity = similarities.get(candidate_id)
            if similarity is None:
                similarity = float(np.dot(reference_embedding, live_embedding))

            observation = {
                "face_number": face.get("face_number"),
                "box": face.get("box"),
                "similarity": similarity,
            }
            if candidate_id in identity_matches:
                matched_faces.append(observation)
            else:
                unmatched_faces.append({
                    **observation,
                    "status": "CANDIDATE_UNMATCHED",
                    "matching_candidate_ids": identity_matches,
                })

            if len(identity_matches) > 1:
                multiple_matches.append({
                    "face_number": face.get("face_number"),
                    "candidate_ids": identity_matches,
                    "similarities": {
                        known_id: similarities[known_id]
                        for known_id in identity_matches
                    },
                })

        if multiple_matches:
            identity_status = "MULTIPLE_MATCHES_FOUND"
        elif matched_faces:
            identity_status = "CANDIDATE_IDENTIFIED"
        elif unmatched_faces and any(
            face["status"] == "REVIEW_REQUIRED"
            for face in unmatched_faces
            if "status" in face
        ):
            identity_status = "REVIEW_REQUIRED"
        else:
            identity_status = "CANDIDATE_UNMATCHED"

        return {
            "timestamp": timestamp,
            "candidate_id": candidate_id,
            "identity_status": identity_status,
            "total_faces": len(face_results),
            "matched_faces": matched_faces,
            "unmatched_faces": unmatched_faces,
            "multiple_matches": multiple_matches,
            "unexpected_faces": [
                face for face in unmatched_faces
                if face.get("status") == "CANDIDATE_UNMATCHED"
            ],
        }
