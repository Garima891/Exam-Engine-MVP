from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from integration.candidate_storage import CandidateStorage


class StudentRegistrationService:
    """Validate registration data and persist one student atomically."""

    def __init__(self, storage: CandidateStorage) -> None:
        self.storage = storage

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_student(
        self,
        full_name: str,
        enrollment_number: str,
        face_embeddings: Iterable[np.ndarray],
        voice_embedding: Any,
        *,
        email: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(full_name, str) or not full_name.strip():
            raise ValueError("Student full name is required.")
        if not isinstance(enrollment_number, str) or not enrollment_number.strip():
            raise ValueError("Enrollment number is required.")
        if email is not None and not isinstance(email, str):
            raise ValueError("Email must be text when provided.")

        student_id = str(uuid4())
        timestamp = self._timestamp()
        stored = self.storage.register_student(
            student_id=student_id,
            full_name=full_name,
            enrollment_number=enrollment_number,
            email=email,
            face_embeddings=face_embeddings,
            voice_embedding=voice_embedding,
            created_at=timestamp,
        )
        return {
            "student_id": stored["student_id"],
            "full_name": stored["full_name"],
            "enrollment_number": stored["enrollment_number"],
            "email": stored["email"],
            "registration_status": stored["registration_status"],
            "face_registration_status": stored["face_registration_status"],
            "voice_registration_status": stored["voice_registration_status"],
            "created_at": stored["created_at"],
            "updated_at": stored["updated_at"],
        }
