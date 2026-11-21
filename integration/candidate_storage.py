from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from integration.biometric_crypto import (
    BiometricProtector,
    ENCRYPTED_BIOMETRIC_BLOB_PREFIX,
    WindowsDpapiProtector,
    open_biometric_payload,
    seal_biometric_payload,
)
from integration.module_adapters.face_identity import FaceIdentityAdapter


class CandidateStorage:
    """SQLite storage for student records and encrypted biometric embeddings."""

    _SCHEMA_VERSION = 2
    _ENCRYPTED_BLOB_PREFIX = ENCRYPTED_BIOMETRIC_BLOB_PREFIX
    _FACE_PURPOSE = "face_embedding"
    _VOICE_PURPOSE = "voice_embedding"

    def __init__(
        self,
        database_path: str | Path,
        biometric_protector: BiometricProtector | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._biometric_protector = (
            biometric_protector
            if biometric_protector is not None
            else WindowsDpapiProtector()
        )
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path))
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA secure_delete = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            schema_version = connection.execute(
                "PRAGMA user_version"
            ).fetchone()[0]
            if schema_version > self._SCHEMA_VERSION:
                raise RuntimeError(
                    "Candidate database schema is newer than this application."
                )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    candidate_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS candidate_embeddings (
                    candidate_id TEXT NOT NULL,
                    embedding_index INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    landmarks_available INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (candidate_id, embedding_index),
                    FOREIGN KEY (candidate_id)
                        REFERENCES candidates(candidate_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    enrollment_number TEXT,
                    normalized_enrollment_number TEXT,
                    email TEXT,
                    registration_status TEXT NOT NULL,
                    face_registration_status TEXT NOT NULL,
                    voice_registration_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (student_id)
                        REFERENCES candidates(candidate_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    students_normalized_enrollment_number_unique
                ON students(normalized_enrollment_number)
                WHERE normalized_enrollment_number IS NOT NULL
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS student_voice_embeddings (
                    student_id TEXT PRIMARY KEY,
                    embedding_blob BLOB NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (student_id)
                        REFERENCES students(student_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS exam_sessions (
                    session_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    FOREIGN KEY (student_id)
                        REFERENCES students(student_id)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS proctoring_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    module TEXT NOT NULL,
                    observation_id TEXT NOT NULL,
                    observation_count INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    model_score REAL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (session_id)
                        REFERENCES exam_sessions(session_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (student_id)
                        REFERENCES students(student_id)
                        ON DELETE CASCADE
                )
                """
            )
            self._migrate_legacy_candidates(connection)
            self._encrypt_legacy_face_embeddings(connection)
            connection.execute(
                f"PRAGMA user_version = {self._SCHEMA_VERSION}"
            )

    def _migrate_legacy_candidates(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        rows = connection.execute(
            """
            SELECT candidate_id, candidate_name, created_at, updated_at
            FROM candidates
            """
        ).fetchall()
        for student_id, full_name, created_at, updated_at in rows:
            connection.execute(
                """
                INSERT OR IGNORE INTO students (
                    student_id, full_name, enrollment_number,
                    normalized_enrollment_number, email, registration_status,
                    face_registration_status, voice_registration_status,
                    created_at, updated_at
                ) VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    full_name,
                    "LEGACY_INCOMPLETE",
                    "LEGACY_INCOMPLETE",
                    "NOT_REGISTERED",
                    created_at,
                    updated_at,
                ),
            )

    def _encrypt_legacy_face_embeddings(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        rows = connection.execute(
            """
            SELECT candidate_id, embedding_index, embedding_blob,
                   embedding_dimension
            FROM candidate_embeddings
            """
        ).fetchall()
        expected_size = (
            FaceIdentityAdapter.EMBEDDING_DIMENSION
            * np.dtype(np.float32).itemsize
        )
        for candidate_id, embedding_index, blob, dimension in rows:
            stored_blob = bytes(blob)
            if stored_blob.startswith(self._ENCRYPTED_BLOB_PREFIX):
                continue
            if dimension != FaceIdentityAdapter.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Corrupted embedding dimension for candidate: {candidate_id}"
                )
            if len(stored_blob) != expected_size:
                raise ValueError(
                    f"Corrupted embedding size for candidate: {candidate_id}"
                )
            legacy = FaceIdentityAdapter.validate_embedding(
                np.frombuffer(stored_blob, dtype=np.float32).copy()
            )
            encrypted = self._seal_embedding(
                legacy,
                self._FACE_PURPOSE,
            )
            connection.execute(
                """
                UPDATE candidate_embeddings SET embedding_blob = ?
                WHERE candidate_id = ? AND embedding_index = ?
                """,
                (encrypted, candidate_id, embedding_index),
            )

    @staticmethod
    def normalize_enrollment_number(enrollment_number: str) -> tuple[str, str]:
        if not isinstance(enrollment_number, str):
            raise ValueError("Enrollment number is required.")
        display_value = enrollment_number.strip()
        normalized_value = display_value.casefold()
        if not normalized_value:
            raise ValueError("Enrollment number is required.")
        return display_value, normalized_value

    @staticmethod
    def _normalize_voice_embedding(embedding: Any) -> np.ndarray:
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("Voice embedding must be a numeric vector.") from exc
        if vector.size == 0:
            raise ValueError("Voice embedding must not be empty.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Voice embedding must contain only finite values.")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("Voice embedding must not be a zero vector.")
        return (vector / norm).astype(np.float32, copy=False)

    def _seal_embedding(self, embedding: np.ndarray, purpose: str) -> bytes:
        return seal_biometric_payload(
            np.asarray(embedding, dtype=np.float32).tobytes(),
            purpose,
            self._biometric_protector,
        )

    def _open_embedding(self, blob: bytes, purpose: str) -> np.ndarray:
        plaintext = open_biometric_payload(
            blob,
            purpose,
            self._biometric_protector,
        )
        if len(plaintext) % np.dtype(np.float32).itemsize:
            raise ValueError("Corrupted biometric embedding size.")
        return np.frombuffer(plaintext, dtype=np.float32).copy()

    def _open_face_embedding(self, blob: bytes) -> np.ndarray:
        if not blob.startswith(self._ENCRYPTED_BLOB_PREFIX):
            expected_size = (
                FaceIdentityAdapter.EMBEDDING_DIMENSION
                * np.dtype(np.float32).itemsize
            )
            if len(blob) != expected_size:
                raise ValueError("Corrupted embedding size.")
            raise ValueError(
                "Unprotected biometric data remains; database migration "
                "did not complete."
            )
        embedding = self._open_embedding(blob, self._FACE_PURPOSE)
        return FaceIdentityAdapter.validate_embedding(embedding)

    def _open_voice_embedding(self, blob: bytes, dimension: int) -> np.ndarray:
        embedding = self._open_embedding(blob, self._VOICE_PURPOSE)
        if embedding.size != dimension:
            raise ValueError("Corrupted voice embedding dimension.")
        return self._normalize_voice_embedding(embedding)

    def register_candidate(
        self,
        candidate_id: str,
        candidate_name: str,
        embeddings: Iterable[np.ndarray],
        created_at: str,
        overwrite: bool = False,
    ) -> dict:
        candidate_id = candidate_id.strip()
        candidate_name = candidate_name.strip()
        if not candidate_id:
            raise ValueError("Candidate ID is required.")
        if not candidate_name:
            raise ValueError("Candidate name is required.")

        validated = []
        for embedding in embeddings:
            validated.append(FaceIdentityAdapter.validate_embedding(embedding))
        if not validated:
            raise ValueError("At least one valid embedding is required.")

        mean = np.mean(validated, axis=0, dtype=np.float32)
        mean = FaceIdentityAdapter.validate_embedding(mean)

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT candidate_id FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if existing and not overwrite:
                raise ValueError(
                    f"Candidate ID already exists: {candidate_id}"
                )

            if existing:
                connection.execute(
                    "DELETE FROM candidate_embeddings WHERE candidate_id = ?",
                    (candidate_id,),
                )
                connection.execute(
                    """
                    UPDATE candidates
                    SET candidate_name = ?, status = ?, updated_at = ?
                    WHERE candidate_id = ?
                    """,
                    (candidate_name, "REGISTERED", created_at, candidate_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO candidates (
                        candidate_id, candidate_name, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (candidate_id, candidate_name, "REGISTERED", created_at, created_at),
                )

            connection.execute(
                """
                INSERT INTO candidate_embeddings (
                    candidate_id, embedding_index, embedding_blob,
                    embedding_dimension, landmarks_available, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    0,
                    self._seal_embedding(mean, self._FACE_PURPOSE),
                    mean.shape[0],
                    1,
                    created_at,
                ),
            )
            self._upsert_legacy_student(
                connection,
                candidate_id,
                candidate_name,
                created_at,
            )

        return self.load_candidate(candidate_id)

    @staticmethod
    def _upsert_legacy_student(
        connection: sqlite3.Connection,
        student_id: str,
        full_name: str,
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO students (
                student_id, full_name, enrollment_number,
                normalized_enrollment_number, email, registration_status,
                face_registration_status, voice_registration_status,
                created_at, updated_at
            ) VALUES (?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                full_name = excluded.full_name,
                registration_status = 'LEGACY_INCOMPLETE',
                face_registration_status = 'LEGACY_INCOMPLETE',
                voice_registration_status = 'NOT_REGISTERED',
                updated_at = excluded.updated_at
            """,
            (
                student_id,
                full_name,
                "LEGACY_INCOMPLETE",
                "LEGACY_INCOMPLETE",
                "NOT_REGISTERED",
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            "DELETE FROM student_voice_embeddings WHERE student_id = ?",
            (student_id,),
        )

    def load_candidate(self, candidate_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT candidate_id, candidate_name, status, created_at, updated_at
                FROM candidates WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            if row is None:
                return None
            embedding_rows = connection.execute(
                """
                SELECT embedding_blob, embedding_dimension, landmarks_available
                FROM candidate_embeddings
                WHERE candidate_id = ?
                ORDER BY embedding_index
                """,
                (candidate_id,),
            ).fetchall()

        if not embedding_rows:
            raise ValueError(f"Candidate record has no embeddings: {candidate_id}")

        embeddings = []
        for embedding_row in embedding_rows:
            blob, dimension = embedding_row[0], embedding_row[1]
            if dimension != FaceIdentityAdapter.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Corrupted embedding dimension for candidate: {candidate_id}"
                )
            embeddings.append(self._open_face_embedding(blob))

        return {
            "candidate_id": row[0],
            "candidate_name": row[1],
            "status": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "embedding": embeddings[0],
            "embedding_count": len(embeddings),
            "landmarks_available": bool(embedding_rows[0][2]),
        }

    def list_candidates(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id, candidate_name, status, created_at, updated_at
                FROM candidates ORDER BY candidate_id
                """
            ).fetchall()
        return [
            {
                "candidate_id": row[0],
                "candidate_name": row[1],
                "status": row[2],
                "created_at": row[3],
                "updated_at": row[4],
            }
            for row in rows
        ]

    def register_student(
        self,
        student_id: str,
        full_name: str,
        enrollment_number: str,
        email: str | None,
        face_embeddings: Iterable[np.ndarray],
        voice_embedding: Any,
        created_at: str,
    ) -> dict:
        if not isinstance(student_id, str) or not isinstance(full_name, str):
            raise ValueError("Student ID and full name must be text.")
        student_id = student_id.strip()
        full_name = full_name.strip()
        if not student_id:
            raise ValueError("Student ID is required.")
        if not full_name:
            raise ValueError("Student full name is required.")
        enrollment_display, enrollment_key = self.normalize_enrollment_number(
            enrollment_number
        )
        if email is not None and not isinstance(email, str):
            raise ValueError("Email must be text when provided.")
        email_value = email.strip() if email is not None else None
        if email_value == "":
            email_value = None

        face_vectors = [
            FaceIdentityAdapter.validate_embedding(embedding)
            for embedding in face_embeddings
        ]
        if len(face_vectors) != 3:
            raise ValueError("Exactly three valid face embeddings are required.")
        for index, embedding in enumerate(face_vectors):
            if any(
                np.allclose(embedding, prior, rtol=0.0, atol=1e-6)
                for prior in face_vectors[:index]
            ):
                raise ValueError(
                    "Face embeddings must represent distinct captures."
                )

        normalized_voice = self._normalize_voice_embedding(voice_embedding)

        try:
            with self._connect() as connection:
                duplicate = connection.execute(
                    """
                    SELECT student_id FROM students
                    WHERE normalized_enrollment_number = ?
                    """,
                    (enrollment_key,),
                ).fetchone()
                if duplicate:
                    raise ValueError(
                        f"Enrollment number already exists: {enrollment_display}"
                    )

                connection.execute(
                    """
                    INSERT INTO candidates (
                        candidate_id, candidate_name, status, created_at, updated_at
                    ) VALUES (?, ?, 'REGISTERED', ?, ?)
                    """,
                    (student_id, full_name, created_at, created_at),
                )
                connection.execute(
                    """
                    INSERT INTO students (
                        student_id, full_name, enrollment_number,
                        normalized_enrollment_number, email,
                        registration_status, face_registration_status,
                        voice_registration_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 'REGISTERED', 'COMPLETE',
                              'COMPLETE', ?, ?)
                    """,
                    (
                        student_id,
                        full_name,
                        enrollment_display,
                        enrollment_key,
                        email_value,
                        created_at,
                        created_at,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO candidate_embeddings (
                        candidate_id, embedding_index, embedding_blob,
                        embedding_dimension, landmarks_available, created_at
                    ) VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    [
                        (
                            student_id,
                            index,
                            self._seal_embedding(
                                embedding,
                                self._FACE_PURPOSE,
                            ),
                            embedding.shape[0],
                            created_at,
                        )
                        for index, embedding in enumerate(face_vectors)
                    ],
                )
                connection.execute(
                    """
                    INSERT INTO student_voice_embeddings (
                        student_id, embedding_blob, embedding_dimension, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        student_id,
                        self._seal_embedding(
                            normalized_voice,
                            self._VOICE_PURPOSE,
                        ),
                        normalized_voice.shape[0],
                        created_at,
                    ),
                )
                record = self._load_student_from_connection(
                    connection,
                    student_id,
                )
                if (
                    record["student_id"] != student_id
                    or len(record["face_embeddings"]) != 3
                    or record["voice_embedding"] is None
                    or not np.allclose(
                        np.stack(record["face_embeddings"]),
                        np.stack(face_vectors),
                        rtol=0.0,
                        atol=1e-6,
                    )
                    or not np.allclose(
                        record["voice_embedding"],
                        normalized_voice,
                        rtol=0.0,
                        atol=1e-6,
                    )
                ):
                    raise RuntimeError(
                        "Student registration reload verification failed."
                    )
        except sqlite3.IntegrityError as exc:
            if "normalized_enrollment_number" in str(exc):
                raise ValueError(
                    f"Enrollment number already exists: {enrollment_display}"
                ) from exc
            if "candidates.candidate_id" in str(exc):
                raise ValueError(f"Student ID already exists: {student_id}") from exc
            raise

        return record

    def _load_student_from_connection(
        self,
        connection: sqlite3.Connection,
        student_id: str,
    ) -> dict:
        row = connection.execute(
            """
            SELECT student_id, full_name, enrollment_number, email,
                   registration_status, face_registration_status,
                   voice_registration_status, created_at, updated_at
            FROM students WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Student record not found: {student_id}")
        face_rows = connection.execute(
            """
            SELECT embedding_blob, embedding_dimension, landmarks_available
            FROM candidate_embeddings
            WHERE candidate_id = ?
            ORDER BY embedding_index
            """,
            (student_id,),
        ).fetchall()
        face_embeddings = []
        for face_row in face_rows:
            blob, dimension = face_row[0], face_row[1]
            if dimension != FaceIdentityAdapter.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Corrupted face embedding dimension for student: {student_id}"
                )
            face_embeddings.append(self._open_face_embedding(bytes(blob)))

        voice_row = connection.execute(
            """
            SELECT embedding_blob, embedding_dimension
            FROM student_voice_embeddings WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()
        voice_embedding = None
        if voice_row is not None:
            voice_embedding = self._open_voice_embedding(
                bytes(voice_row[0]),
                int(voice_row[1]),
            )

        return {
            "student_id": row[0],
            "full_name": row[1],
            "enrollment_number": row[2],
            "email": row[3],
            "registration_status": row[4],
            "face_registration_status": row[5],
            "voice_registration_status": row[6],
            "created_at": row[7],
            "updated_at": row[8],
            "face_embeddings": face_embeddings,
            "voice_embedding": voice_embedding,
        }

    def load_student(self, student_id: str) -> dict | None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM students WHERE student_id = ?",
                (student_id,),
            ).fetchone()
            if exists is None:
                return None
            return self._load_student_from_connection(connection, student_id)

    def list_students(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT student_id, full_name, enrollment_number, email,
                       registration_status, face_registration_status,
                       voice_registration_status, created_at, updated_at
                FROM students
                ORDER BY full_name COLLATE NOCASE, student_id
                """
            ).fetchall()
        return [
            {
                "student_id": row[0],
                "full_name": row[1],
                "enrollment_number": row[2],
                "email": row[3],
                "registration_status": row[4],
                "face_registration_status": row[5],
                "voice_registration_status": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
            for row in rows
        ]

    def is_fully_registered(self, student_id: str) -> bool:
        student = self.load_student(student_id)
        if student is None:
            return False
        return (
            student["registration_status"] == "REGISTERED"
            and student["face_registration_status"] == "COMPLETE"
            and student["voice_registration_status"] == "COMPLETE"
            and len(student["face_embeddings"]) == 3
            and student["voice_embedding"] is not None
        )

    def enrollment_number_available(self, enrollment_number: str) -> bool:
        try:
            _display_value, normalized_value = self.normalize_enrollment_number(
                enrollment_number
            )
        except ValueError:
            return False
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT 1 FROM students
                WHERE normalized_enrollment_number = ?
                """,
                (normalized_value,),
            ).fetchone() is None

    def create_exam_session(self, student_id: str, started_at: str) -> dict:
        student = self.load_student(student_id)
        if student is None:
            raise KeyError(f"Student record not found: {student_id}")
        if not self.is_fully_registered(student_id):
            raise ValueError(
                "Student is not fully registered; face and voice enrollment "
                "must both be complete."
            )
        import uuid

        session_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO exam_sessions (
                    session_id, student_id, started_at, ended_at, status
                ) VALUES (?, ?, ?, NULL, 'ACTIVE')
                """,
                (session_id, student_id, started_at),
            )
        return {
            "session_id": session_id,
            "student_id": student_id,
            "started_at": started_at,
            "status": "ACTIVE",
        }

    def end_exam_session(self, session_id: str, ended_at: str) -> None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE exam_sessions
                SET ended_at = ?, status = 'ENDED'
                WHERE session_id = ? AND status = 'ACTIVE'
                """,
                (ended_at, session_id),
            )
            if updated.rowcount != 1:
                raise KeyError(f"Active exam session not found: {session_id}")

    def load_exam_session(self, session_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT session_id, student_id, started_at, ended_at, status
                FROM exam_sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "student_id": row[1],
            "started_at": row[2],
            "ended_at": row[3],
            "status": row[4],
        }

    def record_proctoring_event(
        self,
        event_id: str,
        session_id: str,
        student_id: str,
        event_type: str,
        module: str,
        observation_id: str,
        observation_count: int,
        timestamp: str,
        model_score: float | None,
        status: str,
        reason: str | None,
    ) -> dict:
        if (
            isinstance(observation_count, bool)
            or not isinstance(observation_count, int)
            or observation_count < 1
            or observation_count > 5
        ):
            raise ValueError("Observation count must be an integer from 1 to 5.")
        if status not in {"OBSERVED", "CONFIRMED", "REVIEW_REQUIRED"}:
            raise ValueError("Invalid proctoring event status.")
        with self._connect() as connection:
            session = connection.execute(
                """
                SELECT student_id FROM exam_sessions WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"Exam session not found: {session_id}")
            if session[0] != student_id:
                raise ValueError(
                    "Student ID does not belong to the exam session."
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO proctoring_events (
                    event_id, session_id, student_id, event_type, module,
                    observation_id, observation_count, timestamp, model_score,
                    status, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    session_id,
                    student_id,
                    event_type,
                    module,
                    observation_id,
                    observation_count,
                    timestamp,
                    model_score,
                    status,
                    reason,
                ),
            )
        return {
            "event_id": event_id,
            "session_id": session_id,
            "student_id": student_id,
            "event_type": event_type,
            "module": module,
            "observation_id": observation_id,
            "observation_count": observation_count,
            "timestamp": timestamp,
            "model_score": model_score,
            "status": status,
            "reason": reason,
        }

    def list_session_events(self, session_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, session_id, student_id, event_type, module,
                       observation_id, observation_count, timestamp,
                       model_score, status, reason
                FROM proctoring_events
                WHERE session_id = ?
                ORDER BY timestamp, event_id
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "event_id": row[0],
                "session_id": row[1],
                "student_id": row[2],
                "event_type": row[3],
                "module": row[4],
                "observation_id": row[5],
                "observation_count": row[6],
                "timestamp": row[7],
                "model_score": row[8],
                "status": row[9],
                "reason": row[10],
            }
            for row in rows
        ]
