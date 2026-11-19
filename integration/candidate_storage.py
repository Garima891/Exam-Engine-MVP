from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from integration.module_adapters.face_identity import FaceIdentityAdapter


class CandidateStorage:
    """Local SQLite storage for candidate metadata and biometric embeddings."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
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

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        return np.asarray(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _deserialize_embedding(blob: bytes) -> np.ndarray:
        embedding = np.frombuffer(blob, dtype=np.float32).copy()
        return FaceIdentityAdapter.validate_embedding(embedding)

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
                    self._serialize_embedding(mean),
                    mean.shape[0],
                    1,
                    created_at,
                ),
            )

        return self.load_candidate(candidate_id)

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
        for blob, dimension, landmarks_available in embedding_rows:
            if dimension != FaceIdentityAdapter.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Corrupted embedding dimension for candidate: {candidate_id}"
                )
            if len(blob) != (
                FaceIdentityAdapter.EMBEDDING_DIMENSION * np.dtype(np.float32).itemsize
            ):
                raise ValueError(
                    f"Corrupted embedding size for candidate: {candidate_id}"
                )
            embeddings.append(self._deserialize_embedding(blob))

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
