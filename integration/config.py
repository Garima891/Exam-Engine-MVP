from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PHONE_MODEL_PATH = (
    REPOSITORY_ROOT / "phone_detection" / "models" / "phone_v2_best.pt"
)
FACE_LANDMARKER_MODEL_PATH = (
    REPOSITORY_ROOT
    / "Identity_monitoring"
    / "models"
    / "face_landmarker.task"
)
FACE_RECOGNITION_MODEL_PATH = (
    REPOSITORY_ROOT
    / "Identity_monitoring"
    / "models"
    / "face_recognition_sface.onnx"
)

PHONE_CONFIDENCE_THRESHOLD = 0.50
PHONE_IMAGE_SIZE = 416
FACE_IDENTITY_THRESHOLD = 0.3822
FACE_MAX_FACES = 5

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Local MVP storage only: SQLite does not encrypt biometric embeddings.
# Restrict database-file permissions outside Git; never store raw webcam images.
CANDIDATE_DATABASE_PATH = (
    REPOSITORY_ROOT / "integration" / "candidate_embeddings.sqlite3"
)
