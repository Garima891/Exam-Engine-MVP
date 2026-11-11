"""All paths and tunable values in one place.
Assumes this layout:   <project>/src/*.py   and   <project>/data/...
Works no matter which folder you run the scripts from.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

SR = 16000          # sample rate used everywhere
DURATION = 20       # seconds per recording

ENROLL_DIR = DATA_DIR / "enrollment"
REFERENCE = ENROLL_DIR / "candidate_reference_v2.wav"
CANDIDATE_TEST = ENROLL_DIR / "candidate_test_v2.wav"
SECOND_PERSON = ENROLL_DIR / "second_person.wav"

RAW = DATA_DIR / "raw" / "live_test.wav"
PROCESSED_DIR = DATA_DIR / "processed"
ENHANCED = PROCESSED_DIR / "live_test_enhanced.wav"
ENHANCED_CANDIDATE = PROCESSED_DIR / "candidate_enhanced.wav"
ENHANCED_REFERENCE = PROCESSED_DIR / "reference_enhanced.wav"
ENHANCED_SECOND = PROCESSED_DIR / "second_person_enhanced.wav"

# Speaker verification (SpeechBrain ECAPA default is 0.25; calibrate on your own data)
SPEAKER_THRESHOLD = 0.25

# Speech segmentation
VAD_THRESHOLD = 0.5
MERGE_GAP_SEC = 0.6      # join speech pieces separated by less than this
CHUNK_SEC = 4.0          # long speech is cut into chunks of this length
MIN_SEGMENT_SEC = 2.0    # shorter pieces are too short for a reliable voice score

# Decision engine
MIN_DIFFERENT_CHUNKS = 2  # different-voice chunks needed before flagging
