import csv
import sys
import os

# Suppress oneDNN and TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Compatibility shim: In Python 3.12+, pkg_resources (from setuptools) is not bundled by default.
# tensorflow_hub imports pkg_resources to verify TF version upon import.
try:
    import pkg_resources
except ImportError:
    import types, re
    pkg_resources = types.ModuleType("pkg_resources")
    def parse_version(v):
        return tuple(int(x) for x in re.findall(r"\d+", str(v)))
    pkg_resources.parse_version = parse_version
    sys.modules["pkg_resources"] = pkg_resources

import numpy as np
import torch
import soundfile as sf
import torchaudio
import torch.nn.functional as F
import huggingface_hub
import tensorflow as tf
import tensorflow_hub as hub

from speechbrain.inference.speaker import EncoderClassifier
from silero_vad import load_silero_vad, get_speech_timestamps

if __package__:
    from .ecapa_model_loader import load_ecapa_classifier
else:
    from ecapa_model_loader import load_ecapa_classifier

# ============================================================
# CONFIGURATION
# ============================================================

SR = 16000

# Window: 3.0 seconds, hop: 0.5 seconds
WIN = 3 * SR
HOP = SR // 2

TARGET_RMS = 0.05
MAX_PEAK = 0.95

# Detection Thresholds (Calibrated from benchmark)
SIM_LOW = 0.22           # Below this -> intruder alert
SIM_HIGH = 0.38          # Above this -> candidate recovered
SIM_INSTANT_ALERT = 0.16 # Extreme mismatch -> immediate instant alert

PHONE_TH = 0.20          # Phone sound detection threshold (calibrated)
PHONE_HIGH_CONF = 0.50   # High confidence phone sound -> instant alert
PHONE_COOLDOWN = 3.0     # Cooldown between phone alerts (sec)

MIN_SPEECH_DUR = 0.6     # Min speech duration (sec) required for ECAPA

# ============================================================
# LOAD PRETRAINED MODELS
# ============================================================

print("Loading Silero VAD...")
vad = load_silero_vad()

print("Loading ECAPA-TDNN...")
spk = load_ecapa_classifier(
    EncoderClassifier,
    huggingface_hub.snapshot_download,
)

print("Loading YAMNet...")
yam = hub.load("https://tfhub.dev/google/yamnet/1")

class_map_path = yam.class_map_path().numpy().decode("utf-8")
with open(class_map_path, "r", encoding="utf-8") as f:
    classes = [row[2] for row in csv.reader(f)][1:]

PHONE_CLASSES = {
    "Telephone bell ringing",
    "Ringtone",
    "Telephone",
    "Telephone dialing, DTMF",
    "Cell phone buzzer",
    "Alarm",
    "Beep, bleep",
    "Bell",
    "Chime",
    "Buzzer",
    "Doorbell",
}

PHONE_IDX = [
    i for i, class_name in enumerate(classes)
    if class_name in PHONE_CLASSES
]

# ============================================================
# AUDIO PREPROCESSING (PRIORITY 1)
# ============================================================

def safe_normalize(w, target_rms=TARGET_RMS, max_peak=MAX_PEAK):
    """
    Apply linear gain adjustment without non-linear clipping distortion.
    If the target RMS would cause peaks to exceed max_peak, attenuate uniformly.
    """
    w = np.asarray(w, dtype=np.float32)
    if len(w) == 0:
        return w

    rms = np.sqrt(np.mean(w ** 2)) + 1e-9
    gain = target_rms / rms
    scaled = w * gain

    peak = np.max(np.abs(scaled))
    if peak > max_peak:
        scaled = scaled * (max_peak / peak)

    return scaled.astype(np.float32)

def load_wav(path, normalize=False):
    """
    Load audio as mono, 16 kHz float32.
    Preserves natural dynamics; applies safe_normalize only if normalize=True.
    """
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)

    if len(w) == 0:
        raise ValueError(f"Audio file is empty: {path}")

    if sr != SR:
        w = torchaudio.functional.resample(
            torch.from_numpy(w),
            sr,
            SR
        ).numpy()

    if normalize:
        w = safe_normalize(w)

    return w.astype(np.float32)

# ============================================================
# VAD & SPEECH EXTRACTION HELPERS (PRIORITY 2)
# ============================================================

def get_speech_segments(w, vad_threshold=0.35):
    """
    Extract speech slices using Silero VAD timestamps and return
    the concatenated speech parts plus total speech duration in seconds.
    """
    w = np.asarray(w, dtype=np.float32)
    if len(w) == 0:
        return np.zeros(0, dtype=np.float32), 0.0

    timestamps = get_speech_timestamps(
        torch.from_numpy(w.copy()),
        vad,
        sampling_rate=SR,
        threshold=vad_threshold
    )

    parts = [w[t["start"]:t["end"]] for t in timestamps if t["end"] > t["start"]]
    if not parts:
        return np.zeros(0, dtype=np.float32), 0.0

    speech = np.concatenate(parts).astype(np.float32)
    speech_dur = len(speech) / SR
    return speech, speech_dur

def speech_only(w):
    """Legacy helper: Extract all speech parts from audio."""
    speech, _ = get_speech_segments(w)
    return speech

def speech_ratio(w):
    """Calculate the proportion of speech in the audio."""
    speech, _ = get_speech_segments(w)
    return len(speech) / max(len(w), 1)

def extract_evaluable_speech(win, speech_buf, min_dur=MIN_SPEECH_DUR):
    """
    Priority 2: Accumulate speech across adjacent windows and use actual
    speech timestamps rather than evaluating unvoiced silence with ECAPA.
    Returns: (evaluable_speech_or_None, updated_speech_buf, speech_dur_in_win)
    """
    speech_parts, dur_in_win = get_speech_segments(win)

    # If this window has sufficient pure speech on its own:
    if dur_in_win >= min_dur:
        return speech_parts, np.zeros(0, dtype=np.float32), dur_in_win

    # If this window has short speech (e.g. 0.15s - 0.6s hints/whispers), accumulate:
    if dur_in_win >= 0.15:
        combined = np.concatenate([speech_buf, speech_parts]) if len(speech_buf) > 0 else speech_parts
        if len(combined) >= int(min_dur * SR):
            eval_speech = combined[:int(min_dur * SR)]
            new_buf = combined[int(HOP):]
            return eval_speech, new_buf, dur_in_win
        else:
            return None, combined, dur_in_win

    # Silence or negligible noise: age out old buffer
    new_buf = speech_buf[int(HOP):] if len(speech_buf) > int(HOP) else np.zeros(0, dtype=np.float32)
    return None, new_buf, dur_in_win

# ============================================================
# MODEL INFERENCE HELPERS
# ============================================================

def embed(w):
    """Extract a normalized ECAPA speaker embedding."""
    w = np.asarray(w, dtype=np.float32)
    if len(w) == 0:
        raise ValueError("Cannot extract embedding from empty audio.")

    audio_tensor = torch.from_numpy(w).float().unsqueeze(0)
    with torch.no_grad():
        embedding = spk.encode_batch(audio_tensor).squeeze()

    embedding = embedding.reshape(-1)
    return F.normalize(embedding, dim=0)

def phone_score(w):
    """Return the maximum configured phone-related YAMNet score."""
    if not PHONE_IDX:
        return 0.0
    w = np.asarray(w, dtype=np.float32)
    # Ensure signal has audible energy for YAMNet feature extraction
    rms = np.sqrt(np.mean(w ** 2)) + 1e-9
    if rms < 0.02:
        w = safe_normalize(w, target_rms=0.04)
    audio_tensor = tf.convert_to_tensor(w, dtype=tf.float32)
    scores, _, _ = yam(audio_tensor)
    return float(scores.numpy()[:, PHONE_IDX].max())

# ============================================================
# ENROLLMENT (PRIORITY 1)
# ============================================================

def enroll(path):
    """
    Build candidate reference embedding from 3-second speech chunks
    using safe normalization and identical preprocessing as live detection.
    """
    print(f"\nProcessing candidate enrollment from: {path}")
    w = load_wav(path, normalize=False)
    s = speech_only(w)

    if len(s) < 8 * SR:
        raise ValueError("Enrollment audio contains less than 8 seconds of speech. Please re-record.")

    chunk_len = WIN  # 3 seconds
    step = WIN // 2  # 50% overlap for comprehensive coverage
    embeddings = []

    for i in range(0, len(s) - chunk_len + 1, step):
        chunk = safe_normalize(s[i:i + chunk_len])
        embeddings.append(embed(chunk))

    if not embeddings:
        chunk = safe_normalize(s[:chunk_len] if len(s) >= chunk_len else s)
        embeddings.append(embed(chunk))

    reference = F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)
    print(f"Enrollment complete: {len(embeddings)} speech segments averaged.")
    return reference


def enroll_array(waveform):
    """Build a candidate reference embedding directly from in-memory mono audio."""
    w = np.asarray(waveform, dtype=np.float32)
    if w.ndim != 1:
        raise ValueError("Voice enrollment requires a one-dimensional mono waveform.")
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("Voice enrollment waveform is empty or invalid.")
    s = speech_only(w)
    if len(s) < 8 * SR:
        raise ValueError(
            "Voice enrollment contains less than 8 seconds of detected speech."
        )
    embeddings = []
    chunk_len = WIN
    step = WIN // 2
    for i in range(0, len(s) - chunk_len + 1, step):
        embeddings.append(embed(safe_normalize(s[i:i + chunk_len])))
    if not embeddings:
        embeddings.append(embed(safe_normalize(s[:chunk_len])))
    reference = F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)
    return reference

# ============================================================
# LIVE DETECTION (PRIORITY 3: REDUCED LATENCY & CONFIDENCE DECISIONS)
# ============================================================

def run(
    enroll_path,
    test_path,
    sim_low=SIM_LOW,
    sim_high=SIM_HIGH,
    phone_th=PHONE_TH
):
    reference = enroll(enroll_path)
    w = load_wav(test_path, normalize=False)

    history = []
    alert = False
    speech_buf = np.zeros(0, dtype=np.float32)
    last_phone_alert = -1e9
    phone_hits = 0

    print("\nStarting audio analysis...")
    print(f"Window: {WIN/SR:.1f}s | Hop: {HOP/SR:.2f}s | Calibrated Thresholds")
    print("-" * 85)
    print(f"{'time(s)':<8} | {'similarity':<10} | {'avg(3)':<8} | {'phone':<6} | {'status'}")
    print("-" * 85)

    for start in range(0, len(w) - WIN + 1, HOP):
        win = w[start:start + WIN]
        t = start / SR
        events = []

        # Priority 2: VAD-guided speech extraction & accumulation
        eval_speech, speech_buf, dur_in_win = extract_evaluable_speech(win, speech_buf)

        similarity = None
        if eval_speech is not None:
            norm_speech = safe_normalize(eval_speech)
            curr_emb = embed(norm_speech)
            similarity = float(reference @ curr_emb)

            # Priority 3: Shorter rolling history (3 windows)
            history.append(similarity)
            history = history[-3:]
            avg_sim = float(np.mean(history))

            # Tier 1: Instant Alert on High Confidence Mismatch
            if similarity < SIM_INSTANT_ALERT and dur_in_win >= MIN_SPEECH_DUR:
                alert = True
                events.append("ALERT:DIFFERENT_SPEAKER (Instant)")

            # Tier 2: 2-Window Fast Trigger
            elif len(history) >= 2 and np.mean(history[-2:]) < sim_low:
                if not alert:
                    alert = True
                    events.append("ALERT:DIFFERENT_SPEAKER (2-Hit)")

            # Tier 3: 3-Window Average
            elif len(history) == 3 and avg_sim < sim_low:
                if not alert:
                    alert = True
                    events.append("ALERT:DIFFERENT_SPEAKER (Average)")

            # Single window suspicion warning
            elif similarity < sim_low and not alert:
                events.append("WARNING:SUSPICIOUS_SPEAKER")

            # Hysteresis recovery
            elif alert and len(history) >= 2 and np.mean(history[-2:]) > sim_high:
                alert = False
                events.append("RECOVERED:ENROLLED_SPEAKER")
        else:
            avg_sim = float(np.mean(history)) if history else None

        # Phone Sound Detection (Instant high-confidence or 2-hit confirmation)
        ps = phone_score(win)
        if ps >= PHONE_HIGH_CONF and (t - last_phone_alert) > PHONE_COOLDOWN:
            last_phone_alert = t
            phone_hits = 0
            events.append("ALERT:PHONE_SOUND (Instant)")
        elif ps >= phone_th:
            phone_hits += 1
            if phone_hits >= 2 and (t - last_phone_alert) > PHONE_COOLDOWN:
                last_phone_alert = t
                phone_hits = 0
                events.append("ALERT:PHONE_SOUND (Confirmed)")
        else:
            phone_hits = 0

        sim_str = f"{similarity:.2f}" if similarity is not None else "   -   "
        avg_str = f"{avg_sim:.2f}" if avg_sim is not None else "  -   "
        event_str = " | ".join(events) if events else "CLEAN"

        print(f"{t:6.1f}s  | {sim_str:>10} | {avg_str:>8} | {ps:6.2f} | {event_str}")

    print("-" * 85)
    print("Audio analysis completed.")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detector.py <enrollment.wav> <test.wav>")
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
