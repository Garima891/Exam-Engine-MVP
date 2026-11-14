# src/replay_detection.py

from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = PROJECT_ROOT / "reports" / "replay_classifier.pkl"

SAMPLE_RATE = 16000
WINDOW_SEC = 6.0
STRIDE_SEC = 3.0

_model_bundle = None


def load_replay_model():
    """Load the trained model only once."""
    global _model_bundle

    if _model_bundle is None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Replay model not found: {MODEL_FILE}. "
                "Run src/train_replay_classifier.py first."
            )

        _model_bundle = joblib.load(MODEL_FILE)

    return _model_bundle


def add_statistics(features, name, values):
    values = np.asarray(values).flatten()

    features[f"{name}_mean"] = float(np.mean(values))
    features[f"{name}_std"] = float(np.std(values))


def extract_features_from_audio(audio, sr=16000):
    """
    Extract the same features used by extract_replay_features.py.
    Input must be mono audio at 16 kHz.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError("Audio must be mono.")

    if sr != SAMPLE_RATE:
        raise ValueError("Audio must have a 16000 Hz sample rate.")

    if len(audio) == 0:
        raise ValueError("Empty audio.")

    features = {}

    # These three measurements are retained for reporting.
    # The trained model does not use them.
    features["duration"] = len(audio) / sr
    features["global_rms"] = float(np.sqrt(np.mean(audio ** 2)))
    features["peak_amplitude"] = float(np.max(np.abs(audio)))

    # MFCC features: 20 coefficients, mean and standard deviation.
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20,
    )

    for i in range(20):
        add_statistics(features, f"mfcc_{i + 1}", mfcc[i])

    # Spectral features: same calculations as training.
    spectral_features = {
        "spectral_centroid": librosa.feature.spectral_centroid(
            y=audio, sr=sr
        ),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(
            y=audio, sr=sr
        ),
        "spectral_rolloff": librosa.feature.spectral_rolloff(
            y=audio, sr=sr
        ),
        "spectral_flatness": librosa.feature.spectral_flatness(
            y=audio
        ),
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(
            y=audio
        ),
    }

    for name, values in spectral_features.items():
        add_statistics(features, name, values)

    return features


def create_speech_windows(audio, speech_segments, sr=16000):
    """Create six-second windows fully inside detected speech segments."""
    audio = np.asarray(audio, dtype=np.float32)

    window_samples = int(WINDOW_SEC * sr)
    stride_samples = int(STRIDE_SEC * sr)

    windows = []

    for segment in speech_segments:
        start_sample = max(0, int(segment["start"] * sr))
        end_sample = min(len(audio), int(segment["end"] * sr))

        current = start_sample

        while current + window_samples <= end_sample:
            window_audio = audio[current:current + window_samples]

            windows.append({
                "start": current / sr,
                "end": (current + window_samples) / sr,
                "audio": window_audio,
            })

            current += stride_samples

    return windows


def detect_replay(audio, speech_segments, sr=16000):
    """
    Predict live vs phone replay for each six-second speech window.

    Predictions are experimental indicators, not proof of replay.
    """
    bundle = load_replay_model()

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    label_mapping = bundle["label_mapping"]
    reverse_mapping = {
        value: key for key, value in label_mapping.items()
    }

    windows = create_speech_windows(
        audio,
        speech_segments,
        sr,
    )

    results = []

    for window in windows:
        features = extract_features_from_audio(
            window["audio"],
            sr,
        )

        # Use the exact feature names and order saved during training.
        feature_frame = pd.DataFrame([features]).reindex(
            columns=feature_columns
        )

        prediction = int(model.predict(feature_frame)[0])
        predicted_label = reverse_mapping[prediction]

        result = {
            "start": round(window["start"], 2),
            "end": round(window["end"], 2),
            "prediction": predicted_label,
            "global_rms": features["global_rms"],
            "peak_amplitude": features["peak_amplitude"],
        }

        # Random Forest vote fraction, not a calibrated probability.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature_frame)[0]
            class_values = model.classes_

            result["model_vote_fraction"] = float(
                probabilities[list(class_values).index(prediction)]
            )

        results.append(result)

    return results