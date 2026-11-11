
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "replay_dataset"
OUTPUT_DIR = PROJECT_ROOT / "reports"
OUTPUT_FILE = OUTPUT_DIR / "replay_features.csv"


def add_statistics(features, name, values):
    """Store mean and standard deviation of an audio feature."""
    values = np.asarray(values).flatten()

    features[f"{name}_mean"] = float(np.mean(values))
    features[f"{name}_std"] = float(np.std(values))


def extract_features(file_path):
    # Load audio as mono at 16 kHz
    audio, sr = librosa.load(
        str(file_path),
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Empty audio file")

    features = {}

    # Basic audio measurements
    features["duration"] = len(audio) / sr
    features["global_rms"] = float(np.sqrt(np.mean(audio ** 2)))
    features["peak_amplitude"] = float(np.max(np.abs(audio)))

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=20
    )

    for i in range(20):
        add_statistics(features, f"mfcc_{i + 1}", mfcc[i])

    # Spectral features
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


def main():
    print("VOICE VYORAI - AUDIO FEATURE EXTRACTION")

    records = []

    categories = {
        "live": "live",
        "phone_replay": "phone_replay"
    }

    for folder_name, label in categories.items():
        folder = DATASET_ROOT / folder_name

        if not folder.exists():
            print("Missing folder:", folder)
            continue

        files = sorted(folder.glob("*.wav"))

        for file_path in files:
            filename = file_path.name.lower()

            if filename.startswith("candidate_"):
                speaker = "candidate"
            elif filename.startswith("other_"):
                speaker = "other"
            else:
                speaker = "unknown"

            try:
                features = extract_features(file_path)

                # Metadata, not model input features
                record = {
                    "filename": file_path.name,
                    "label": label,
                    "speaker": speaker,
                    **features
                }

                records.append(record)

                print(
                    f"Processed: {file_path.name} "
                    f"| Label: {label} "
                    f"| Speaker: {speaker}"
                )

            except Exception as error:
                print(f"Failed: {file_path.name} | {error}")

    if not records:
        print("\nNo features extracted.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(records)
    dataframe.to_csv(OUTPUT_FILE, index=False)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Total processed:", len(dataframe))
    print("\nLabel counts:")
    print(dataframe["label"].value_counts())
    print("\nSpeaker counts:")
    print(dataframe["speaker"].value_counts())
    print("\nFeature columns:", len(dataframe.columns) - 3)
    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
