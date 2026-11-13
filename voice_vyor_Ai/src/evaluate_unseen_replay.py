
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from extract_replay_features import extract_features


ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "data" / "replay_holdout"
MODEL_FILE = ROOT / "reports" / "replay_classifier.pkl"

OUTPUT_DIR = ROOT / "reports"
PREDICTIONS_FILE = OUTPUT_DIR / "unseen_replay_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "unseen_replay_metrics.json"


def main():
    print("\nVOICE VYORAI - UNSEEN SPEAKER EVALUATION")
    print("========================================")

    if not MODEL_FILE.exists():
        raise FileNotFoundError("Trained classifier not found.")

    if not DATASET.exists():
        raise FileNotFoundError("Holdout dataset not found.")

    # Load the existing trained model. Do not retrain it here.
    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    records = []

    # Read both holdout folders
    for label in ["live", "phone_replay"]:
        folder = DATASET / label

        if not folder.exists():
            continue

        for file_path in sorted(folder.glob("*.wav")):
            try:
                features = extract_features(file_path)

                parts = file_path.stem.split("__")

                if len(parts) != 3:
                    print("Invalid filename format:", file_path.name)
                    continue

                speaker = parts[0]
                device = parts[1]

                records.append({
                    "filename": file_path.name,
                    "speaker": speaker,
                    "device": device,
                    "actual_label": label,
                    **features,
                })

                print("Processed:", file_path.name)

            except Exception as error:
                print("Failed:", file_path.name, "|", error)

    if not records:
        print("\nNo holdout recordings found.")
        return

    df = pd.DataFrame(records)

    # Verify both classes exist
    if df["actual_label"].nunique() != 2:
        raise ValueError(
            "Collect both live and phone_replay recordings."
        )

    # Use exactly the same feature columns as training
    X = df.reindex(columns=feature_columns)

    if X.isna().all(axis=1).any():
        raise ValueError("Some recordings have missing features.")

    # Predict without retraining
    predictions = model.predict(X)

    df["predicted_label"] = [
        "live" if prediction == 0 else "phone_replay"
        for prediction in predictions
    ]

    df["correct"] = (
        df["actual_label"] == df["predicted_label"]
    )

    y_true = df["actual_label"].map({
        "live": 0,
        "phone_replay": 1,
    })

    y_pred = pd.Series(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    balanced = balanced_accuracy_score(y_true, y_pred)

    print("\n========== HOLDOUT RESULTS ==========")
    print("Total recordings:", len(df))
    print("Accuracy:", round(accuracy, 4))
    print("Balanced accuracy:", round(balanced, 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Live", "Phone Replay"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print("Confusion matrix:")
    print(matrix)

    # Performance for each unseen speaker
    print("\n========== PER-SPEAKER RESULTS ==========")

    for speaker, group in df.groupby("speaker"):
        print(
            speaker,
            "| Correct:",
            int(group["correct"].sum()),
            "/",
            len(group),
        )

    # Save predictions and metrics
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(PREDICTIONS_FILE, index=False)

    metrics = {
        "total_recordings": int(len(df)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "confusion_matrix": matrix.tolist(),
        "speaker_count": int(df["speaker"].nunique()),
        "device_count": int(df["device"].nunique()),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Live", "Phone Replay"],
            output_dict=True,
            zero_division=0,
        ),
        "warning": (
            "Independent holdout results apply only to the "
            "recorded speakers, devices and conditions."
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\nSaved predictions:", PREDICTIONS_FILE)
    print("Saved metrics:", METRICS_FILE)

    print("\nIndependent evaluation completed.")


if __name__ == "__main__":
    main()
