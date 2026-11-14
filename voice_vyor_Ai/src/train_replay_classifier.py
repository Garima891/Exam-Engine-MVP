
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = PROJECT_ROOT / "reports" / "replay_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"

MODEL_FILE = OUTPUT_DIR / "replay_classifier.pkl"
METRICS_FILE = OUTPUT_DIR / "replay_classifier_metrics.json"


# Exclude recordings flagged TOO QUIET by the latest audit.
LOW_QUALITY_FILES = {
    "candidate_20261111_215604_781876.wav",
    "other_20261111_215842_465676.wav",
    "other_20261111_220206_264392.wav",
    "other_20261111_221529_483400.wav",
    "other_20261111_221556_801743.wav",
    "other_20261111_222012_872513.wav",
}


def main():
    print("\nVOICE VYORAI - REPLAY CLASSIFIER TRAINING")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}\n"
            "Run src/extract_replay_features.py first."
        )

    # 1. Load extracted features
    df = pd.read_csv(FEATURE_FILE)

    print("\nOriginal recordings:", len(df))

    # 2. Remove low-quality recordings provisionally
    df = df[~df["filename"].isin(LOW_QUALITY_FILES)].copy()

    print("After quality filtering:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # 3. Keep only the two expected classes
    df = df[df["label"].isin(["live", "phone_replay"])].copy()

    if df["label"].nunique() != 2:
        raise ValueError("Both live and phone_replay classes are required.")

    # 4. Select acoustic features.
    # Exclude basic volume and duration measurements.
    excluded_columns = {
        "filename",
        "label",
        "speaker",
        "duration",
        "global_rms",
        "peak_amplitude",
    }

    feature_columns = [
        column
        for column in df.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]

    if not feature_columns:
        raise ValueError("No numeric acoustic features found.")

    X = df[feature_columns]
    y = df["label"].map({
        "live": 0,
        "phone_replay": 1,
    })

    print("\nFeature count:", len(feature_columns))
    print("Training samples:", len(df))

    # 5. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 6. Simple baseline model
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    # 7. Random Forest pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    # 8. Evaluate predictions
    predictions = model.predict(X_test)

    print("\n========== BASELINE RESULTS ==========")
    print(
        "Accuracy:",
        round(accuracy_score(y_test, baseline_predictions), 4)
    )

    print("\n========== RANDOM FOREST RESULTS ==========")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 4)
    )

    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Live", "Phone Replay"],
        zero_division=0,
    )
    print(report)

    print("Confusion matrix:")
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    print(matrix)

    # 9. Save trained model and metadata
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "label_mapping": {
            "live": 0,
            "phone_replay": 1,
        },
        "excluded_low_quality_files": sorted(LOW_QUALITY_FILES),
    }

    joblib.dump(model_bundle, MODEL_FILE)

    # 10. Save evaluation results
    metrics = {
        "dataset_size_after_filtering": int(len(df)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "feature_count": int(len(feature_columns)),
        "class_distribution": {
            str(key): int(value)
            for key, value in df["label"].value_counts().items()
        },
        "baseline_accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        ),
        "random_forest_accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "random_forest_balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Live", "Phone Replay"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "warning": (
            "Small experimental dataset. Results do not establish "
            "real-world or unseen-speaker performance."
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\n========== SAVED FILES ==========")
    print("Model:", MODEL_FILE)
    print("Metrics:", METRICS_FILE)

    print("\nTraining completed.")
    print("IMPORTANT: Experimental baseline only, not production-ready.")


if __name__ == "__main__":
    main()
