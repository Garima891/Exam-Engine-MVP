
from pathlib import Path
import json

import joblib
import numpy as np
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# Project paths
ROOT = Path(__file__).resolve().parent.parent
FEATURE_FILE = ROOT / "reports" / "replay_features.csv"
OUTPUT_DIR = ROOT / "reports"

METRICS_FILE = OUTPUT_DIR / "replay_cross_validation.json"
IMPORTANCE_FILE = OUTPUT_DIR / "replay_feature_importance.csv"

# Same six files excluded during initial training
LOW_QUALITY_FILES = {
    "candidate_20261111_215604_781876.wav",
    "other_20261111_215842_465676.wav",
    "other_20261111_220206_264392.wav",
    "other_20261111_221529_483400.wav",
    "other_20261111_221556_801743.wav",
    "other_20261111_222012_872513.wav",
}


def create_model():
    return Pipeline([
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


def main():
    print("\nVOICE VYORAI - CROSS VALIDATION")
    print("================================")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_FILE}"
        )

    # 1. Load and filter dataset
    df = pd.read_csv(FEATURE_FILE)

    df = df[
        ~df["filename"].isin(LOW_QUALITY_FILES)
    ].copy()

    df = df[
        df["label"].isin(["live", "phone_replay"])
    ].copy()

    print("\nUsable recordings:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # 2. Select the same 50 acoustic features
    excluded = {
        "filename",
        "label",
        "speaker",
        "duration",
        "global_rms",
        "peak_amplitude",
    }

    feature_columns = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in excluded
    ]

    X = df[feature_columns].reset_index(drop=True)

    y = (
        df["label"]
        .map({"live": 0, "phone_replay": 1})
        .reset_index(drop=True)
    )

    if y.value_counts().min() < 3:
        raise ValueError(
            "At least 3 samples per class are needed for 3-fold CV."
        )

    # 3. Create stratified three-fold cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    oof_predictions = np.full(len(y), -1, dtype=int)
    baseline_predictions = np.full(len(y), -1, dtype=int)

    fold_scores = []
    baseline_scores = []
    importance_records = []

    # 4. Train and evaluate each fold
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print(f"\n---------- FOLD {fold} ----------")
        print("Training recordings:", len(train_idx))
        print("Testing recordings:", len(test_idx))

        # Random Forest
        model = create_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        oof_predictions[test_idx] = predictions

        score = balanced_accuracy_score(
            y_test, predictions
        )
        fold_scores.append(float(score))

        print("Random Forest balanced accuracy:", round(score, 4))

        # Dummy baseline
        baseline = DummyClassifier(strategy="prior")
        baseline.fit(X_train, y_train)

        base_pred = baseline.predict(X_test)
        baseline_predictions[test_idx] = base_pred

        base_score = balanced_accuracy_score(
            y_test, base_pred
        )
        baseline_scores.append(float(base_score))

        # Feature importance from this fold
        importances = (
            model.named_steps["classifier"].feature_importances_
        )

        for feature, importance in zip(
            feature_columns, importances
        ):
            importance_records.append({
                "fold": fold,
                "feature": feature,
                "importance": float(importance),
            })

    # 5. Overall out-of-fold evaluation
    print("\n========== OVERALL RESULTS ==========")

    accuracy = accuracy_score(y, oof_predictions)
    balanced_accuracy = balanced_accuracy_score(
        y, oof_predictions
    )

    print("Overall accuracy:", round(accuracy, 4))
    print("Overall balanced accuracy:", round(balanced_accuracy, 4))

    print("\nFold balanced accuracies:")
    for i, score in enumerate(fold_scores, start=1):
        print(f"Fold {i}: {score:.4f}")

    print(
        "\nMean fold balanced accuracy:",
        round(float(np.mean(fold_scores)), 4),
    )
    print(
        "Fold score standard deviation:",
        round(float(np.std(fold_scores)), 4),
    )

    print(
        "\nBaseline overall accuracy:",
        round(accuracy_score(y, baseline_predictions), 4),
    )

    print("\nClassification report:")
    report = classification_report(
        y,
        oof_predictions,
        labels=[0, 1],
        target_names=["Live", "Phone Replay"],
        zero_division=0,
    )
    print(report)

    print("Confusion matrix:")
    matrix = confusion_matrix(
        y,
        oof_predictions,
        labels=[0, 1],
    )
    print(matrix)

    # 6. Aggregate feature importance across all folds
    importance_df = pd.DataFrame(importance_records)

    importance_summary = (
        importance_df
        .groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )

    importance_summary.columns = [
        "feature",
        "mean_importance",
        "std_importance",
    ]

    print("\n========== TOP 15 FEATURES ==========")
    print(
        importance_summary.head(15).to_string(index=False)
    )

    # 7. Save reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    importance_summary.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    metrics = {
        "usable_recordings": int(len(df)),
        "feature_count": len(feature_columns),
        "fold_count": 3,
        "fold_balanced_accuracies": fold_scores,
        "mean_fold_balanced_accuracy": float(
            np.mean(fold_scores)
        ),
        "fold_score_std": float(np.std(fold_scores)),
        "overall_oof_accuracy": float(accuracy),
        "overall_oof_balanced_accuracy": float(
            balanced_accuracy
        ),
        "baseline_oof_accuracy": float(
            accuracy_score(y, baseline_predictions)
        ),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            y,
            oof_predictions,
            labels=[0, 1],
            target_names=["Live", "Phone Replay"],
            output_dict=True,
            zero_division=0,
        ),
        "warning": (
            "Small dataset with only two speaker identities. "
            "Recordings from the same speakers can occur in "
            "both training and validation folds. Results are "
            "experimental and do not establish generalization."
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    print("\n========== SAVED REPORTS ==========")
    print("Metrics:", METRICS_FILE)
    print("Feature importance:", IMPORTANCE_FILE)

    print("\nCross-validation completed.")
    print("Results are experimental, not production accuracy.")


if __name__ == "__main__":
    main()
