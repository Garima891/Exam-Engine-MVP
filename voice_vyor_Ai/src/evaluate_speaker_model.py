"""
ECAPA-TDNN Speaker Verification Evaluation
-------------------------------------------
Biometric Verification Protocol:
- Reference recording: clip_1.wav for each speaker.
- Test recordings: all other WAV files in the speaker's folder.
- Positive Class: DIFFERENT speaker (the security event: an unauthorized voice).
- Negative Class: SAME speaker (the legitimate enrolled candidate voice).

Confusion Matrix Definitions:
- True Positive (TP): Different speaker correctly flagged as DIFFERENT (score < threshold).
- False Positive (FP): Legitimate speaker falsely flagged as DIFFERENT (score < threshold) -> False Rejection.
- True Negative (TN): Legitimate speaker correctly accepted as SAME (score >= threshold).
- False Negative (FN): Different speaker falsely accepted as SAME (score >= threshold) -> False Acceptance.

Biometric Rates:
- False Acceptance Rate (FAR) = FN / (FN + TP)  [impostor accepted]
- False Rejection Rate (FRR)  = FP / (FP + TN)  [genuine rejected]
- Equal Error Rate (EER)      = Point where FAR == FRR
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import csv

import numpy as np
import soundfile as sf
import torch

from scipy.signal import resample_poly
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy

import settings


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

EVALUATION_DIR = getattr(settings, "EVALUATION_DIR", settings.DATA_DIR / "evaluation")
REPORT_DIR = getattr(settings, "REPORTS_DIR", settings.PROJECT_ROOT / "reports")
REFERENCE_NAME = "clip_1.wav"

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
MODEL_DIR = settings.MODELS_DIR / "ecapa-voxceleb"
MIN_AUDIO_DURATION_SEC = 1.0


# --------------------------------------------------
# DATASET VALIDATION (Task 1)
# --------------------------------------------------

class DatasetValidationError(Exception):
    """Raised when the evaluation dataset structure or audio files are invalid."""
    pass


def validate_audio_file(file_path: Path, min_duration: float = MIN_AUDIO_DURATION_SEC) -> Tuple[float, int]:
    """
    Check that an audio file exists, is non-empty, uncorrupted, and meets duration requirements.

    Returns:
        (duration_in_seconds, sample_rate)
    """
    if not file_path.exists():
        raise DatasetValidationError(
            f"File does not exist: {file_path}\n"
            f"Fix: Place a valid 16 kHz WAV file at this path."
        )

    if file_path.stat().st_size == 0:
        raise DatasetValidationError(
            f"Audio file is empty (0 bytes): {file_path}\n"
            f"Fix: Replace the empty file with a valid WAV recording."
        )

    try:
        info = sf.info(str(file_path))
    except Exception as exc:
        raise DatasetValidationError(
            f"Corrupted or unreadable audio file: {file_path}\n"
            f"Error details: {exc}\n"
            f"Fix: Re-encode this file as a valid PCM WAV (16 kHz mono recommended)."
        )

    duration = info.duration
    if duration < min_duration:
        raise DatasetValidationError(
            f"Audio recording too short ({duration:.2f}s < {min_duration:.1f}s): {file_path}\n"
            f"Fix: Speaker verification requires at least {min_duration:.1f} seconds of speech."
        )

    return duration, info.samplerate


def validate_evaluation_dataset(
    evaluation_dir: Path,
    min_speakers: int = 2,
    min_recordings_per_speaker: int = 2,
    min_duration: float = MIN_AUDIO_DURATION_SEC,
) -> Dict[str, Dict[str, Any]]:
    """
    Validates evaluation dataset structure and audio integrity.

    Expected structure:
        data/evaluation/
            ├── speaker_01/
            │   ├── clip_1.wav   (reference recording)
            │   ├── clip_2.wav   (test sample)
            │   └── ...
            └── speaker_02/
                ├── clip_1.wav
                └── ...

    Returns:
        Dict mapping speaker_id to {"reference": Path, "test_files": [Path, ...]}
    """
    if not evaluation_dir.exists():
        raise DatasetValidationError(
            f"Evaluation folder not found: {evaluation_dir}\n"
            f"Fix: Create the directory '{evaluation_dir}' and add speaker folders.\n"
            f"Example:\n"
            f"  {evaluation_dir}/speaker_01/clip_1.wav (reference)\n"
            f"  {evaluation_dir}/speaker_01/clip_2.wav (test)\n"
            f"  {evaluation_dir}/speaker_02/clip_1.wav (reference)\n"
            f"  {evaluation_dir}/speaker_02/clip_2.wav (test)"
        )

    speaker_dirs = sorted(
        p for p in evaluation_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if len(speaker_dirs) < min_speakers:
        found_names = [p.name for p in speaker_dirs]
        raise DatasetValidationError(
            f"Insufficient speaker folders in {evaluation_dir}.\n"
            f"Found {len(speaker_dirs)} folder(s): {found_names}. Required: at least {min_speakers}.\n"
            f"Fix: To evaluate speaker discrimination, add at least {min_speakers} different speaker folders."
        )

    dataset_manifest: Dict[str, Dict[str, Any]] = {}

    for s_dir in speaker_dirs:
        speaker_id = s_dir.name
        wav_files = sorted(p for p in s_dir.glob("*.wav") if not p.name.startswith("."))

        if len(wav_files) < min_recordings_per_speaker:
            raise DatasetValidationError(
                f"Speaker '{speaker_id}' in {s_dir} has only {len(wav_files)} WAV file(s).\n"
                f"Required: at least {min_recordings_per_speaker} WAV recordings (clip_1.wav reference + test clips).\n"
                f"Fix: Add more WAV recordings to '{s_dir}'."
            )

        ref_file = s_dir / REFERENCE_NAME
        if not ref_file.exists():
            available = [p.name for p in wav_files]
            raise DatasetValidationError(
                f"Missing reference recording '{REFERENCE_NAME}' in {s_dir}.\n"
                f"Available files: {available}\n"
                f"Fix: Rename the primary enrollment recording for this speaker to '{REFERENCE_NAME}'."
            )

        # Validate reference audio
        validate_audio_file(ref_file, min_duration=min_duration)

        test_files = [p for p in wav_files if p.name.lower() != REFERENCE_NAME.lower()]
        if not test_files:
            raise DatasetValidationError(
                f"No test audio samples for speaker '{speaker_id}' in {s_dir}.\n"
                f"Only reference '{REFERENCE_NAME}' was found.\n"
                f"Fix: Add at least one test WAV file (e.g., clip_2.wav) to evaluate against the reference."
            )

        # Validate each test file
        for t_file in test_files:
            validate_audio_file(t_file, min_duration=min_duration)

        dataset_manifest[speaker_id] = {
            "reference": ref_file,
            "test_files": test_files,
        }

    return dataset_manifest


# --------------------------------------------------
# AUDIO LOADING & PREPROCESSING
# --------------------------------------------------

def load_audio(path: Path) -> np.ndarray:
    """Load audio, convert to mono and resample to 16 kHz."""
    audio, sr = sf.read(str(path), dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {path}")

    # Check signal level (avoid silent clips)
    if float(np.max(np.abs(audio))) < 0.001:
        raise ValueError(f"Recording is virtually silent (peak < 0.001): {path}")

    if sr != settings.SR:
        audio = resample_poly(audio, settings.SR, sr).astype(np.float32)

    return audio


# --------------------------------------------------
# EMBEDDING GENERATION
# --------------------------------------------------

def load_model(savedir: Optional[Path] = None) -> SpeakerRecognition:
    """Load ECAPA-TDNN speaker recognition model."""
    savedir = savedir or MODEL_DIR
    model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )
    return model


def generate_embedding(model: SpeakerRecognition, audio: np.ndarray) -> np.ndarray:
    """Generate normalized speaker embedding from 1D mono audio array."""
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_batch(waveform)
    return embedding.squeeze().cpu().numpy()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D embeddings."""
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Cannot compute cosine similarity of zero-norm embedding.")

    score = np.dot(emb1, emb2) / (norm1 * norm2)
    return float(np.clip(score, -1.0, 1.0))


# --------------------------------------------------
# COMPARISON PROTOCOL (Task 2: Prevent Data Leakage)
# --------------------------------------------------

def prepare_dataset_embeddings(
    model: SpeakerRecognition, manifest: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    Extract reference and test embeddings based on the validated manifest.
    Returns:
        references: {speaker_id: ref_embedding}
        embeddings: {speaker_id: {test_filename: test_embedding}}
    """
    references: Dict[str, np.ndarray] = {}
    embeddings: Dict[str, Dict[str, np.ndarray]] = {}

    for speaker, files in manifest.items():
        ref_audio = load_audio(files["reference"])
        references[speaker] = generate_embedding(model, ref_audio)

        embeddings[speaker] = {}
        for test_path in files["test_files"]:
            test_audio = load_audio(test_path)
            embeddings[speaker][test_path.name] = generate_embedding(model, test_audio)

    return references, embeddings


def create_comparisons(
    references: Dict[str, np.ndarray],
    embeddings: Dict[str, Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """
    Generate independent comparison pairs:
    1. SAME speaker: compare speaker's reference (clip_1) vs speaker's own test clips.
    2. DIFFERENT speaker: compare speaker A's reference (clip_1) vs speaker B's test clips.
       Directional pair: (Ref A, Test B) avoids self-comparison and prevents data leakage.
    """
    comparisons = []
    speakers = sorted(references.keys())

    for ref_speaker in speakers:
        ref_emb = references[ref_speaker]

        # Same-speaker comparisons
        for filename, test_emb in embeddings[ref_speaker].items():
            score = cosine_similarity(ref_emb, test_emb)
            comparisons.append({
                "reference_speaker": ref_speaker,
                "test_speaker": ref_speaker,
                "test_file": filename,
                "score": round(score, 4),
                "actual": "SAME",
            })

        # Different-speaker comparisons (prevent comparing to self)
        for other_speaker in speakers:
            if other_speaker == ref_speaker:
                continue

            for filename, test_emb in embeddings[other_speaker].items():
                score = cosine_similarity(ref_emb, test_emb)
                comparisons.append({
                    "reference_speaker": ref_speaker,
                    "test_speaker": other_speaker,
                    "test_file": filename,
                    "score": round(score, 4),
                    "actual": "DIFFERENT",
                })

    return comparisons


# --------------------------------------------------
# BIOMETRIC EVALUATION METRICS (Task 2 & Task 3)
# --------------------------------------------------

def calculate_metrics(comparisons: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """
    Calculate classification and biometric verification metrics at a given threshold.

    Decision Rule:
        score < threshold  -> Predicted: DIFFERENT (Alert)
        score >= threshold -> Predicted: SAME (Pass)
    """
    tp = fp = tn = fn = 0

    for item in comparisons:
        predicted = "DIFFERENT" if item["score"] < threshold else "SAME"
        actual = item["actual"]

        if actual == "DIFFERENT" and predicted == "DIFFERENT":
            tp += 1
        elif actual == "SAME" and predicted == "DIFFERENT":
            fp += 1
        elif actual == "SAME" and predicted == "SAME":
            tn += 1
        elif actual == "DIFFERENT" and predicted == "SAME":
            fn += 1

    total = tp + fp + tn + fn
    total_different = tp + fn   # total impostor trials
    total_same = tn + fp        # total genuine trials

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_different if total_different > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Biometric Security Metrics:
    # FAR (False Acceptance Rate): Impostor falsely accepted as candidate (actual DIFFERENT, predicted SAME)
    far = fn / total_different if total_different > 0 else 0.0

    # FRR (False Rejection Rate): Genuine candidate falsely rejected as different (actual SAME, predicted DIFFERENT)
    frr = fp / total_same if total_same > 0 else 0.0

    return {
        "threshold": float(threshold),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "FAR": round(far, 4),
        "FRR": round(frr, 4),
        "FAR_percent": round(far * 100, 2),
        "FRR_percent": round(frr * 100, 2),
    }


def calculate_eer(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate Equal Error Rate (EER) and the EER threshold.
    Uses linear interpolation across threshold intervals where FAR and FRR curves cross.
    """
    if not results:
        return {"eer": None, "eer_threshold": None}

    # Sort results by ascending threshold
    sorted_res = sorted(results, key=lambda x: x["threshold"])

    best_diff = float("inf")
    best_entry = sorted_res[0]

    # Discrete search
    for r in sorted_res:
        diff = abs(r["FAR"] - r["FRR"])
        if diff < best_diff:
            best_diff = diff
            best_entry = r

    # Check for crossing point to interpolate more precisely
    interpolated_eer = None
    interpolated_threshold = None

    for i in range(len(sorted_res) - 1):
        r1, r2 = sorted_res[i], sorted_res[i + 1]
        diff1 = r1["FAR"] - r1["FRR"]
        diff2 = r2["FAR"] - r2["FRR"]

        # Check if curves cross between r1 and r2
        if (diff1 <= 0 and diff2 >= 0) or (diff1 >= 0 and diff2 <= 0):
            denom = (r2["FAR"] - r1["FAR"]) - (r2["FRR"] - r1["FRR"])
            if denom != 0:
                alpha = (r1["FRR"] - r1["FAR"]) / denom
                alpha = np.clip(alpha, 0.0, 1.0)
                interpolated_threshold = r1["threshold"] + alpha * (r2["threshold"] - r1["threshold"])
                interpolated_eer = r1["FAR"] + alpha * (r2["FAR"] - r1["FAR"])
                break

    final_eer = interpolated_eer if interpolated_eer is not None else (best_entry["FAR"] + best_entry["FRR"]) / 2.0
    final_thresh = interpolated_threshold if interpolated_threshold is not None else best_entry["threshold"]

    return {
        "eer": round(float(final_eer), 4),
        "eer_percent": round(float(final_eer) * 100, 2),
        "eer_threshold": round(float(final_thresh), 3),
        "closest_discrete_threshold": best_entry["threshold"],
    }


def select_operating_threshold(
    results: List[Dict[str, Any]], target_far: float = getattr(settings, "TARGET_FAR", 0.05)
) -> Dict[str, Any]:
    """
    Select recommended operating threshold based on target False Acceptance Rate (FAR).
    Among candidates meeting FAR <= target_far, chooses the one with the lowest FRR.
    """
    candidates = [r for r in results if r["FAR"] <= target_far]
    if candidates:
        # Lowest FRR; tie-break on highest F1
        best = min(candidates, key=lambda x: (x["FRR"], -x["f1"]))
    else:
        # Fallback to overall lowest FAR
        best = min(results, key=lambda x: (x["FAR"], x["FRR"]))

    return best


# --------------------------------------------------
# CSV REPORT SAVING
# --------------------------------------------------

def save_comparison_report(comparisons: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save detailed pairwise cosine similarity scores."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["reference_speaker", "test_speaker", "test_file", "score", "actual"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    return output_path


def save_threshold_report(results: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save threshold optimization results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    return output_path


# --------------------------------------------------
# MAIN EVALUATION PIPELINE
# --------------------------------------------------

def main():
    print("=" * 65)
    print("VOICE VYORAI - ECAPA-TDNN SPEAKER MODEL EVALUATION")
    print("=" * 65)

    # 1. Dataset validation (Task 1)
    print(f"\n[1/5] Validating evaluation dataset at: {EVALUATION_DIR}")
    try:
        manifest = validate_evaluation_dataset(EVALUATION_DIR)
        print(f"Dataset valid! Found {len(manifest)} speakers:")
        for spk, files in manifest.items():
            print(f"  - {spk}: Ref='{files['reference'].name}', Test Clips={len(files['test_files'])}")
    except DatasetValidationError as err:
        print(f"\n[ERROR] Dataset Validation Failed:\n{err}")
        return

    # 2. Load model
    print("\n[2/5] Loading ECAPA-TDNN model...")
    model = load_model()
    print("Model ready.")

    # 3. Compute embeddings
    print("\n[3/5] Computing speaker embeddings...")
    references, embeddings = prepare_dataset_embeddings(model, manifest)

    # 4. Generate comparison pairs
    comparisons = create_comparisons(references, embeddings)
    same_count = sum(item["actual"] == "SAME" for item in comparisons)
    diff_count = len(comparisons) - same_count

    print(f"\nComparison Summary:")
    print(f"  Same-speaker pairs (Genuine):    {same_count}")
    print(f"  Different-speaker pairs (Impostor): {diff_count}")
    print(f"  Total evaluation comparisons:   {len(comparisons)}")

    scores_file = REPORT_DIR / "evaluation_scores.csv"
    save_comparison_report(comparisons, scores_file)
    print(f"Saved raw pairwise scores to: {scores_file}")

    # 5. Evaluate thresholds (Task 3)
    print("\n[4/5] Evaluating threshold range...")
    results = [calculate_metrics(comparisons, t) for t in THRESHOLDS]

    thresholds_file = REPORT_DIR / "threshold_results.csv"
    save_threshold_report(results, thresholds_file)
    print(f"Saved threshold table to: {thresholds_file}")

    # Calculate EER
    eer_info = calculate_eer(results)

    # Calculate recommended operating threshold
    op_thresh = select_operating_threshold(results, target_far=getattr(settings, "TARGET_FAR", 0.05))

    # Print threshold table
    print("\n" + "=" * 88)
    print(f"{'Threshold':>9} | {'Precision':>9} | {'Recall':>7} | {'F1':>6} | {'Accuracy':>8} | {'FAR (%)':>8} | {'FRR (%)':>8}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['threshold']:9.2f} | "
            f"{r['precision']:9.3f} | "
            f"{r['recall']:7.3f} | "
            f"{r['f1']:6.3f} | "
            f"{r['accuracy']:8.3f} | "
            f"{r['FAR_percent']:7.1f}% | "
            f"{r['FRR_percent']:7.1f}%"
        )
    print("=" * 88)

    # Summary of findings
    print("\n[5/5] Biometric Optimization Summary:")
    print("-" * 50)
    if eer_info["eer"] is not None:
        print(f"Equal Error Rate (EER):       {eer_info['eer_percent']:.2f}% (at threshold {eer_info['eer_threshold']:.3f})")
    print(f"Configured Production Threshold: {settings.SPEAKER_THRESHOLD}")
    print(f"Recommended Operating Threshold: {op_thresh['threshold']:.2f} (Target FAR <= {settings.TARGET_FAR*100:.0f}%)")
    print(f"  At threshold {op_thresh['threshold']:.2f}: FAR={op_thresh['FAR_percent']}%, FRR={op_thresh['FRR_percent']}%, Accuracy={op_thresh['accuracy']*100:.1f}%, F1={op_thresh['f1']:.3f}")
    print(f"  Confusion Matrix: TP={op_thresh['TP']}, FP={op_thresh['FP']}, TN={op_thresh['TN']}, FN={op_thresh['FN']}")
    print("-" * 50)
    print("Speaker evaluation finished successfully.\n")


if __name__ == "__main__":
    main()
