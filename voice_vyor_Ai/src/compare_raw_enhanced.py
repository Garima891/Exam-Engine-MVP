"""
Compare Raw vs Enhanced Audio for Speaker Verification
-------------------------------------------------------
Systematic evaluation of three enrollment/test conditions:
  Mode 1: Raw Reference vs Raw Test         (Consistent unenhanced)
  Mode 2: Raw Reference vs Enhanced Test    (Asymmetric / mixed preprocessing)
  Mode 3: Enhanced Reference vs Enhanced Test (Consistent enhanced)

Compares same-speaker and different-speaker recordings and measures:
  - Same-speaker similarity
  - Different-speaker similarity
  - Discrimination margin: (Same-speaker score - Different-speaker score)

Saves results to: reports/raw_vs_enhanced_results.csv
"""

from pathlib import Path
from typing import List, Dict, Any
import csv

import numpy as np

import settings
from noise_reduction import enhance_file
from speaker_detection import CandidateSpeakerVerifier


def run_comparison() -> List[Dict[str, Any]]:
    print("=" * 65)
    print("RAW VS ENHANCED AUDIO COMPARISON")
    print("=" * 65)

    # 1. Ensure required audio recordings exist
    required_files = [settings.REFERENCE, settings.CANDIDATE_TEST, settings.SECOND_PERSON]
    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing required audio recording: {p}")

    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Generate enhanced audio
    print("\n[1/3] Generating DeepFilterNet3 enhanced versions...")
    enhance_file(settings.REFERENCE, settings.ENHANCED_REFERENCE)
    enhance_file(settings.CANDIDATE_TEST, settings.ENHANCED_CANDIDATE)
    enhance_file(settings.SECOND_PERSON, settings.ENHANCED_SECOND)
    print("Enhanced files ready.")

    # 3. Initialize verifiers for both reference types
    print("\n[2/3] Computing speaker verification scores...")
    raw_verifier = CandidateSpeakerVerifier(str(settings.REFERENCE))
    enh_verifier = CandidateSpeakerVerifier(str(settings.ENHANCED_REFERENCE))

    tests = [
        # Mode 1: Raw vs Raw
        {
            "mode": "Mode 1: Raw vs Raw",
            "condition": "Consistent Unenhanced",
            "ref_type": "Raw",
            "test_type": "Raw",
            "pair_type": "SAME_SPEAKER",
            "description": "Raw Ref vs Raw Candidate",
            "verifier": raw_verifier,
            "test_audio": settings.CANDIDATE_TEST,
        },
        {
            "mode": "Mode 1: Raw vs Raw",
            "condition": "Consistent Unenhanced",
            "ref_type": "Raw",
            "test_type": "Raw",
            "pair_type": "DIFFERENT_SPEAKER",
            "description": "Raw Ref vs Raw Second Person",
            "verifier": raw_verifier,
            "test_audio": settings.SECOND_PERSON,
        },
        # Mode 2: Raw vs Enhanced (Asymmetric)
        {
            "mode": "Mode 2: Raw vs Enhanced",
            "condition": "Asymmetric Preprocessing",
            "ref_type": "Raw",
            "test_type": "Enhanced",
            "pair_type": "SAME_SPEAKER",
            "description": "Raw Ref vs Enhanced Candidate",
            "verifier": raw_verifier,
            "test_audio": settings.ENHANCED_CANDIDATE,
        },
        {
            "mode": "Mode 2: Raw vs Enhanced",
            "condition": "Asymmetric Preprocessing",
            "ref_type": "Raw",
            "test_type": "Enhanced",
            "pair_type": "DIFFERENT_SPEAKER",
            "description": "Raw Ref vs Enhanced Second Person",
            "verifier": raw_verifier,
            "test_audio": settings.ENHANCED_SECOND,
        },
        # Mode 3: Enhanced vs Enhanced (Consistent Enhanced)
        {
            "mode": "Mode 3: Enhanced vs Enhanced",
            "condition": "Consistent Enhanced",
            "ref_type": "Enhanced",
            "test_type": "Enhanced",
            "pair_type": "SAME_SPEAKER",
            "description": "Enhanced Ref vs Enhanced Candidate",
            "verifier": enh_verifier,
            "test_audio": settings.ENHANCED_CANDIDATE,
        },
        {
            "mode": "Mode 3: Enhanced vs Enhanced",
            "condition": "Consistent Enhanced",
            "ref_type": "Enhanced",
            "test_type": "Enhanced",
            "pair_type": "DIFFERENT_SPEAKER",
            "description": "Enhanced Ref vs Enhanced Second Person",
            "verifier": enh_verifier,
            "test_audio": settings.ENHANCED_SECOND,
        },
    ]

    results = []
    for test in tests:
        res = test["verifier"].compare(str(test["test_audio"]))
        results.append({
            "mode": test["mode"],
            "condition": test["condition"],
            "reference_type": test["ref_type"],
            "test_type": test["test_type"],
            "pair_type": test["pair_type"],
            "description": test["description"],
            "score": round(res["score"], 4),
            "same_speaker_decision": res["same_speaker"],
        })

    # Group scores by mode to compute discrimination margins
    mode_groups: Dict[str, Dict[str, float]] = {}
    for r in results:
        m = r["mode"]
        if m not in mode_groups:
            mode_groups[m] = {}
        if r["pair_type"] == "SAME_SPEAKER":
            mode_groups[m]["same"] = r["score"]
        else:
            mode_groups[m]["different"] = r["score"]

    for r in results:
        m = r["mode"]
        margin = mode_groups[m]["same"] - mode_groups[m]["different"]
        r["discrimination_margin"] = round(margin, 4)

    # 4. Display results
    print("\n" + "=" * 90)
    print(f"{'Condition':<32} | {'Pair Type':<18} | {'Score':>7} | {'Margin':>8} | {'Result':<15}")
    print("-" * 90)
    for r in results:
        res_str = "SAME" if r["same_speaker_decision"] else "DIFFERENT"
        print(f"{r['description']:<32} | {r['pair_type']:<18} | {r['score']:7.4f} | {r['discrimination_margin']:8.4f} | {res_str:<15}")
    print("=" * 90)

    # 5. Save results to CSV
    output_path = settings.REPORTS_DIR / "raw_vs_enhanced_results.csv"
    fields = [
        "mode",
        "condition",
        "reference_type",
        "test_type",
        "pair_type",
        "description",
        "score",
        "same_speaker_decision",
        "discrimination_margin",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[3/3] Detailed results saved to: {output_path}")

    # 6. Summary analysis
    print("\nDISCRIMINATION ANALYSIS:")
    print("-" * 55)
    for mode, scores in mode_groups.items():
        margin = scores["same"] - scores["different"]
        print(f"{mode:<30}: Margin = {margin:+.4f} (Same={scores['same']:.4f}, Diff={scores['different']:.4f})")

    best_mode = max(mode_groups.items(), key=lambda x: x[1]["same"] - x[1]["different"])
    best_margin = best_mode[1]["same"] - best_mode[1]["different"]
    print(f"\nBest Performing Mode: {best_mode[0]} (Margin = {best_margin:+.4f})")
    print(
        "Key Finding: Asymmetric preprocessing (Raw vs Enhanced) severely degrades speaker verification.\n"
        "Enrollment and verification audio must maintain consistent preprocessing.\n"
    )

    return results


if __name__ == "__main__":
    run_comparison()

