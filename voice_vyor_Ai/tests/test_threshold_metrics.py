"""
Tests for Biometric Verification Metrics and Threshold Optimization (Task 2 & Task 3)
"""

import pytest
from evaluate_speaker_model import (
    calculate_metrics,
    calculate_eer,
    select_operating_threshold,
)


@pytest.fixture
def sample_comparisons():
    """
    Synthetic comparisons with 5 SAME pairs and 5 DIFFERENT pairs.
    True scores:
      SAME:      [0.70, 0.75, 0.80, 0.85, 0.90] (high similarity)
      DIFFERENT: [0.10, 0.15, 0.20, 0.30, 0.40] (low similarity)
    """
    return [
        {"actual": "SAME", "score": 0.70},
        {"actual": "SAME", "score": 0.75},
        {"actual": "SAME", "score": 0.80},
        {"actual": "SAME", "score": 0.85},
        {"actual": "SAME", "score": 0.90},
        {"actual": "DIFFERENT", "score": 0.10},
        {"actual": "DIFFERENT", "score": 0.15},
        {"actual": "DIFFERENT", "score": 0.20},
        {"actual": "DIFFERENT", "score": 0.30},
        {"actual": "DIFFERENT", "score": 0.40},
    ]


def test_calculate_metrics_perfect_threshold(sample_comparisons):
    # At threshold 0.50, all DIFFERENT scores are < 0.50, all SAME are >= 0.50
    metrics = calculate_metrics(sample_comparisons, threshold=0.50)

    assert metrics["TP"] == 5
    assert metrics["TN"] == 5
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["FAR"] == 0.0
    assert metrics["FRR"] == 0.0


def test_calculate_metrics_low_threshold(sample_comparisons):
    # At threshold 0.05, no scores are below 0.05
    # Everything predicted as SAME
    metrics = calculate_metrics(sample_comparisons, threshold=0.05)

    assert metrics["TP"] == 0
    assert metrics["TN"] == 5
    assert metrics["FP"] == 0
    assert metrics["FN"] == 5  # all impostors accepted
    assert metrics["FAR"] == 1.0
    assert metrics["FRR"] == 0.0


def test_calculate_metrics_high_threshold(sample_comparisons):
    # At threshold 0.95, all scores are below 0.95
    # Everything predicted as DIFFERENT
    metrics = calculate_metrics(sample_comparisons, threshold=0.95)

    assert metrics["TP"] == 5
    assert metrics["TN"] == 0
    assert metrics["FP"] == 5  # all genuine rejected
    assert metrics["FN"] == 0
    assert metrics["FAR"] == 0.0
    assert metrics["FRR"] == 1.0


def test_calculate_eer_linear_crossing():
    # Synthetic results where FAR and FRR cross
    results = [
        {"threshold": 0.30, "FAR": 0.40, "FRR": 0.00},
        {"threshold": 0.40, "FAR": 0.20, "FRR": 0.10},
        {"threshold": 0.50, "FAR": 0.05, "FRR": 0.25},
        {"threshold": 0.60, "FAR": 0.00, "FRR": 0.50},
    ]
    eer_info = calculate_eer(results)
    assert eer_info["eer"] is not None
    assert 0.05 <= eer_info["eer"] <= 0.25
    assert 0.40 <= eer_info["eer_threshold"] <= 0.50


def test_select_operating_threshold():
    results = [
        {"threshold": 0.30, "FAR": 0.30, "FRR": 0.00, "f1": 0.6, "accuracy": 0.8},
        {"threshold": 0.40, "FAR": 0.10, "FRR": 0.05, "f1": 0.85, "accuracy": 0.9},
        {"threshold": 0.45, "FAR": 0.04, "FRR": 0.08, "f1": 0.92, "accuracy": 0.94},  # FAR <= 0.05
        {"threshold": 0.50, "FAR": 0.01, "FRR": 0.15, "f1": 0.88, "accuracy": 0.92},  # FAR <= 0.05
    ]
    best = select_operating_threshold(results, target_far=0.05)
    # Among threshold 0.45 (FRR=0.08) and 0.50 (FRR=0.15), 0.45 has lower FRR
    assert best["threshold"] == 0.45

