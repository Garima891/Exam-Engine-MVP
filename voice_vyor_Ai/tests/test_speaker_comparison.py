"""
Tests for Cosine Similarity and Pair Comparison Generation (Task 2 & Task 7)
"""

import numpy as np
import pytest

from evaluate_speaker_model import cosine_similarity, create_comparisons


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert pytest.approx(cosine_similarity(v, v), 1e-5) == 1.0


def test_cosine_similarity_opposite():
    v = np.array([1.0, 2.0, 3.0])
    assert pytest.approx(cosine_similarity(v, -v), 1e-5) == -1.0


def test_cosine_similarity_orthogonal():
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert pytest.approx(cosine_similarity(v1, v2), 1e-5) == 0.0


def test_cosine_similarity_zero_norm_raises():
    v1 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Cannot compute cosine similarity of zero-norm embedding"):
        cosine_similarity(v1, v2)


def test_create_comparisons_no_leakage():
    # 2 speakers: A has 2 test clips, B has 1 test clip
    ref_a = np.array([1.0, 0.0])
    ref_b = np.array([0.0, 1.0])

    references = {"spk_A": ref_a, "spk_B": ref_b}
    embeddings = {
        "spk_A": {
            "clip_2.wav": np.array([0.9, 0.1]),
            "clip_3.wav": np.array([0.95, 0.05]),
        },
        "spk_B": {
            "clip_2.wav": np.array([0.1, 0.9]),
        },
    }

    comparisons = create_comparisons(references, embeddings)

    # Total comparisons:
    # A ref vs A clips: 2 (SAME)
    # A ref vs B clips: 1 (DIFFERENT)
    # B ref vs B clips: 1 (SAME)
    # B ref vs A clips: 2 (DIFFERENT)
    # Total = 6
    assert len(comparisons) == 6

    same_pairs = [c for c in comparisons if c["actual"] == "SAME"]
    diff_pairs = [c for c in comparisons if c["actual"] == "DIFFERENT"]

    assert len(same_pairs) == 3
    assert len(diff_pairs) == 3

    # Ensure no comparison compares clip_1 against clip_1 as a test sample
    for c in comparisons:
        assert c["test_file"] != "clip_1.wav"
        if c["actual"] == "SAME":
            assert c["reference_speaker"] == c["test_speaker"]
        else:
            assert c["reference_speaker"] != c["test_speaker"]

