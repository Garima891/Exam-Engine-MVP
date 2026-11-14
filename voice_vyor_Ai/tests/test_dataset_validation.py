"""
Tests for Speaker Evaluation Dataset Validation (Task 1 & Task 7)
"""

from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

from evaluate_speaker_model import (
    validate_audio_file,
    validate_evaluation_dataset,
    DatasetValidationError,
)


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for generating test datasets."""
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def _create_wav(path: Path, duration_sec: float = 2.0, sr: int = 16000, amplitude: float = 0.1):
    """Helper to create a synthetic valid WAV file for testing validation logic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine wave tone
    waveform = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(path), waveform, sr)


def test_validate_audio_file_valid(tmp_path: Path):
    wav_path = tmp_path / "test.wav"
    _create_wav(wav_path, duration_sec=2.5)
    duration, sr = validate_audio_file(wav_path, min_duration=1.0)
    assert pytest.approx(duration, 0.01) == 2.5
    assert sr == 16000


def test_validate_audio_file_missing(tmp_path: Path):
    missing_path = tmp_path / "missing.wav"
    with pytest.raises(DatasetValidationError, match="File does not exist"):
        validate_audio_file(missing_path)


def test_validate_audio_file_empty(tmp_path: Path):
    empty_path = tmp_path / "empty.wav"
    empty_path.write_bytes(b"")
    with pytest.raises(DatasetValidationError, match="Audio file is empty"):
        validate_audio_file(empty_path)


def test_validate_audio_file_corrupted(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.wav"
    corrupt_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt invalid audio data bytes")
    with pytest.raises(DatasetValidationError, match="Corrupted or unreadable"):
        validate_audio_file(corrupt_path)


def test_validate_audio_file_too_short(tmp_path: Path):
    short_path = tmp_path / "short.wav"
    _create_wav(short_path, duration_sec=0.5)
    with pytest.raises(DatasetValidationError, match="Audio recording too short"):
        validate_audio_file(short_path, min_duration=1.0)


def test_validate_dataset_missing_folder(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(DatasetValidationError, match="Evaluation folder not found"):
        validate_evaluation_dataset(missing_dir)


def test_validate_dataset_insufficient_speakers(temp_dataset_dir: Path):
    # Only 1 speaker folder
    spk1 = temp_dataset_dir / "speaker_01"
    _create_wav(spk1 / "clip_1.wav", 2.0)
    _create_wav(spk1 / "clip_2.wav", 2.0)

    with pytest.raises(DatasetValidationError, match="Insufficient speaker folders"):
        validate_evaluation_dataset(temp_dataset_dir, min_speakers=2)


def test_validate_dataset_missing_clip_1(temp_dataset_dir: Path):
    # Two speakers, but speaker_01 lacks clip_1.wav
    spk1 = temp_dataset_dir / "speaker_01"
    _create_wav(spk1 / "other_clip.wav", 2.0)
    _create_wav(spk1 / "test_clip.wav", 2.0)

    spk2 = temp_dataset_dir / "speaker_02"
    _create_wav(spk2 / "clip_1.wav", 2.0)
    _create_wav(spk2 / "clip_2.wav", 2.0)

    with pytest.raises(DatasetValidationError, match="Missing reference recording 'clip_1.wav'"):
        validate_evaluation_dataset(temp_dataset_dir)


def test_validate_dataset_no_test_clips(temp_dataset_dir: Path):
    # Speaker has clip_1.wav but no second recording
    spk1 = temp_dataset_dir / "speaker_01"
    _create_wav(spk1 / "clip_1.wav", 2.0)

    spk2 = temp_dataset_dir / "speaker_02"
    _create_wav(spk2 / "clip_1.wav", 2.0)
    _create_wav(spk2 / "clip_2.wav", 2.0)

    with pytest.raises(DatasetValidationError, match="has only 1 WAV file"):
        validate_evaluation_dataset(temp_dataset_dir)


def test_validate_dataset_valid_multi_speaker(temp_dataset_dir: Path):
    # Valid 2-speaker setup
    spk1 = temp_dataset_dir / "speaker_01"
    _create_wav(spk1 / "clip_1.wav", 2.0)
    _create_wav(spk1 / "clip_2.wav", 2.0)

    spk2 = temp_dataset_dir / "speaker_02"
    _create_wav(spk2 / "clip_1.wav", 2.0)
    _create_wav(spk2 / "clip_2.wav", 2.0)
    _create_wav(spk2 / "clip_3.wav", 2.0)

    manifest = validate_evaluation_dataset(temp_dataset_dir)
    assert len(manifest) == 2
    assert manifest["speaker_01"]["reference"].name == "clip_1.wav"
    assert len(manifest["speaker_01"]["test_files"]) == 1
    assert len(manifest["speaker_02"]["test_files"]) == 2

