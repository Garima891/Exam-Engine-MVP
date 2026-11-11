
"""DeepFilterNet3 enhancement with safe input normalization
and 16 kHz <-> model-rate resampling.
"""

import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

_cache = None

# Audio normalization settings
TARGET_PEAK = 0.5
MINIMUM_PEAK = 1e-4
MINIMUM_RMS = 1e-4


def _load_deepfilter():
    """Load DeepFilterNet once and reuse it."""
    global _cache

    if _cache is not None:
        return _cache

    # Compatibility shim for torchaudio versions
    try:
        import torchaudio

        try:
            from torchaudio._backend.common import AudioMetaData
        except ImportError:
            AudioMetaData = getattr(
                torchaudio, "AudioMetaData", object
            )

        backend = types.ModuleType("torchaudio.backend")
        backend.__path__ = []

        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = AudioMetaData

        sys.modules.setdefault("torchaudio.backend", backend)
        sys.modules.setdefault("torchaudio.backend.common", common)

        torchaudio.backend = sys.modules["torchaudio.backend"]

    except Exception:
        pass

    from df.enhance import init_df, enhance

    model, state, _ = init_df()

    _cache = (model, state, enhance)

    return _cache


def normalize_audio(audio):
    """Normalize audio safely without modifying the original file."""

    audio = np.asarray(audio, dtype=np.float32)

    # Reject invalid audio
    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid values.")

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    # Avoid amplifying silence or extremely quiet recordings
    if peak < MINIMUM_PEAK or rms < MINIMUM_RMS:
        raise ValueError(
            f"Audio level too low for safe normalization. "
            f"Peak={peak:.8f}, RMS={rms:.8f}"
        )

    # Peak normalization
    normalized = audio * (TARGET_PEAK / peak)

    return normalized.astype(np.float32)


def enhance_file(input_path, output_path):
    """Normalize and enhance audio using DeepFilterNet3."""

    # Read original recording
    audio, original_sr = sf.read(
        str(input_path),
        dtype="float32"
    )

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError("Expected mono or stereo audio.")

    # Normalize before enhancement
    audio = normalize_audio(audio)

    print(
        "Normalized input RMS:",
        float(np.sqrt(np.mean(audio ** 2)))
    )

    print(
        "Normalized input peak:",
        float(np.max(np.abs(audio)))
    )

    # Load DeepFilterNet3
    model, state, enhance = _load_deepfilter()

    model_sr = state.sr()

    # Resample to model sample rate
    if original_sr != model_sr:
        audio_model_rate = resample_poly(
            audio,
            model_sr,
            original_sr
        ).astype(np.float32)
    else:
        audio_model_rate = audio

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(
        np.ascontiguousarray(audio_model_rate)
    ).unsqueeze(0)

    # Run enhancement
    with torch.no_grad():
        enhanced = enhance(model, state, tensor)

    result = enhanced.squeeze().cpu().numpy()

    result = np.asarray(result, dtype=np.float32)

    if result.ndim != 1 or result.size == 0:
        raise ValueError("DeepFilterNet returned invalid audio.")

    if not np.all(np.isfinite(result)):
        raise ValueError("Enhanced audio contains invalid values.")

    # Convert back to original sample rate
    if model_sr != original_sr:
        result = resample_poly(
            result,
            original_sr,
            model_sr
        ).astype(np.float32)

    # Save enhanced recording
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        str(output_path),
        result,
        original_sr
    )

    print(
        "Enhanced output RMS:",
        float(np.sqrt(np.mean(result ** 2)))
    )

    print(
        "Enhanced output peak:",
        float(np.max(np.abs(result)))
    )

    return str(output_path)


if __name__ == "__main__":
    import settings

    print(
        enhance_file(
            settings.CANDIDATE_TEST,
            settings.ENHANCED_CANDIDATE
        )
    )
