from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.fetching import LocalStrategy
import settings


def _load_mono_16k(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != settings.SR:
        audio = resample_poly(audio, settings.SR, sr).astype(np.float32)
    return audio


class CandidateSpeakerVerifier:
    """Compares audio against the enrolled candidate using ECAPA-TDNN embeddings.
    The reference embedding is computed ONCE (the old code re-encoded it for every segment).
    """

    def __init__(self, reference_path, savedir=None, threshold=settings.SPEAKER_THRESHOLD):
        self.reference = Path(reference_path)
        if not self.reference.exists():
            raise FileNotFoundError(self.reference)
        savedir = savedir or str(settings.MODELS_DIR / "ecapa-voxceleb")
        self.model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )
        self.threshold = threshold
        self.reference_embedding = self._embed(_load_mono_16k(self.reference))

    def _embed(self, audio_1d):
        wav = torch.from_numpy(np.ascontiguousarray(audio_1d)).float().unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode_batch(wav)
        return emb.squeeze()

    def compare_waveform(self, audio_1d):
        """audio_1d: numpy array, mono, 16 kHz."""
        emb = self._embed(audio_1d)
        score = torch.nn.functional.cosine_similarity(
            self.reference_embedding, emb, dim=0).item()
        return {"score": float(score), "same_speaker": bool(score >= self.threshold)}

    def compare(self, audio_path):
        return self.compare_waveform(_load_mono_16k(audio_path))
