import csv
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
_model = None
_class_names = None


def load_yamnet():
    global _model, _class_names
    if _model is None:
        _model = hub.load(YAMNET_URL)
        class_map = _model.class_map_path().numpy().decode("utf-8")
        with open(class_map, newline="", encoding="utf-8") as f:
            _class_names = [row["display_name"] for row in csv.DictReader(f)]
    return _model


def classify_audio(audio_path, top_k: int = 5):
    """Return top_k [(sound_name, score), ...] for a 16 kHz file."""
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError("YAMNet input must be 16000 Hz mono audio.")
    model = load_yamnet()
    scores, _, _ = model(tf.convert_to_tensor(audio, dtype=tf.float32))
    mean_scores = np.mean(scores.numpy(), axis=0)
    indices = np.argsort(mean_scores)[::-1][:top_k]
    return [(_class_names[i], float(mean_scores[i])) for i in indices]
