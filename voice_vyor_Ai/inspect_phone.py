import sys
import os

# Suppress oneDNN and TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Compatibility shim for Python 3.12+ environments missing setuptools
try:
    import pkg_resources
except ImportError:
    import types, re
    pkg_resources = types.ModuleType("pkg_resources")
    def parse_version(v):
        return tuple(int(x) for x in re.findall(r"\d+", str(v)))
    pkg_resources.parse_version = parse_version
    sys.modules["pkg_resources"] = pkg_resources

import numpy as np
import tensorflow as tf

# Reuse the YAMNet instance and classes already loaded in detector
from detector import yam, classes, load_wav, SR

def diagnose(wav_path):
    print(f"\nLoading {wav_path}...")
    w = load_wav(wav_path, normalize=False)

    step = int(0.5 * SR)
    win_len = int(1.5 * SR)
    print("\nScanning top predicted sound classes (every 0.5s):")
    print(f"{'Time':<8} | {'Top 3 Detected Audio Classes & Probabilities'}")
    print("-" * 75)

    for s in range(0, len(w) - win_len + 1, step):
        chunk = w[s:s + win_len]
        t = s / SR
        # Safe gain boost if audio is quiet
        rms = np.sqrt(np.mean(chunk ** 2)) + 1e-9
        if rms < 0.02:
            chunk = chunk * (0.04 / rms)
        scores, _, _ = yam(tf.convert_to_tensor(chunk, dtype=tf.float32))
        mean_scores = scores.numpy().mean(axis=0)
        top3_idx = np.argsort(mean_scores)[-3:][::-1]
        top3 = [f"{classes[i]} ({mean_scores[i]:.2f})" for i in top3_idx if mean_scores[i] > 0.05]
        if top3:
            print(f"{t:5.1f}s   | {', '.join(top3)}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test_phone.wav"
    diagnose(path)
