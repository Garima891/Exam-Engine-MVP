
from pathlib import Path
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data" / "replay_dataset"

FOLDERS = {
    "LIVE": ROOT / "live",
    "PHONE REPLAY": ROOT / "phone_replay",
}

def audit_folder(label, folder):
    print(f"\n========== {label} ==========")

    if not folder.exists():
        print("Folder not found:", folder)
        return

    files = list(folder.glob("*.wav"))

    print("Total files:", len(files))

    counts = {"candidate": 0, "other": 0, "unknown": 0}

    for file in files:
        name = file.stem.lower()

        if name.startswith("candidate_"):
            speaker = "candidate"
        elif name.startswith("other_"):
            speaker = "other"
        else:
            speaker = "unknown"

        counts[speaker] += 1

        try:
            audio, sr = sf.read(file)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            duration = len(audio) / sr
            peak = float(np.max(np.abs(audio))) if len(audio) else 0
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0
            clipped = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) else 0

            status = "OK"

            if sr != 16000:
                status = "CHECK SAMPLE RATE"
            if duration < 3:
                status = "CHECK DURATION"
            if rms < 0.005:
                status = "TOO QUIET"
            if clipped > 0.01:
                status = "POSSIBLE CLIPPING"

            print(
                f"{file.name} | {speaker} | "
                f"{duration:.1f}s | {sr}Hz | "
                f"RMS {rms:.4f} | Peak {peak:.3f} | {status}"
            )

        except Exception as error:
            print(f"{file.name} | ERROR: {error}")

    print("\nSpeaker counts:", counts)


def main():
    print("VOICE VYORAI - REPLAY DATASET AUDIT")

    for label, folder in FOLDERS.items():
        audit_folder(label, folder)

    print("\n========== AUDIT COMPLETE ==========")


if __name__ == "__main__":
    main()
