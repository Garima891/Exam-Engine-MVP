
import sounddevice as sd
import soundfile as sf
import numpy as np

from pathlib import Path
from datetime import datetime

import settings


ROOT = settings.PROJECT_ROOT / "data" / "replay_dataset"

LIVE_DIR = ROOT / "live"
REPLAY_DIR = ROOT / "phone_replay"

DURATION = 6
SAMPLE_RATE = settings.SR


def record_sample():

    print("\n========== REPLAY DATA COLLECTION ==========")

    print("Choose recording type:")
    print("1. Live human speech")
    print("2. Phone speaker playback")
    print("q. Exit")

    choice = input("Your choice: ").strip().lower()

    if choice == "q":
        return False

    if choice == "1":
        label = "live"
        folder = LIVE_DIR

    elif choice == "2":
        label = "phone_replay"
        folder = REPLAY_DIR

    else:
        print("Invalid choice.")
        return True

    speaker = input(
        "Who is speaking? (candidate / other): "
    ).strip().lower()

    if speaker not in ("candidate", "other"):
        print("Enter candidate or other.")
        return True

    if label == "live":
        print("\nSpeak directly near the laptop microphone.")

    else:
        print(
            "\nPlay a recorded voice through your phone speaker "
            "near the laptop microphone."
        )

    input("\nPress ENTER when ready to record...")

    print(f"\nRecording for {DURATION} seconds...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    if np.max(np.abs(audio)) < 0.005:
        print("Recording is almost silent. Please try again.")
        return True

    folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    filename = f"{speaker}_{timestamp}.wav"

    filepath = folder / filename

    sf.write(filepath, audio, SAMPLE_RATE)

    print("\nSaved:", filepath)
    print("Class:", label)
    print("Speaker:", speaker)

    return True


def main():

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    print("Voice VyorAI - Replay Dataset Collector")
    print("Sample duration:", DURATION, "seconds")
    print("Sample rate:", SAMPLE_RATE)

    while True:

        keep_running = record_sample()

        if not keep_running:
            break

    print("\nDataset collection stopped.")

    print("Live recordings:", len(list(LIVE_DIR.glob("*.wav"))))
    print("Phone replay recordings:", len(list(REPLAY_DIR.glob("*.wav"))))


if __name__ == "__main__":
    main()
