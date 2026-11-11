
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from datetime import datetime
import re

SAMPLE_RATE = 16000
DURATION = 6

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LIVE_DIR = PROJECT_ROOT / "data" / "replay_holdout" / "live"
REPLAY_DIR = PROJECT_ROOT / "data" / "replay_holdout" / "phone_replay"

LIVE_DIR.mkdir(parents=True, exist_ok=True)
REPLAY_DIR.mkdir(parents=True, exist_ok=True)


def clean_label(value):
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]", "_", value)
    return value.strip("_")


def record_audio():
    print(f"\nRecording for {DURATION} seconds...")
    print("Speak clearly or play the phone recording now.")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()
    return audio


def main():
    print("\nVOICE VYORAI - HOLDOUT DATA COLLECTION")
    print("=" * 45)

    while True:
        print("\nChoose recording type:")
        print("1. Live human speech")
        print("2. Phone speaker playback")
        print("q. Exit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            print("Collection stopped.")
            break

        if choice not in ("1", "2"):
            print("Invalid choice. Enter 1, 2, or q.")
            continue

        speaker_id = clean_label(input("Speaker ID (example: speaker_03): "))
        device_id = clean_label(input("Device ID (example: phone_a): "))

        if not speaker_id or not device_id:
            print("Speaker ID and device ID cannot be empty.")
            continue

        if choice == "1":
            label = "live"
            save_dir = LIVE_DIR
        else:
            label = "phone_replay"
            save_dir = REPLAY_DIR

        input("\nPress Enter when ready to record...")

        try:
            audio = record_audio()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{speaker_id}__{device_id}__{timestamp}.wav"
            output_path = save_dir / filename

            sf.write(output_path, audio, SAMPLE_RATE)

            print(f"\nSaved successfully: {output_path}")

        except Exception as error:
            print(f"\nRecording failed: {error}")
            print("Check your microphone and sounddevice installation.")


if __name__ == "__main__":
    main()
