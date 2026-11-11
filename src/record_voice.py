import numpy as np
import sounddevice as sd
import soundfile as sf
import settings

settings.ENROLL_DIR.mkdir(parents=True, exist_ok=True)


def record_audio(path, description: str):
    print(f"\n{description}")
    input("Press ENTER when ready...")
    print(f"Recording for {settings.DURATION} seconds. Speak naturally.")
    audio = sd.rec(int(settings.DURATION * settings.SR), samplerate=settings.SR,
                   channels=1, dtype="float32")
    sd.wait()
    if float(np.max(np.abs(audio))) < 0.005:
        print("WARNING: recording is almost silent - check your microphone and record again.")
    sf.write(path, audio, settings.SR)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Record only with the speaker's informed consent.")
    print("Tip: use the same microphone and a quiet room for the enrollment recording.")
    record_audio(settings.REFERENCE, "Candidate enrollment voice")
    record_audio(settings.CANDIDATE_TEST, "Candidate test voice")
    record_audio(settings.SECOND_PERSON, "Second speaker test voice")
