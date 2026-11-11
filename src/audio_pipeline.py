
"""Record -> enhance -> VAD -> speaker verification -> decision.

Important:
- DeepFilterNet-enhanced audio is used for VAD and YAMNet.
- Original raw audio is used for ECAPA speaker comparison.
- A possible additional speaker alert requires multiple different-speaker
  chunks. A low similarity score alone is not proof of another speaker.
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from replay_detection import detect_replay
import settings

from noise_reduction import enhance_file
from vad_detection import detect_speech
from speaker_detection import CandidateSpeakerVerifier
from yamnet_detection import classify_audio
from decision_engine import make_decision
from segment_utils import merge_segments, split_into_chunks


def main():

    # --------------------------------------------------
    # 1. Check reference recording
    # --------------------------------------------------

    if not settings.REFERENCE.exists():
        raise FileNotFoundError(
            f"Candidate reference not found: {settings.REFERENCE}\n"
            "Please run record_voice.py first."
        )

    settings.RAW.parent.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Record interview audio
    # --------------------------------------------------

    print("\n========== VOICE VYORAI ==========")
    print(f"Recording for {settings.DURATION} seconds.")
    print("Speak naturally during the recording.\n")

    audio = sd.rec(
    int(settings.DURATION * settings.SR),
    samplerate=settings.SR,
    channels=1,
    dtype="float32",
    device=1
)

    sd.wait()

    sf.write(settings.RAW, audio, settings.SR)

    print("Raw recording saved:", settings.RAW)

    if float(np.max(np.abs(audio))) < 0.005:
        print("WARNING: Recording is almost silent. Check your microphone.")

    # --------------------------------------------------
    # 3. Noise reduction
    # --------------------------------------------------

    # --------------------------------------------------
# 3. Noise reduction
# --------------------------------------------------

    print("\nEnhancing audio...")

    enhance_file(
    settings.RAW,
    settings.ENHANCED,
)

    print("Enhanced audio saved:", settings.ENHANCED)

    # --------------------------------------------------
    # 4. Voice Activity Detection
    # --------------------------------------------------

    print("\nDetecting speech...")

    detected_segments = detect_speech(settings.RAW)
    segments = merge_segments(
        detected_segments,
        settings.MERGE_GAP_SEC,
    )

    chunks = split_into_chunks(
        segments,
        settings.CHUNK_SEC,
        settings.MIN_SEGMENT_SEC,
    )

    print("\nSpeech segments:")

    for seg in segments:
        print(
            f"Start: {seg['start']:.2f}s | "
            f"End: {seg['end']:.2f}s"
        )

    print(f"\nTotal speech segments: {len(segments)}")
    print(f"Total speaker chunks: {len(chunks)}")

    # --------------------------------------------------
    # 5. Load original audio
    # --------------------------------------------------

    raw_audio, raw_sr = sf.read(
        settings.RAW,
        dtype="float32",
    )

    if raw_audio.ndim == 2:
        raw_audio = raw_audio.mean(axis=1)

    if raw_sr != settings.SR:
        raise ValueError(
            f"Expected raw audio at {settings.SR} Hz, "
            f"but received {raw_sr} Hz."
        )

    # --------------------------------------------------
    # 6. ECAPA speaker verification
    # --------------------------------------------------

    print("\nLoading ECAPA speaker verification model...")

    verifier = CandidateSpeakerVerifier(
        str(settings.REFERENCE)
    )

    results = []

    print("\n========== SPEAKER COMPARISON ==========")

    for seg in chunks:

        # Convert segment timestamps into sample positions
        a = int(seg["start"] * settings.SR)
        b = min(
            int(seg["end"] * settings.SR),
            len(raw_audio),
        )

        # Skip chunks that are too short
        if b <= a:
            continue

        if b - a < settings.MIN_SEGMENT_SEC * settings.SR:
            print(
                f"Skipping short chunk: "
                f"{seg['start']:.1f}-{seg['end']:.1f}s"
            )
            continue

        # Extract the original, unenhanced audio
        audio_chunk = raw_audio[a:b]

        # Compare against enrolled candidate voice
        r = verifier.compare_waveform(audio_chunk)

        # Store timestamps
        r["start"] = float(seg["start"])
        r["end"] = float(seg["end"])

        results.append(r)

        # Display individual comparison
        label = (
            "CANDIDATE"
            if r["same_speaker"]
            else "DIFFERENT FROM REFERENCE"
        )

        print(
            f"{r['start']:.1f}-{r['end']:.1f}s | "
            f"Score: {r['score']:.3f} | "
            f"{label}"
        )

    # --------------------------------------------------
    # 7. YAMNet sound classification
    # --------------------------------------------------

    print("\nClassifying sound context...")

    try:
        context = classify_audio(settings.ENHANCED)

    except Exception as exc:
        print("YAMNet classification unavailable:", exc)
        context = []

    # --------------------------------------------------
    # 8. Final decision
    # --------------------------------------------------

    result = make_decision(
        speech_segments=segments,
        speaker_results=results,
        yamnet_predictions=context,
    )

    # --------------------------------------------------
    # 9. Display interview result
    # --------------------------------------------------

    print("\n")
    print("=" * 55)
    print("          INTERVIEW AUDIO RESULT")
    print("=" * 55)

    print("\nDecision:", result["decision"])
    print("Event type:", result["event_type"])

    print("\nSpeaker statistics:")
    print(
        "Candidate voice chunks:",
        result["candidate_speaker_chunks"],
    )
    print(
        "Different-speaker chunks:",
        result["different_speaker_chunks"],
    )
    print(
        "Total comparisons:",
        result["speaker_comparisons"],
    )

    if result["mean_similarity"] is not None:
        print(
            f"Mean similarity: "
            f"{result['mean_similarity']:.3f}"
        )

    # --------------------------------------------------
    # 10. Interviewer alert
    # --------------------------------------------------

    if result["alert_required"]:

        print("\n" + "!" * 55)
        print("             INTERVIEWER ALERT")
        print("!" * 55)

        print(result["alert_message"])

        print("\nEvent:", result["event_type"])

        print(
            "Different-speaker chunks:",
            result["different_speaker_chunks"],
        )

        print(
            "\nIMPORTANT: This is an automated indication, "
            "not proof of another speaker. Human review is required."
        )

        print("!" * 55)

    else:
        print("\nNo interviewer alert required.")

    # --------------------------------------------------
    # 11. Display YAMNet context
    # --------------------------------------------------

    if context:
        print("\nSound context:")

        for name, score in context:
            print(f"{name}: {score:.3f}")

    # --------------------------------------------------
    # 12. Display additional notes
    # --------------------------------------------------

    if result["notes"]:
        print("\nAdditional notes:")

        for note in result["notes"]:
            print("-", note)

    print("\n========== PROCESS COMPLETED ==========\n")


if __name__ == "__main__":
    main()
