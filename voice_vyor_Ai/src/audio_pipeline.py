"""
Voice VyorAI Audio Pipeline
----------------------------
Workflow:
  Audio Input (Live mic OR --file offline WAV)
    -> DeepFilterNet3 Noise Reduction
    -> Silero Voice Activity Detection (VAD)
    -> Speech Chunking & Alignment
    -> SpeechBrain ECAPA-TDNN Speaker Verification
    -> YAMNet Environmental Acoustic Scene Classification
    -> Random Forest Replay / Phone Playback Detection
    -> Decision Engine Synthesis & Recruiter Alert Generation
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

import settings
from decision_engine import make_decision
from noise_reduction import enhance_file
from replay_detection import detect_replay
from segment_utils import merge_segments, split_into_chunks
from speaker_detection import CandidateSpeakerVerifier
from vad_detection import detect_speech
from yamnet_detection import classify_audio


def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice VyorAI - Voice Proctoring and Second-Speaker Detection Pipeline"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Path to an existing WAV file to evaluate offline (skips live microphone recording).",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=settings.DURATION,
        help=f"Recording duration in seconds for live microphone mode (default: {settings.DURATION}s).",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip DeepFilterNet noise reduction and process raw audio directly.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------
    # 1. Check reference enrollment recording
    # --------------------------------------------------
    if not settings.REFERENCE.exists():
        raise FileNotFoundError(
            f"Candidate reference voice not found: {settings.REFERENCE}\n"
            "Please run 'uv run python src/record_voice.py' first to enroll the candidate's voice."
        )

    settings.RAW.parent.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 2. Acquire interview audio (Offline file or Live mic)
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("                 VOICE VYORAI PIPELINE")
    print("=" * 60)

    if args.file:
        input_path = Path(args.file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input audio file not found: {input_path}")

        print(f"\n[Mode: Offline File Evaluation]")
        print(f"Loading: {input_path}")

        raw_audio, sr = sf.read(str(input_path), dtype="float32")
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.mean(axis=1)

        if sr != settings.SR:
            print(f"Resampling from {sr} Hz to {settings.SR} Hz...")
            raw_audio = resample_poly(raw_audio, settings.SR, sr).astype(np.float32)

        # Save normalized copy to settings.RAW for pipeline consistency
        sf.write(str(settings.RAW), raw_audio, settings.SR)
        print(f"Loaded {len(raw_audio) / settings.SR:.1f}s of audio.")

    else:
        print(f"\n[Mode: Live Microphone Recording]")
        print(f"Recording for {args.duration:.1f} seconds. Speak naturally into your mic...\n")

        raw_audio = sd.rec(
            int(args.duration * settings.SR),
            samplerate=settings.SR,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        raw_audio = raw_audio.squeeze()

        sf.write(str(settings.RAW), raw_audio, settings.SR)
        print(f"Raw recording saved to: {settings.RAW}")

        if float(np.max(np.abs(raw_audio))) < 0.005:
            print("[WARNING] Recording is almost completely silent. Please check your microphone.")

    # --------------------------------------------------
    # 3. Noise reduction (DeepFilterNet3)
    # --------------------------------------------------
    if args.no_enhance:
        print("\nSkipping noise reduction (--no-enhance requested).")
        enhanced_path = settings.RAW
    else:
        print("\nEnhancing audio with DeepFilterNet3...")
        try:
            enhanced_path = settings.ENHANCED
            enhance_file(settings.RAW, enhanced_path)
            print(f"Enhanced audio saved to: {enhanced_path}")
        except Exception as exc:
            print(f"[WARNING] Noise reduction encountered an issue ({exc}). Falling back to raw audio.")
            enhanced_path = settings.RAW

    # --------------------------------------------------
    # 4. Voice Activity Detection (Silero VAD)
    # --------------------------------------------------
    print("\nDetecting speech with Silero VAD...")
    detected_segments = detect_speech(enhanced_path)
    segments = merge_segments(detected_segments, settings.MERGE_GAP_SEC)
    chunks = split_into_chunks(segments, settings.CHUNK_SEC, settings.MIN_SEGMENT_SEC)

    print(f"Detected speech segments: {len(segments)}")
    for i, seg in enumerate(segments, 1):
        print(f"  Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s (Duration: {seg['end'] - seg['start']:.2f}s)")

    print(f"Total speaker verification chunks: {len(chunks)}")

    # --------------------------------------------------
    # 5. Speaker Verification (ECAPA-TDNN)
    # --------------------------------------------------
    print("\nLoading ECAPA-TDNN speaker verification model...")
    verifier = CandidateSpeakerVerifier(str(settings.REFERENCE))

    results = []
    print("\n========== SPEAKER VERIFICATION CHUNKS ==========")

    for seg in chunks:
        a = int(seg["start"] * settings.SR)
        b = min(int(seg["end"] * settings.SR), len(raw_audio))

        if b <= a or (b - a) < (settings.MIN_SEGMENT_SEC * settings.SR):
            continue

        chunk_audio = raw_audio[a:b]
        r = verifier.compare_waveform(chunk_audio)
        r["start"] = float(seg["start"])
        r["end"] = float(seg["end"])
        results.append(r)

        label = "CANDIDATE" if r["same_speaker"] else "DIFFERENT FROM CANDIDATE"
        print(f"  {r['start']:5.1f}s - {r['end']:5.1f}s | Cosine Score: {r['score']:.4f} | Result: {label}")

    # --------------------------------------------------
    # 6. Environmental Sound Classification (YAMNet)
    # --------------------------------------------------
    print("\nClassifying acoustic context with YAMNet...")
    try:
        yamnet_context = classify_audio(enhanced_path, top_k=5)
    except Exception as exc:
        print(f"YAMNet classification unavailable: {exc}")
        yamnet_context = []

    # --------------------------------------------------
    # 7. Replay Attack Detection (Task 6 Integration)
    # --------------------------------------------------
    print("\nEvaluating replay / phone speaker acoustic characteristics...")
    replay_results = []
    try:
        replay_results = detect_replay(raw_audio, segments, sr=settings.SR)
        for r in replay_results:
            tag = "PHONE REPLAY" if r["prediction"] == "phone_replay" else "LIVE"
            vote = r.get("model_vote_fraction", 0.0)
            print(f"  Window {r['start']:5.1f}s - {r['end']:5.1f}s | Prediction: {tag} (Confidence: {vote:.1%})")
    except Exception as exc:
        print(f"Replay detection unavailable or model uninitialized ({exc}). Proceeding without replay signal.")

    # --------------------------------------------------
    # 8. Decision Engine Synthesis
    # --------------------------------------------------
    decision = make_decision(
        speech_segments=segments,
        speaker_results=results,
        yamnet_predictions=yamnet_context,
        replay_predictions=replay_results,
    )

    # --------------------------------------------------
    # 9. Formatted Final Monitoring Report
    # --------------------------------------------------
    print("\n" + "=" * 65)
    print("                     INTERVIEW MONITORING REPORT")
    print("=" * 65)
    print(f"Decision:               {decision['decision']}")
    print(f"Event Type:             {decision['event_type']}")
    print(f"Severity:               {decision['severity']}")
    print(f"Candidate Match Chunks: {decision['metrics']['candidate_speaker_chunks']}")
    print(f"Different Voice Chunks: {decision['metrics']['different_speaker_chunks']}")
    print(f"Replay Windows Flagged: {decision['metrics']['replay_windows_flagged']}")

    if decision['metrics']['mean_similarity_score'] is not None:
        print(f"Mean Speaker Score:     {decision['metrics']['mean_similarity_score']:.4f}")

    if yamnet_context:
        print("\nTop Environmental Sounds (YAMNet):")
        for name, score in yamnet_context:
            print(f"  - {name}: {score:.3f}")

    if decision["alert_required"]:
        print("\n" + "!" * 65)
        print("                        INTERVIEWER ALERT")
        print("!" * 65)
        print(f"Alert Message: {decision['alert_message']}")
        print("!" * 65)
    else:
        print("\nStatus: Normal session. No interviewer alert triggered.")

    if decision["notes"]:
        print("\nMonitoring Notes:")
        for note in decision["notes"]:
            print(f"  * {note}")

    print("\n" + "-" * 65)
    print(f"NOTE: {decision['disclaimer']}")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    main()
