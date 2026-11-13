"""
Decision Engine for Voice VyorAI
---------------------------------
Synthesizes multi-model acoustic evidence:
1. Silero VAD (human speech vs silence/noise)
2. ECAPA-TDNN (candidate voice vs different speaker)
3. YAMNet (environmental acoustic scene / background sounds)
4. Random Forest Replay Classifier (live voice vs phone speaker playback)

Guiding Principles:
- Background noise alone MUST NOT generate an alert.
- Alerts require repeated evidence (configurable min_different chunks).
- Distinguishes candidate speech, second speaker, replay spoofing, and environmental sounds.
- Automated indications are flags for human recruiter review, NOT proof of identity or cheating.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import settings


def make_decision(
    speech_segments: List[Dict[str, float]],
    speaker_results: List[Dict[str, Any]],
    yamnet_predictions: Optional[List[tuple]] = None,
    replay_predictions: Optional[List[Dict[str, Any]]] = None,
    min_different: int = settings.MIN_DIFFERENT_CHUNKS,
    replay_threshold: float = getattr(settings, "REPLAY_CONFIDENCE_THRESHOLD", 0.60),
) -> Dict[str, Any]:
    """
    Synthesize multi-modal audio signals into a structured monitoring decision.

    Args:
        speech_segments: VAD detected speech time intervals [{'start': float, 'end': float}, ...]
        speaker_results: ECAPA verification chunks [{'start': f, 'end': f, 'score': f, 'same_speaker': bool}]
        yamnet_predictions: Top environmental acoustic classes [('Sound Name', score), ...]
        replay_predictions: Replay detection windows [{'start': f, 'end': f, 'prediction': str, 'model_vote_fraction': f}]
        min_different: Minimum different-speaker chunks needed to raise a SECOND_SPEAKER alert
        replay_threshold: Minimum RF confidence fraction to trigger replay warning

    Returns:
        Structured monitoring event dictionary.
    """
    notes: List[str] = []
    events: List[Dict[str, Any]] = []

    # 1. Speaker Verification Statistics
    different = sum(not r["same_speaker"] for r in speaker_results)
    same = sum(r["same_speaker"] for r in speaker_results)
    mean_score = (
        sum(r["score"] for r in speaker_results) / len(speaker_results)
        if speaker_results
        else None
    )

    # 2. Replay Detection Statistics
    replay_count = 0
    replay_windows = []
    if replay_predictions:
        for r in replay_predictions:
            is_replay = r.get("prediction") == "phone_replay"
            conf = r.get("model_vote_fraction", 0.5)
            if is_replay and conf >= replay_threshold:
                replay_count += 1
                replay_windows.append({
                    "start": r.get("start"),
                    "end": r.get("end"),
                    "confidence": round(float(conf), 3),
                })

    # 3. Environmental Sound Context (YAMNet)
    top_sounds = [str(name).lower() for name, _ in (yamnet_predictions or [])]
    phone_or_alarm_sounds = any(
        kw in name
        for name in top_sounds
        for kw in ["telephone", "ringtone", "bell", "alarm", "siren"]
    )
    speech_in_yamnet = any(
        kw in name
        for name in top_sounds
        for kw in ["speech", "conversation", "narration", "whispering"]
    )

    # 4. Core Decision Logic
    if not speech_segments:
        # Case A: No speech detected by VAD
        if phone_or_alarm_sounds:
            decision = "NO SPEECH DETECTED - BACKGROUND PHONE/ALARM SOUND PRESENT"
            event_type = "ENVIRONMENTAL_SOUND"
            severity = "LOW"
            notes.append("YAMNet detected phone/alarm acoustic context while no human speech was detected.")
        else:
            decision = "NO SPEECH DETECTED"
            event_type = "NO_SPEECH"
            severity = "INFO"

    elif not speaker_results:
        # Case B: Speech detected but segments too short for reliable ECAPA embedding
        decision = "SHORT SPEECH DETECTED - INSUFFICIENT BIOMETRIC EVIDENCE"
        event_type = "REVIEW_REQUIRED"
        severity = "LOW"
        notes.append("Detected speech segments were shorter than the minimum reliable duration.")

    elif replay_count >= 1:
        # Case C: Replay spoofing detected (phone speaker playback)
        decision = "POSSIBLE PHONE REPLAY / PRE-RECORDED AUDIO DETECTED - HUMAN REVIEW REQUIRED"
        event_type = "POSSIBLE_REPLAY_SPOOFING"
        severity = "MEDIUM"
        notes.append(
            f"Acoustic spectral signatures match phone speaker playback in {replay_count} window(s) "
            f"(confidence >= {replay_threshold*100:.0f}%)."
        )

    elif different >= min_different and different > same:
        # Case D: Consistent mismatch with candidate enrollment
        decision = "POSSIBLE ADDITIONAL SPEAKER DETECTED - HUMAN REVIEW REQUIRED"
        event_type = "POSSIBLE_SECOND_SPEAKER"
        severity = "HIGH"
        notes.append(
            f"Detected {different} segment(s) differing from the candidate's enrolled reference voice."
        )

    elif different > 0:
        # Case E: Inconsistent / mixed speaker evidence
        decision = "MIXED SPEAKER EVIDENCE - HUMAN REVIEW REQUIRED"
        event_type = "REVIEW_REQUIRED"
        severity = "MEDIUM"
        notes.append(
            f"Evidence is mixed: {different} different chunk(s) and {same} matching candidate chunk(s)."
        )

    else:
        # Case F: Candidate voice confirmed
        decision = "CANDIDATE VOICE DETECTED"
        event_type = "CANDIDATE_VOICE"
        severity = "INFO"

    # Cross-check VAD with YAMNet for speech confirmation
    if speech_segments and yamnet_predictions and not speech_in_yamnet:
        notes.append(
            "Silero VAD detected speech, but YAMNet did not identify human speech among top classes. "
            "Noise or atypical voice characteristics may be present."
        )

    # Secondary alert for background ringtones if candidate is speaking
    if phone_or_alarm_sounds and event_type == "CANDIDATE_VOICE":
        notes.append("Background phone ringtone or alarm detected during candidate speech.")

    # 5. Alert Construction
    alert_required = event_type in (
        "POSSIBLE_SECOND_SPEAKER",
        "POSSIBLE_REPLAY_SPOOFING",
        "REVIEW_REQUIRED",
    )

    if event_type == "POSSIBLE_SECOND_SPEAKER":
        alert_message = (
            "Potential additional speaker detected during the interview session. "
            "Please review the audio evidence."
        )
    elif event_type == "POSSIBLE_REPLAY_SPOOFING":
        alert_message = (
            "Acoustic characteristics suggest potential pre-recorded or phone speaker playback. "
            "Human review recommended."
        )
    elif event_type == "REVIEW_REQUIRED":
        alert_message = (
            "Speech detected, but speaker identity evidence is ambiguous or incomplete. "
            "Please review the audio recording."
        )
    else:
        alert_message = None

    # Construct structured monitoring event
    structured_event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "decision": decision,
        "severity": severity,
        "alert_required": alert_required,
        "alert_message": alert_message,
        "metrics": {
            "speech_segment_count": len(speech_segments),
            "speaker_comparisons": len(speaker_results),
            "candidate_speaker_chunks": same,
            "different_speaker_chunks": different,
            "mean_similarity_score": round(mean_score, 4) if mean_score is not None else None,
            "replay_windows_flagged": replay_count,
        },
        "replay_windows": replay_windows,
        "yamnet_top_classes": yamnet_predictions or [],
        "notes": notes,
        "disclaimer": (
            "Automated monitoring indicator intended solely for recruiter review. "
            "Does not constitute legal proof of identity, fraud, or cheating."
        ),
    }

    return structured_event
