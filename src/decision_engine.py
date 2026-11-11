
import settings


def make_decision(
    speech_segments,
    speaker_results,
    yamnet_predictions=None,
    min_different=settings.MIN_DIFFERENT_CHUNKS,
):
    """Decide whether speech may belong to an additional speaker."""

    notes = []
    mean_score = None

    different = sum(
        not result["same_speaker"] for result in speaker_results
    )
    same = sum(
        result["same_speaker"] for result in speaker_results
    )

    if speaker_results:
        mean_score = sum(
            result["score"] for result in speaker_results
        ) / len(speaker_results)

    if not speech_segments:
        decision = "NO SPEECH DETECTED"
        event_type = "NO_ALERT"

    elif not speaker_results:
        decision = "SPEECH DETECTED - INSUFFICIENT SPEAKER EVIDENCE"
        event_type = "REVIEW_REQUIRED"

    elif different >= min_different and different > same:
        decision = "POSSIBLE ADDITIONAL SPEAKER - HUMAN REVIEW REQUIRED"
        event_type = "POSSIBLE_SECOND_SPEAKER"

    elif different > 0:
        decision = "MIXED SPEAKER EVIDENCE - HUMAN REVIEW REQUIRED"
        event_type = "REVIEW_REQUIRED"

    else:
        decision = "CANDIDATE VOICE DETECTED"
        event_type = "NO_ALERT"

    # YAMNet provides sound context, not proof of speaker identity.
    if speech_segments and yamnet_predictions:
        names = [str(name).lower() for name, _ in yamnet_predictions]

        if not any(
            "speech" in name
            or "conversation" in name
            or "narration" in name
            for name in names
        ):
            notes.append(
                "VAD detected speech, but YAMNet did not list speech "
                "among its top sounds. Manual review recommended."
            )

    alert_required = event_type in (
        "POSSIBLE_SECOND_SPEAKER",
        "REVIEW_REQUIRED",
    )

    # Different messages for uncertain evidence and possible second speaker.
    if event_type == "REVIEW_REQUIRED":
        alert_message = (
            "Speech detected, but speaker identity is uncertain. "
            "Please review the audio evidence."
        )

    elif event_type == "POSSIBLE_SECOND_SPEAKER":
        alert_message = (
            "Possible additional speaker detected. "
            "Please review the audio evidence."
        )

    else:
        alert_message = None

    return {
        "decision": decision,
        "event_type": event_type,
        "alert_required": alert_required,
        "alert_message": alert_message,
        "different_speaker_chunks": different,
        "candidate_speaker_chunks": same,
        "speech_segment_count": len(speech_segments),
        "speaker_comparisons": len(speaker_results),
        "mean_similarity": mean_score,
        "yamnet_context": yamnet_predictions or [],
        "notes": notes,
    }
