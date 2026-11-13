"""
Robustness and Edge Case Tests for Decision Engine (Task 6 & Task 7)
"""

from decision_engine import make_decision


def test_scenario_no_speech():
    """Scenario: Silence or quiet room (no speech segments detected)."""
    result = make_decision(
        speech_segments=[],
        speaker_results=[],
        yamnet_predictions=[("Silence", 0.98), ("Inside, small room", 0.02)],
    )
    assert result["event_type"] == "NO_SPEECH"
    assert result["decision"] == "NO SPEECH DETECTED"
    assert not result["alert_required"]
    assert "Automated monitoring indicator" in result["disclaimer"]


def test_scenario_environmental_noise_only():
    """Scenario: Background fan or traffic noise with no human speech (MUST NOT trigger false cheating alerts)."""
    result = make_decision(
        speech_segments=[],
        speaker_results=[],
        yamnet_predictions=[("Motor vehicle (road)", 0.85), ("Traffic noise", 0.60), ("White noise", 0.40)],
    )
    assert result["event_type"] == "NO_SPEECH"
    assert not result["alert_required"]


def test_scenario_background_alarm_without_speech():
    """Scenario: Background ringtone or telephone bell while candidate is silent."""
    result = make_decision(
        speech_segments=[],
        speaker_results=[],
        yamnet_predictions=[("Telephone bell ringing", 0.90), ("Silence", 0.40)],
    )
    assert result["event_type"] == "ENVIRONMENTAL_SOUND"
    assert not result["alert_required"]
    assert any("YAMNet detected phone/alarm" in note for note in result["notes"])


def test_scenario_short_segments_insufficient_evidence():
    """Scenario: Speech detected by VAD but too brief to form valid ECAPA speaker chunks (< MIN_SEGMENT_SEC)."""
    result = make_decision(
        speech_segments=[{"start": 1.0, "end": 1.4}],  # 0.4s clip
        speaker_results=[],  # No chunks survived minimum duration filter
        yamnet_predictions=[("Speech", 0.80)],
    )
    assert result["event_type"] == "REVIEW_REQUIRED"
    assert result["severity"] == "LOW"
    assert result["alert_required"]
    assert "insufficient" in result["decision"].lower() or "short" in result["decision"].lower()


def test_scenario_candidate_voice_clear():
    """Scenario: Enrolled candidate speaking clearly throughout the interview."""
    speaker_results = [
        {"start": 0.0, "end": 4.0, "score": 0.65, "same_speaker": True},
        {"start": 4.0, "end": 8.0, "score": 0.72, "same_speaker": True},
        {"start": 8.0, "end": 12.0, "score": 0.60, "same_speaker": True},
    ]
    result = make_decision(
        speech_segments=[{"start": 0.0, "end": 12.0}],
        speaker_results=speaker_results,
        yamnet_predictions=[("Speech", 0.95), ("Narration, monologue", 0.05)],
    )
    assert result["event_type"] == "CANDIDATE_VOICE"
    assert not result["alert_required"]
    assert result["metrics"]["candidate_speaker_chunks"] == 3
    assert result["metrics"]["different_speaker_chunks"] == 0


def test_scenario_different_speaker_alert():
    """Scenario: Unauthorized second person speaking during the interview (repeated mismatched chunks)."""
    speaker_results = [
        {"start": 0.0, "end": 4.0, "score": 0.12, "same_speaker": False},
        {"start": 4.0, "end": 8.0, "score": 0.08, "same_speaker": False},
        {"start": 8.0, "end": 12.0, "score": 0.15, "same_speaker": False},
    ]
    result = make_decision(
        speech_segments=[{"start": 0.0, "end": 12.0}],
        speaker_results=speaker_results,
        yamnet_predictions=[("Speech", 0.90), ("Conversation", 0.80)],
        min_different=2,
    )
    assert result["event_type"] == "POSSIBLE_SECOND_SPEAKER"
    assert result["severity"] == "HIGH"
    assert result["alert_required"]
    assert "additional speaker" in result["alert_message"].lower()


def test_scenario_mixed_speaker_evidence():
    """Scenario: 1 chunk slightly below threshold, but 2 chunks matching candidate (single anomaly)."""
    speaker_results = [
        {"start": 0.0, "end": 4.0, "score": 0.65, "same_speaker": True},
        {"start": 4.0, "end": 8.0, "score": 0.20, "same_speaker": False},
        {"start": 8.0, "end": 12.0, "score": 0.58, "same_speaker": True},
    ]
    result = make_decision(
        speech_segments=[{"start": 0.0, "end": 12.0}],
        speaker_results=speaker_results,
        yamnet_predictions=[("Speech", 0.92)],
        min_different=2,
    )
    # Different chunks = 1 (< min_different=2), but > 0 -> REVIEW_REQUIRED, NOT immediate SECOND_SPEAKER alert
    assert result["event_type"] == "REVIEW_REQUIRED"
    assert result["alert_required"]
    assert "uncertain" in result["alert_message"].lower() or "ambiguous" in result["alert_message"].lower()


def test_scenario_phone_replay_spoofing():
    """Scenario: Pre-recorded audio played through a phone speaker near microphone."""
    replay_predictions = [
        {"start": 0.0, "end": 6.0, "prediction": "phone_replay", "model_vote_fraction": 0.85},
    ]
    result = make_decision(
        speech_segments=[{"start": 0.0, "end": 6.0}],
        speaker_results=[{"start": 0.0, "end": 4.0, "score": 0.40, "same_speaker": True}],
        yamnet_predictions=[("Speech", 0.70)],
        replay_predictions=replay_predictions,
        replay_threshold=0.60,
    )
    assert result["event_type"] == "POSSIBLE_REPLAY_SPOOFING"
    assert result["alert_required"]
    assert result["metrics"]["replay_windows_flagged"] == 1
    assert "phone speaker playback" in result["alert_message"].lower() or "pre-recorded" in result["alert_message"].lower()

