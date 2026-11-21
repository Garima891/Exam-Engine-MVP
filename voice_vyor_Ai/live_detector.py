import sys
import time
import queue
import numpy as np
import sounddevice as sd

from detector import (
    SR,
    WIN,
    HOP,
    SIM_LOW,
    SIM_HIGH,
    SIM_INSTANT_ALERT,
    PHONE_TH,
    PHONE_HIGH_CONF,
    PHONE_COOLDOWN,
    MIN_SPEECH_DUR,
    enroll,
    embed,
    phone_score,
    safe_normalize,
    extract_evaluable_speech,
)

def start_live_monitor(enroll_path, device=None):
    # 1. Enroll candidate
    print(f"\n[1/3] Enrolling candidate from: {enroll_path}...")
    reference = enroll(enroll_path)
    print("Candidate enrolled successfully.")

    # 2. Setup audio buffer & queue
    audio_queue = queue.Queue()
    ring_buffer = np.zeros(WIN, dtype=np.float32)

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Warning] Audio buffer status: {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    # 3. Detection state
    history = []
    alert_spk = False
    speech_buf = np.zeros(0, dtype=np.float32)
    last_phone_alert = -1e9
    phone_hits = 0
    start_time = time.time()

    print("\n[2/3] Initializing microphone stream...")
    print(f"Sample Rate : {SR} Hz | Window: {WIN/SR:.1f}s | Step: {HOP/SR:.2f}s")
    print(f"Thresholds  : Speaker Alert < {SIM_LOW:.2f} | Phone Alert >= {PHONE_TH:.2f}")
    print("[3/3] Live Monitoring Active! (Press Ctrl+C to stop)\n")
    print("-" * 88)
    print(f"{'Time':<8} | {'Similarity':<12} | {'Avg(3)':<8} | {'Phone':<6} | {'Status'}")
    print("-" * 88)

    # Start audio input stream
    with sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocksize=HOP,
        device=device,
        callback=audio_callback,
    ):
        elapsed_hops = 0
        try:
            while True:
                # Wait for next 0.5s audio chunk
                chunk = audio_queue.get()
                ring_buffer = np.roll(ring_buffer, -HOP)
                ring_buffer[-HOP:] = chunk
                elapsed_hops += 1

                # Wait until ring buffer has filled with initial 3 seconds
                if elapsed_hops < (WIN // HOP):
                    continue

                t = time.time() - start_time
                win = ring_buffer.copy()
                events = []

                # --- VAD & Speaker Verification ---
                eval_speech, speech_buf, dur_in_win = extract_evaluable_speech(win, speech_buf)
                similarity = None

                if eval_speech is not None:
                    norm_speech = safe_normalize(eval_speech)
                    curr_emb = embed(norm_speech)
                    similarity = float(reference @ curr_emb)

                    history = (history + [similarity])[-3:]
                    avg_sim = float(np.mean(history))

                    # Tier 1: Instant Alert on High Confidence Mismatch
                    if similarity < SIM_INSTANT_ALERT and dur_in_win >= MIN_SPEECH_DUR:
                        alert_spk = True
                        events.append("!! ALERT: DIFFERENT SPEAKER (Instant) !!")

                    # Tier 2: 2-Window Fast Trigger
                    elif len(history) >= 2 and np.mean(history[-2:]) < SIM_LOW:
                        if not alert_spk:
                            alert_spk = True
                        events.append("!! ALERT: DIFFERENT SPEAKER (2-Hit) !!")

                    # Tier 3: 3-Window Average
                    elif len(history) == 3 and avg_sim < SIM_LOW:
                        if not alert_spk:
                            alert_spk = True
                        events.append("!! ALERT: DIFFERENT SPEAKER (Average) !!")

                    elif similarity < SIM_LOW and not alert_spk:
                        events.append("WARNING: Suspicious voice")

                    # Hysteresis recovery
                    elif alert_spk and len(history) >= 2 and np.mean(history[-2:]) > SIM_HIGH:
                        alert_spk = False
                        events.append("[OK] Candidate voice recovered")
                    elif not alert_spk:
                        events.append("[OK] Enrolled Candidate")
                else:
                    avg_sim = float(np.mean(history)) if history else None
                    if dur_in_win < 0.15:
                        events.append("Silence / Listening...")

                # --- Phone Sound Detection ---
                ps = phone_score(win)
                if ps >= PHONE_HIGH_CONF and (t - last_phone_alert) > PHONE_COOLDOWN:
                    last_phone_alert = t
                    phone_hits = 0
                    events.append("!! ALERT: PHONE SOUND (High Conf) !!")
                elif ps >= PHONE_TH:
                    phone_hits += 1
                    if phone_hits >= 2 and (t - last_phone_alert) > PHONE_COOLDOWN:
                        last_phone_alert = t
                        phone_hits = 0
                        events.append("!! ALERT: PHONE SOUND (Confirmed) !!")
                else:
                    phone_hits = 0

                sim_str = f"{similarity:.2f}" if similarity is not None else "     -    "
                avg_str = f"{avg_sim:.2f}" if avg_sim is not None else "   -   "
                status_str = " | ".join(events) if events else "Normal"

                print(f"{t:6.1f}s  | {sim_str:>12} | {avg_str:>8} | {ps:6.2f} | {status_str}")

        except KeyboardInterrupt:
            print("\n" + "-" * 88)
            print("Live monitoring stopped by user.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python live_detector.py <enrollment.wav> [mic_device_number]")
        print("Tip: If you don't know your mic_device_number, run without it to use default mic.")
        sys.exit(1)

    enrol = sys.argv[1]
    dev = int(sys.argv[2]) if len(sys.argv) > 2 else None
    start_live_monitor(enrol, dev)
