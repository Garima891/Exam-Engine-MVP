import sys, time, queue, threading
import numpy as np, sounddevice as sd
from detector import (
    SR, WIN, HOP, SIM_LOW, SIM_HIGH, SIM_INSTANT_ALERT,
    PHONE_TH, PHONE_HIGH_CONF, PHONE_COOLDOWN, MIN_SPEECH_DUR,
    enroll, embed, phone_score, safe_normalize, extract_evaluable_speech
)

try:
    import winsound
except ImportError:
    winsound = None

SPK_REPEAT = 8.0              # Beep again every this many seconds if intruder keeps speaking
DEVICE = int(sys.argv[2]) if len(sys.argv) > 2 else None

def _beep(kind):
    if winsound is None:
        print("\a", end="", flush=True); return
    if kind == "SPEAKER":                     # 2 long beeps for different speaker
        winsound.Beep(1000, 400); winsound.Beep(1000, 400)
    else:                                     # PHONE: 3 short, high-pitched beeps
        for _ in range(3): winsound.Beep(1800, 150)

def alert_sound(kind):
    threading.Thread(target=_beep, args=(kind,), daemon=True).start()

if len(sys.argv) < 2:
    print("Usage: python live.py <enrollment.wav> [device_number]")
    sys.exit(1)

e = enroll(sys.argv[1])
print("\nEnrollment done. Starting live monitoring... (Ctrl+C to stop)\n")

q = queue.Queue()
def cb(indata, frames, t, status):
    q.put(indata[:, 0].copy())

buf = np.zeros(0, dtype="float32")
speech_buf = np.zeros(0, dtype="float32")
since, hist, alert = 0, [], False
phone_hits, last_phone, last_spk = 0, 0.0, 0.0
mute_until = 0.0
t0 = time.time()

with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                    callback=cb, device=DEVICE):
    try:
        while True:
            chunk = q.get()
            now = time.time()

            # Ignore audio during/after beep (so the alert beep itself isn't picked up by mic)
            if now < mute_until:
                buf = np.zeros(0, dtype="float32"); since = 0
                continue

            buf = np.concatenate([buf, chunk])[-WIN:]
            since += len(chunk)
            if len(buf) < WIN or since < HOP:
                continue
            since = 0
            win = buf.copy()
            t = now - t0
            events = []

            # ---- Priority 2: VAD-guided Speech Extraction & Accumulation ----
            eval_speech, speech_buf, dur_in_win = extract_evaluable_speech(win, speech_buf)
            sim = None

            if eval_speech is not None:
                norm_speech = safe_normalize(eval_speech)
                w_emb = embed(norm_speech)
                sim_val = float(e @ w_emb)
                sim = f"{sim_val:.2f}"

                hist.append(sim_val)
                hist = hist[-3:]
                avg_val = np.mean(hist)

                # Tier 1: Instant high-confidence alert
                if sim_val < SIM_INSTANT_ALERT and dur_in_win >= MIN_SPEECH_DUR:
                    if not alert or (now - last_spk > SPK_REPEAT):
                        alert = True; last_spk = now
                        events.append("DIFFERENT_SPEAKER (Instant)")
                        alert_sound("SPEAKER"); mute_until = time.time() + 1.2
                # Tier 2: 2-Window Fast Trigger
                elif len(hist) >= 2 and np.mean(hist[-2:]) < SIM_LOW:
                    if not alert or (now - last_spk > SPK_REPEAT):
                        alert = True; last_spk = now
                        events.append("DIFFERENT_SPEAKER (2-Hit)")
                        alert_sound("SPEAKER"); mute_until = time.time() + 1.2
                # Tier 3: 3-Window Average
                elif len(hist) == 3 and avg_val < SIM_LOW:
                    if not alert or (now - last_spk > SPK_REPEAT):
                        alert = True; last_spk = now
                        events.append("DIFFERENT_SPEAKER (Avg)")
                        alert_sound("SPEAKER"); mute_until = time.time() + 1.2
                # Recovery hysteresis
                elif alert and len(hist) >= 2 and np.mean(hist[-2:]) > SIM_HIGH:
                    alert = False
                    events.append("(candidate back)")
            else:
                avg_val = np.mean(hist) if hist else None

            # ---- Priority 3: Calibrated Phone Sound Detection ----
            ps = phone_score(win)
            if ps >= PHONE_HIGH_CONF and (now - last_phone > PHONE_COOLDOWN):
                last_phone = now; phone_hits = 0
                events.append("PHONE_SOUND (Instant)")
                alert_sound("PHONE"); mute_until = time.time() + 1.2
            elif ps >= PHONE_TH:
                phone_hits += 1
                if phone_hits >= 2 and (now - last_phone > PHONE_COOLDOWN):
                    last_phone = now; phone_hits = 0
                    events.append("PHONE_SOUND (Confirmed)")
                    alert_sound("PHONE"); mute_until = time.time() + 1.2
            else:
                phone_hits = 0

            avg_s = f"{avg_val:.2f}" if avg_val is not None else " - "
            sim_s = sim if sim is not None else " - "
            flag = ">>> " + " | ".join(events) if events else ""
            print(f"{t:6.1f}s | sim {sim_s:>5} (avg {avg_s:>5}) | phone {ps:.2f} | {flag}")

            while q.qsize() > 10:
                q.get()
    except KeyboardInterrupt:
        print("\nStopped.")
