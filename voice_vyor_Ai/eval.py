import sys
import csv
import numpy as np

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
    load_wav,
    safe_normalize,
    extract_evaluable_speech,
)

GRACE = 2.0  # Alert grace window after event end (seconds)

def ov(a0, a1, b0, b1):
    """Calculate overlap duration between intervals [a0, a1] and [b0, b1]."""
    return max(0.0, min(a1, b1) - max(a0, b0))

def score_file(path, reference):
    """
    Evaluate audio file using identical preprocessing, VAD gating,
    confidence decisions, and latency optimizations as detector.py.
    """
    w = load_wav(path, normalize=False)
    rows = []
    hist = []
    alert_spk = False
    speech_buf = np.zeros(0, dtype=np.float32)

    spk_alerts = []
    ph_alerts = []
    last_ph = -1e9
    ph_hits = 0

    for s in range(0, len(w) - WIN + 1, HOP):
        win = w[s:s + WIN]
        t0 = s / SR
        t1 = (s + WIN) / SR

        # Priority 2: VAD-guided speech extraction & accumulation
        eval_speech, speech_buf, dur_in_win = extract_evaluable_speech(win, speech_buf)

        sim = None
        spk_flag = False

        if eval_speech is not None:
            norm_speech = safe_normalize(eval_speech)
            curr_emb = embed(norm_speech)
            sim = float(reference @ curr_emb)

            # Priority 3: Shorter history (3 frames)
            hist = (hist + [sim])[-3:]
            avg_sim = np.mean(hist)

            # Tier 1: Instant Alert on High Confidence Mismatch
            if sim < SIM_INSTANT_ALERT and dur_in_win >= MIN_SPEECH_DUR:
                alert_spk = True
                spk_alerts.append(t1)
                spk_flag = True

            # Tier 2: 2-Window Fast Trigger
            elif len(hist) >= 2 and np.mean(hist[-2:]) < SIM_LOW:
                if not alert_spk:
                    alert_spk = True
                    spk_alerts.append(t1)
                spk_flag = True

            # Tier 3: 3-Window Average
            elif len(hist) == 3 and avg_sim < SIM_LOW:
                if not alert_spk:
                    alert_spk = True
                    spk_alerts.append(t1)
                spk_flag = True

            # Recovery hysteresis
            elif alert_spk and len(hist) >= 2 and np.mean(hist[-2:]) > SIM_HIGH:
                alert_spk = False

        # Phone Sound Detection (Priority 3: Instant High-Conf or 2-Hit Confirmation)
        ps = phone_score(win)
        ph_flag = False

        if ps >= PHONE_HIGH_CONF and (t1 - last_ph) > PHONE_COOLDOWN:
            last_ph = t1
            ph_hits = 0
            ph_alerts.append(t1)
            ph_flag = True
        elif ps >= PHONE_TH:
            ph_hits += 1
            if ph_hits >= 2 and (t1 - last_ph) > PHONE_COOLDOWN:
                last_ph = t1
                ph_hits = 0
                ph_alerts.append(t1)
                ph_flag = True
        else:
            ph_hits = 0

        rows.append((t0, t1, sim, ps, spk_flag, ph_flag))

    return rows, spk_alerts, ph_alerts, len(w) / SR

def auc_eer(pos, neg):
    """Calculate ROC-AUC and Equal Error Rate (EER)."""
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) == 0 or len(neg) == 0:
        return 0.5, 0.5, 0.0

    auc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
    thresholds = np.unique(np.concatenate([pos, neg]))
    best = (9.0, thresholds[0], 0.5)

    for th in thresholds:
        fnr = (pos < th).mean()
        fpr = (neg >= th).mean()
        diff = abs(fnr - fpr)
        if diff < best[0]:
            best = (diff, th, (fnr + fpr) / 2.0)

    return auc, best[2], best[1]

def confusion_metrics(tp, fp, fn, tn):
    """Compute Accuracy, Precision, Recall, Specificity, and F1."""
    total = max(tp + fp + fn + tn, 1)
    acc = (tp + tn) / total
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    f1 = 2 * (pr * rc) / max(pr + rc, 1e-9)
    return acc, pr, rc, sp, f1

def print_metrics_table(title, tp, fp, fn, tn, auc=None, eer=None, best_th=None):
    """Nicely formatted metrics display with full confusion matrix."""
    acc, pr, rc, sp, f1 = confusion_metrics(tp, fp, fn, tn)
    print(f"\n{title}")
    print("-" * 65)
    print(f"  Confusion Matrix: TP={tp:<4} | FP={fp:<4} | FN={fn:<4} | TN={tn:<4}")
    print(f"  Accuracy        : {acc * 100:.2f}%")
    print(f"  Precision       : {pr * 100:.2f}%")
    print(f"  Recall (Sens.)  : {rc * 100:.2f}%")
    print(f"  Specificity     : {sp * 100:.2f}%")
    print(f"  F1-Score        : {f1 * 100:.2f}%")
    if auc is not None and eer is not None:
        print(f"  ROC-AUC         : {auc:.3f} | EER: {eer * 100:.2f}% | Best Thresh: {best_th:.2f}")

# ============================================================
# MAIN EVALUATION
# ============================================================

if len(sys.argv) < 3:
    print("Usage: python eval.py <enrollment.wav> <labels.csv>")
    sys.exit(1)

enrol_path, labels_path = sys.argv[1], sys.argv[2]
e = enroll(enrol_path)

labels = {}
raw = open(labels_path, "rb").read()
if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
    text = raw.decode("utf-16")
else:
    text = raw.decode("utf-8-sig")

lines = [l.strip() for l in text.splitlines() if l.strip()]
if lines[0].replace(" ", "").lower() != "file,start,end,label":
    lines.insert(0, "file,start,end,label")

for r in csv.DictReader(lines):
    r = {k.strip(): v.strip() for k, v in r.items()}
    labels.setdefault(r["file"], []).append((float(r["start"]), float(r["end"]), r["label"]))

print(f"\nLoaded {len(labels)} file annotations.")

# Priority 4: All-Window Categorization (Pure, Transition, All)
sp_pure_pos, sp_pure_neg, sp_trans = [], [], []
ph_pure_pos, ph_pure_neg, ph_trans = [], [], []

ev = {"speaker": [0, 0, []], "phone": [0, 0, []]}  # [detected, total, delays]
fa = {"speaker": 0, "phone": 0}
total_sec = 0.0

for f, evs in labels.items():
    print(f"Evaluating {f}...")
    rows, spk_al, ph_al, dur = score_file(f, e)
    total_sec += dur

    # Priority 4 & 3: Event-Level Recall, False Alarms, and Detection Delays
    for kind, al in (("speaker", spk_al), ("phone", ph_al)):
        ivs = [(a, b) for a, b, l in evs if l == kind]
        for a, b in ivs:
            ev[kind][1] += 1
            hit = [t for t in al if a <= t <= b + GRACE]
            if hit:
                ev[kind][0] += 1
                ev[kind][2].append(max(0.0, hit[0] - a))  # Delay in seconds from event start
        fa[kind] += sum(1 for t in al if not any(a <= t <= b + GRACE for a, b in ivs))

    # Priority 4: Window-Level Evaluation without Discarding Partial Windows
    win_len = WIN / SR
    for t0, t1, sim, ps, _, _ in rows:
        o_sp = sum(ov(t0, t1, a, b) for a, b, l in evs if l == "speaker")
        o_ph = sum(ov(t0, t1, a, b) for a, b, l in evs if l == "phone")

        # Speaker evaluation (lower similarity = intruder positive)
        if sim is not None:
            score = -sim
            if o_sp >= 0.5 * win_len:
                sp_pure_pos.append(score)
            elif o_sp == 0.0:
                sp_pure_neg.append(score)
            else:
                sp_trans.append((score, o_sp / win_len))

        # Phone evaluation (higher score = phone positive)
        if o_ph >= 0.5 * win_len:
            ph_pure_pos.append(ps)
        elif o_ph == 0.0:
            ph_pure_neg.append(ps)
        else:
            ph_trans.append((ps, o_ph / win_len))

hrs = total_sec / 3600.0

print("\n" + "=" * 65)
print(f"EVALUATION RESULTS (Total Test Audio: {total_sec/60:.2f} min)")
print("=" * 65)

# ------------------------------------------------------------
# 1. SECOND SPEAKER EVALUATION (PRIORITY 1, 2, 3, 4)
# ------------------------------------------------------------
print("\n" + "#" * 65)
print("TASK 1: SECOND SPEAKER DETECTION")
print("#" * 65)

# A. Pure Windows
if len(sp_pure_pos) >= 2 and len(sp_pure_neg) >= 2:
    auc, eer, th = auc_eer(sp_pure_pos, sp_pure_neg)
    cur = -SIM_LOW
    tp = sum(p >= cur for p in sp_pure_pos)
    fn = len(sp_pure_pos) - tp
    fp = sum(n >= cur for n in sp_pure_neg)
    tn = len(sp_pure_neg) - fp
    print_metrics_table(
        f"A. Pure Windows (Pure Intruder >=50% vs Pure Enrolled, threshold={SIM_LOW})",
        tp, fp, fn, tn, auc, eer, -th
    )

# B. Transition Windows
if sp_trans:
    trans_scores = [s for s, _ in sp_trans]
    trans_flagged = sum(s >= -SIM_LOW for s in trans_scores)
    print(f"\nB. Transition Windows (0% < Overlap < 50%)")
    print("-" * 65)
    print(f"  Total Transition Windows : {len(sp_trans)}")
    print(f"  Flagged as Intruder      : {trans_flagged} ({trans_flagged / len(sp_trans) * 100:.1f}%)")
    print(f"  Mean Intruder Overlap    : {np.mean([r for _, r in sp_trans]) * 100:.1f}%")

# C. All Windows Combined (No Dropped Windows!)
all_sp_pos = sp_pure_pos + [s for s, _ in sp_trans]
all_sp_neg = sp_pure_neg
if len(all_sp_pos) >= 2 and len(all_sp_neg) >= 2:
    auc, eer, th = auc_eer(all_sp_pos, all_sp_neg)
    cur = -SIM_LOW
    tp = sum(p >= cur for p in all_sp_pos)
    fn = len(all_sp_pos) - tp
    fp = sum(n >= cur for n in all_sp_neg)
    tn = len(all_sp_neg) - fp
    print_metrics_table(
        f"C. ALL Windows Evaluated (Pure + Transitions, threshold={SIM_LOW})",
        tp, fp, fn, tn, auc, eer, -th
    )

# ------------------------------------------------------------
# 2. PHONE SOUND EVALUATION
# ------------------------------------------------------------
print("\n" + "#" * 65)
print("TASK 2: PHONE SOUND DETECTION")
print("#" * 65)

# A. Pure Windows
if len(ph_pure_pos) >= 2 and len(ph_pure_neg) >= 2:
    auc, eer, th = auc_eer(ph_pure_pos, ph_pure_neg)
    tp = sum(p >= PHONE_TH for p in ph_pure_pos)
    fn = len(ph_pure_pos) - tp
    fp = sum(n >= PHONE_TH for n in ph_pure_neg)
    tn = len(ph_pure_neg) - fp
    print_metrics_table(
        f"A. Pure Windows (Pure Phone >=50% vs Pure Clean, threshold={PHONE_TH})",
        tp, fp, fn, tn, auc, eer, th
    )

# B. Transition Windows
if ph_trans:
    trans_scores = [s for s, _ in ph_trans]
    trans_flagged = sum(s >= PHONE_TH for s in trans_scores)
    print(f"\nB. Transition Windows (0% < Overlap < 50%)")
    print("-" * 65)
    print(f"  Total Transition Windows : {len(ph_trans)}")
    print(f"  Flagged as Phone Alert   : {trans_flagged} ({trans_flagged / len(ph_trans) * 100:.1f}%)")

# C. All Windows Combined
all_ph_pos = ph_pure_pos + [s for s, _ in ph_trans]
all_ph_neg = ph_pure_neg
if len(all_ph_pos) >= 2 and len(all_ph_neg) >= 2:
    auc, eer, th = auc_eer(all_ph_pos, all_ph_neg)
    tp = sum(p >= PHONE_TH for p in all_ph_pos)
    fn = len(all_ph_pos) - tp
    fp = sum(n >= PHONE_TH for n in all_ph_neg)
    tn = len(all_ph_neg) - fp
    print_metrics_table(
        f"C. ALL Windows Evaluated (Pure + Transitions, threshold={PHONE_TH})",
        tp, fp, fn, tn, auc, eer, th
    )

# ------------------------------------------------------------
# 3. EVENT-LEVEL PERFORMANCE & DETECTION LATENCY (PRIORITIES 3 & 4)
# ------------------------------------------------------------
print("\n" + "#" * 65)
print("EVENT-LEVEL DETECTION & LATENCY REPORT")
print("#" * 65)

for kind, title in (("speaker", "SECOND SPEAKER"), ("phone", "PHONE SOUND")):
    det, total, delays = ev[kind]
    false_alarms = fa[kind]
    missed = total - det
    fa_per_hr = false_alarms / max(hrs, 1e-6)
    event_recall = (det / total * 100.0) if total > 0 else 0.0
    event_precision = (det / max(det + false_alarms, 1) * 100.0)

    print(f"\n--- {title} (Event-Level) ---")
    print(f"  Total Ground Truth Events : {total}")
    print(f"  Successfully Detected     : {det} ({event_recall:.1f}% Recall)")
    print(f"  Missed Events (FN)        : {missed}")
    print(f"  False Alarms (FP)         : {false_alarms} ({fa_per_hr:.2f} per hour)")
    print(f"  Event Precision           : {event_precision:.1f}%")

    if delays:
        print(f"  Detection Delay (Latency) :")
        print(f"    - Mean   : {np.mean(delays):.2f} sec")
        print(f"    - Median : {np.median(delays):.2f} sec")
        print(f"    - Min    : {np.min(delays):.2f} sec")
        print(f"    - Max    : {np.max(delays):.2f} sec")
    else:
        print("  Detection Delay           : No events detected")

print("\n" + "=" * 65)
print("Evaluation Complete.")
print("=" * 65)
