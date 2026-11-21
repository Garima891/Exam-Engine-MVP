import sys, numpy as np
from detector import SR, WIN, HOP, enroll, embed, speech_ratio, load_wav, yam, classes

e = enroll(sys.argv[1])
w = load_wav(sys.argv[2], normalize=False)
hist = []
print("time | sim (avg) | YAMNet top 3 (max over window)")
for s in range(0, len(w) - WIN + 1, HOP):
    win = w[s:s + WIN]
    t = s / SR
    sim = None
    if speech_ratio(win) > 0.4:
        rms = np.sqrt((win ** 2).mean()) + 1e-9
        sim = float(e @ embed(np.clip(win * (0.05 / rms), -1, 1)))
        hist = (hist + [sim])[-5:]
    m = yam(win)[0].numpy().max(axis=0)
    top = np.argsort(m)[-3:][::-1]
    tops = ", ".join(f"{classes[i]} {m[i]:.2f}" for i in top)
    sim_s = f"{sim:.2f}" if sim is not None else " - "
    avg_s = f"{np.mean(hist):.2f}" if hist else " - "
    print(f"{t:5.1f} | {sim_s} ({avg_s}) | {tops}")