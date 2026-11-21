import sys, numpy as np
from detector import load_wav, speech_only, SR

w = load_wav(sys.argv[1], normalize=False)
s = speech_only(w)
print("Total length :", round(len(w) / SR, 1), "sec")
print("Speech mili  :", round(len(s) / SR, 1), "sec")
print("Max volume   :", round(float(np.abs(w).max()), 3))
print("Avg volume   :", round(float(np.sqrt((w ** 2).mean())), 4))
