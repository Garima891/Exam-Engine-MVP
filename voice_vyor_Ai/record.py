# Mic se wav record karo:  python record.py enroll.wav 30 [device_number]
import sys, sounddevice as sd, soundfile as sf

name = sys.argv[1]
secs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
device = int(sys.argv[3]) if len(sys.argv) > 3 else None
SR = 16000

print(f"{secs} sec ki recording 3 sec baad shuru hogi...")
sd.sleep(3000)
print(">>> AB BOLO <<< (lagataar bolte raho, koi paragraph padh do)")
audio = sd.rec(int(secs * SR), samplerate=SR, channels=1, dtype="float32", device=device)
sd.wait()
sf.write(name, audio, SR)
print("Ho gaya, saved:", name)
