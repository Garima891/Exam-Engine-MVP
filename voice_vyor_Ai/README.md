# Interview Monitor (zero-shot, no training)

Models: ECAPA (speaker), YAMNet (phone sound), Silero (VAD). Pehli baar run pe internet se download honge.
ECAPA is fetched from pinned revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`; optional `custom.py` is not required.

## Setup (Windows PowerShell)
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

## Use
    python record.py enroll.wav 30          # lagataar bolo
    python check_enroll.py enroll.wav       # Speech mili 10 sec+ honi chahiye
    python record.py test_same.wav 60
    python record.py test_other.wav 60      # beech mein doosra insaan bole
    python record.py test_phone.wav 40      # beech mein ringtone bajao
    python detector.py enroll.wav test_same.wav
    python detector.py enroll.wav test_other.wav
    python detector.py enroll.wav test_phone.wav

Enroll aur test SAME mic se karo. Thresholds (sim_low, sim_high, phone_th) detector.py ke run() mein hain.
