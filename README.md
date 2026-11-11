# Voice Proctoring Module

Detects a **second voice** or **phone sound** during an online interview, and ignores steady background noise such as fans, bikes and traffic. It is one module of the AI-Powered Interview & Recruitment Platform.

> **Status:** Proposed design. Mark each component as Implemented, In progress or Planned before presenting project status.

> **Important:** Alerts are flags for human review, not proof of cheating. A recruiter makes every decision. Do not auto-reject candidates based on these alerts.

---

## 1. What it does

| Situation | Result |
|---|---|
| Candidate speaks | No alert |
| A different person speaks (near or far) | Alert: `other_voice` |
| Phone ringtone or alarm | Lower-severity alert: `phone_sound` |
| Fan, bike, car, traffic noise | No alert |
| AI interviewer's voice leaking from speakers | Ignored |

## 2. How it works

Three layers, each removing one type of false alarm:

1. **Speech or not (Silero VAD).** Steady noise is ignored. Only human speech continues down the pipeline.
2. **Whose voice? (SpeechBrain ECAPA-TDNN).** A short voice sample is recorded at the start of the interview. Every later speech segment is compared against it. A mismatch means a possible second person.
3. **What sound? (YAMNet, optional).** Non-speech audio is labelled (telephone bell, ringtone, vehicle, and so on) so phone sounds can be reported even though they are not speech.

```
Browser mic -> Silero VAD (browser) -> speech segments -> WebSocket -> FastAPI
                                                                          |
                                                                  Celery worker
                                                          ECAPA match + YAMNet label
                                                                          |
                                                    Alert rules -> Recruiter dashboard
```

## 3. Tech stack

| Job | Tool | Runs on |
|---|---|---|
| Mic capture and speech detection | `getUserMedia` + `@ricky0123/vad-web` (Silero VAD) | Browser |
| Transport | WebSocket (FastAPI) | Browser to server |
| Speaker verification | SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) | Server, CPU |
| Sound labelling | YAMNet via ONNX Runtime or TFLite | Server, CPU |
| Audio utilities | NumPy, torchaudio or librosa, soundfile | Server |
| Async work | Celery + Redis | Server |
| Local testing only | SoundDevice | Your laptop |

**Notes**

- SoundDevice records from the machine running the Python code. In production the candidate's mic is in their browser, so SoundDevice is for local prototyping only.
- YAMNet is a TensorFlow model. Run it through TFLite or ONNX Runtime, not PyTorch.
- All models expect 16 kHz mono audio.

## 4. Laptop requirements (Lenovo IdeaPad Slim 3 and similar)

Everything runs on the CPU. No NVIDIA GPU is needed.

| Component | Fit |
|---|---|
| Silero VAD | Excellent |
| YAMNet | Excellent |
| ECAPA-TDNN | Good on short segments |
| faster-whisper tiny/base (int8), if you add transcription | Fine for development |
| Full pyannote diarization | Slow on CPU; use only for post-interview review |

Recommended: 8 GB RAM, Python 3.10 or newer, about 3 GB free disk space (CPU-only PyTorch).

## 5. Installation

```bash
git clone <your-repo-url>
cd voice-proctoring

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# CPU-only PyTorch (smaller download)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install speechbrain onnxruntime numpy soundfile librosa fastapi "uvicorn[standard]" celery redis

# Local testing only
pip install sounddevice
```

Download the YAMNet model (TFLite or ONNX) and place it in `models/`. Check the official YAMNet page on TensorFlow Hub for the current download.

## 6. Proposed project structure

```
voice-proctoring/
├── app/
│   ├── main.py              # FastAPI app and WebSocket endpoint
│   ├── speaker.py           # ECAPA enrollment and matching
│   ├── sounds.py            # YAMNet labelling
│   ├── alerts.py            # Alert rules, cooldowns, severity
│   ├── tasks.py             # Celery tasks
│   └── config.py            # Thresholds
├── scripts/
│   └── offline_test.py      # Test on WAV files before using the browser
├── models/                  # YAMNet model file
├── data/test_clips/         # Your labelled test recordings
├── frontend/                # Browser capture + VAD (React + TypeScript)
└── README.md
```

## 7. Usage

### Step 1: Offline prototype (do this first)

Test the pipeline on WAV files before building the browser part:

```bash
python scripts/offline_test.py --enroll data/test_clips/candidate_enroll.wav \
                               --input  data/test_clips/session_01.wav
```

### Step 2: Browser capture

```ts
import { MicVAD } from "@ricky0123/vad-web";

const vad = await MicVAD.new({
  positiveSpeechThreshold: 0.6,
  negativeSpeechThreshold: 0.4,
  minSpeechFrames: 5,
  additionalAudioConstraints: {
    echoCancellation: true,    // stops the AI voice from counting as speech
    noiseSuppression: false,   // can erase faint background voices
    autoGainControl: false,
  },
  onSpeechEnd: (audio) => socket.send(audio.buffer), // 16 kHz Float32Array
});
vad.start();
```

### Step 3: Server

```bash
redis-server
celery -A app.tasks worker --loglevel=info
uvicorn app.main:app --reload
```

### Core logic (reference)

```python
import torch
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier

enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def embed(wav_16k: torch.Tensor) -> torch.Tensor:
    return enc.encode_batch(wav_16k.unsqueeze(0)).squeeze()

def check_segment(wav, candidate_emb, ai_is_speaking, threshold=0.45):
    if len(wav) / 16000 < 1.0 or ai_is_speaking:
        return None                      # too short to judge, or the AI's own audio
    sim = F.cosine_similarity(embed(wav), candidate_emb, dim=0).item()
    if sim < threshold:
        return {"type": "other_voice", "similarity": sim}
    return None
```

## 8. Configuration

| Setting | Starting value | Meaning |
|---|---|---|
| `VAD_POSITIVE_THRESHOLD` | 0.6 | Speech probability needed to start a segment |
| `SPEAKER_THRESHOLD` | 0.45 | Similarity below this means a different voice |
| `MIN_SEGMENT_SECONDS` | 1.0 | Shorter segments are not judged |
| `MIN_ALERT_SPEECH_SECONDS` | 1.5 to 2.0 | Mismatched speech must last this long |
| `REPEAT_WINDOW_SECONDS` | 30 | Alternative trigger: 2 or more mismatches in this window |
| `ALERT_COOLDOWN_SECONDS` | 30 | Minimum gap between alerts of the same type |
| `YAMNET_PHONE_CONFIDENCE` | 0.5 | Minimum score for a `phone_sound` alert |

These are starting values. Tune them on your own recordings.

### Alert rules

- Alert only if mismatched speech lasts long enough, or repeats within the window.
- In text and coding rounds, treat any speech as suspicious. In voice rounds, treat only a different voice as suspicious.
- Run YAMNet on segments the VAD marked as non-speech, to detect ringtones.
- Save about 10 seconds of audio around each alert as evidence.

## 9. Alert format

```json
{
  "interview_id": "int_123",
  "type": "other_voice",
  "severity": "medium",
  "confidence": 0.71,
  "timestamp": "2026-11-05T10:42:15Z",
  "evidence_url": "s3://bucket/evidence/int_123/clip_042.wav",
  "review_status": "pending"
}
```

## 10. Testing and evaluation

Do not quote an accuracy number without testing.

1. Record 50 to 100 labelled clips: fan, bike, car horn, traffic, TV speech, ringtone, a second person near and far, a whisper, and the candidate alone.
2. Run the pipeline and compare against your labels.
3. Report **precision, recall, F1-score and false positive rate** for each alert type.
4. Adjust thresholds, then re-test. Repeat after every model or threshold change.

## 11. Known limitations

- **TV, YouTube or phone speakers playing speech** sound like a person and will be flagged.
- **Very quiet or distant voices** may be missed, depending on the microphone.
- **Short segments** (under about 1 second) give unreliable voice matches.
- **Voice changes** (a cold, a different mic distance) can cause false mismatches.
- **Loud horns and sirens** can trigger the VAD. The YAMNet layer helps filter them.
- **Headphones** reduce the AI voice leaking into the mic and improve accuracy.
- Browser-side checks can be tampered with, so validate critical signals on the server.

## 12. Privacy and compliance

- Show candidates what is monitored and get consent before recording starts.
- Store the voice embedding and evidence clips only for the retention period set by the company, then delete them.
- A voice embedding is biometric data under several laws (for example GDPR and some US state laws). Get legal review before launching in each market.
- Keep an audit log of who accessed evidence.

## 13. Roadmap

- [ ] Offline prototype on WAV files
- [ ] Labelled test set and threshold tuning
- [ ] Browser capture with Silero VAD
- [ ] FastAPI WebSocket endpoint and Celery worker
- [ ] YAMNet phone-sound alerts
- [ ] Recruiter dashboard integration and evidence timeline
- [ ] Post-interview diarization with pyannote (optional)

## 14. References

- Silero VAD: GitHub `snakers4/silero-vad`
- `@ricky0123/vad-web`: browser Silero VAD package
- SpeechBrain: GitHub `speechbrain/speechbrain`
- YAMNet: TensorFlow Hub

## 15. License

Add your license here.