# Unified live proctoring demo

The demo displays the existing phone detector, registered face identity
verification, and audio analysis in a single OpenCV dashboard. The dashboard
shows the live webcam, separate face/identity/phone/microphone/voice/speaker/
sound cards, capped event counters, confirmed alerts, and timestamped history.
It does not enroll candidates, modify the candidate database, save
camera/audio recordings, or change model thresholds.

## Run

Launch the Tkinter student registration and registered-student workflow:

```powershell
.venv\Scripts\python.exe -m integration.exam_app
```

Set `EXAM_AUDIO_PYTHON` if the separate audio environment is outside the
known local path:

```powershell
$env:EXAM_AUDIO_PYTHON = "C:\path\to\exam-engine-audio311\Scripts\python.exe"
.venv\Scripts\python.exe -m integration.exam_app
```

From Registered Students, select only a fully registered student and choose
Start Exam. The app creates a database-backed session and launches the existing
live OpenCV dashboard with that student's three face references and one voice
reference. Legacy incomplete candidates cannot start an exam.

Run the parent from the project environment that has OpenCV, MediaPipe, and
Ultralytics. Pass the Python executable from the existing isolated audio
environment (which has TensorFlow and the audio dependencies):

```powershell
python -m integration.live_demo `
  --audio-python "C:\path\to\exam-engine-audio311\Scripts\python.exe"
```

The demo uses the sole registered candidate automatically when the database
contains exactly one registered record. Otherwise, pass `--candidate-id`.
Candidate embeddings are read through a read-only SQLite connection.
At startup the selected candidate ID, stored embedding count/dimension, and
configured face-similarity threshold are reported. Live face scores stay bound
to their detected face and are shown in the overlay and terminal summary.

Speaker verification is unavailable unless a voice reference is explicitly
linked to the selected candidate. The existing `voice_vyor_Ai\enroll.wav` is
not used by default. To opt in, pass both the WAV and the matching registered
candidate ID:

```powershell
python -m integration.live_demo `
  --audio-python "C:\path\to\exam-engine-audio311\Scripts\python.exe" `
  --audio-reference "C:\path\to\candidate-voice.wav" `
  --audio-reference-candidate-id "candidate-id"
```

Supplying both values is an explicit operator assertion that the WAV belongs
to that candidate; the demo cannot independently prove this association. The
worker processes the reference in memory only. Without it, voice activity and
suspicious-sound analysis continue, while speaker verification and
unknown-speaker events remain unavailable. Cached ECAPA and YAMNet models are
required; the worker refuses to download models.

`CameraManager` and the audio worker remain the sole owners of their devices.
The camera starts even if the worker is still loading or the microphone is
unavailable. The audio worker sends only JSON analysis results and scalar
level telemetry to the video process; captured audio remains in memory. Press
**Q** in the video window to stop and release both devices. Press **1-9** or
**0** in that window to display the matching manual test scenario and its
instruction in both the window and terminal. Vision inference runs outside the
dashboard-rendering loop. The live clock continues updating while inference is
running; detection overlays appear only when their result belongs to the
displayed camera frame.

Microphone diagnostics report the selected input device, callback count,
sample range, RMS, queue occupancy, and per-window RMS/peak. The dashboard
shows VAD speech duration and marks audio processing stalled when complete
analysis windows stop arriving; it does not convert an old no-speech result
into a current silence result. Existing sample rate, window/hop sizes, and
model thresholds are unchanged.

Microphone capture uses a four-block bounded queue (two seconds at the current
block size). If inference falls behind, the callback discards queued stale
blocks and advances the stream generation. The monitor reports the gap,
discards any partial analysis window, resets accumulated speech state, and
resumes only after assembling a complete window from the new generation.
Consequently, an analysis window never combines samples across a dropped
region. Repeated device open/stream failures receive up to three retry
attempts; an unrecovered error is shown as audio unavailable and does not stop
phone or face monitoring.

## Event rules

`PHONE_DETECTED`, `IDENTITY_UNMATCHED`, `NO_FACE_DETECTED`,
`UNKNOWN_HUMAN_VOICE`, and `SUSPICIOUS_AUDIO_SOUND` have independent counters.
Each unique frame/window contributes at most once to each applicable event.
Counters start at zero for each new session and stop at 5/5. Five observations
confirm that event; observations need not be consecutive.
Scores, missing references, invalid results, and inference errors alone do not
count as events. The session event history is displayed in the window and
printed as timestamped JSON lines in the terminal. Events are also appended to
`integration\proctoring_event_history.jsonl` (ignored by Git). Previous
confirmed events are shown separately from the current session's counters and
event history. Candidate/audio recordings are not stored.

Run mocked integration tests before any physical-device test:

```powershell
python -m unittest discover -s integration\tests
```

## Student registration storage foundation

The SQLite database remains the single local store. `candidate_id` is retained
as the stable `student_id` for compatibility with existing candidate records.
On the first writable `CandidateStorage` initialization after the schema
upgrade, a transaction creates the `students` and
`student_voice_embeddings` tables and migrates existing candidates without
deleting their candidate rows or face embeddings. Existing candidates are
marked `LEGACY_INCOMPLETE`: an old averaged face embedding is not treated as
three separately enrolled samples, and no voice identity is inferred for them.
They must complete registration before being eligible for a new exam session.

New registrations are written atomically through
`StudentRegistrationService`: student details, three separately indexed face
embeddings, and one candidate-linked voice embedding are committed together.
Enrollment numbers are unique after trimming and case-folding. The storage API
does not capture media or generate embeddings; callers must provide validated
outputs from the existing face and audio models. Student-list results omit
embedding values.

Voice registration initializes cached Silero VAD and the pinned ECAPA model in
the separate audio environment. It does not load YAMNet, which is used by live
audio monitoring rather than voice enrollment.

Face and voice embedding blobs are protected with Windows DPAPI for the
current Windows user. The first schema migration encrypts existing plaintext
face embeddings in place; a failed migration rolls back. DPAPI-protected
embeddings are not portable: restoring them under another Windows user or
machine may make them unrecoverable and require biometric re-enrollment. Older
manual database backups are not modified by the migration and may still contain
plaintext embeddings; protect those backups accordingly. This protection does
not encrypt student names, enrollment numbers, or email fields.
