"""Sets up the initial 2-speaker evaluation dataset from existing real recordings."""
import shutil
from pathlib import Path

def setup():
    root = Path(__file__).resolve().parent.parent
    data = root / "data"
    eval_dir = data / "evaluation"

    # Clean up empty folders
    for s in ["speaker_03", "speaker_04", "speaker_05"]:
        p = eval_dir / s
        if p.exists() and not list(p.glob("*.wav")):
            shutil.rmtree(p)
            print(f"Removed empty directory: {p}")

    # Speaker 01: Candidate recordings
    s1 = eval_dir / "speaker_01"
    s1.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data / "enrollment" / "candidate_reference_v2.wav", s1 / "clip_1.wav")
    shutil.copy2(data / "enrollment" / "candidate_test_v2.wav", s1 / "clip_2.wav")
    shutil.copy2(data / "enrollment" / "candidate_test.wav", s1 / "clip_3.wav")
    print(f"Populated speaker_01 with 3 clips.")

    # Speaker 02: Holdout live human speaker (Speaker 03)
    s2 = eval_dir / "speaker_02"
    s2.mkdir(parents=True, exist_ok=True)
    holdout = data / "replay_holdout" / "live"
    shutil.copy2(holdout / "speaker_03__phone_a__20261111_225400_669396.wav", s2 / "clip_1.wav")
    shutil.copy2(holdout / "speaker_03__phone_a__20261111_225535_425144.wav", s2 / "clip_2.wav")
    shutil.copy2(holdout / "speaker_03__phone_a__20261111_230901_063343.wav", s2 / "clip_3.wav")
    shutil.copy2(holdout / "speaker_03__phone_a__20261111_230920_084713.wav", s2 / "clip_4.wav")
    print(f"Populated speaker_02 with 4 clips.")

if __name__ == "__main__":
    setup()

