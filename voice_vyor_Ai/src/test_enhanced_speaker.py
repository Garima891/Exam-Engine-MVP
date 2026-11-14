"""Compares 3 ways of using DeepFilterNet so you can see which one keeps the voice.
  A) raw reference  vs raw test            (baseline)
  B) raw reference  vs enhanced test       (your old setup -> low score)
  C) enhanced ref   vs enhanced test       (enroll and verify on the same type of audio)
"""
import settings
from noise_reduction import enhance_file
from speaker_detection import CandidateSpeakerVerifier

for p in [settings.REFERENCE, settings.CANDIDATE_TEST, settings.SECOND_PERSON]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required recording: {p}")

enhance_file(settings.REFERENCE, settings.ENHANCED_REFERENCE)
enhance_file(settings.CANDIDATE_TEST, settings.ENHANCED_CANDIDATE)
enhance_file(settings.SECOND_PERSON, settings.ENHANCED_SECOND)

raw_ver = CandidateSpeakerVerifier(str(settings.REFERENCE))
enh_ver = CandidateSpeakerVerifier(str(settings.ENHANCED_REFERENCE))

rows = [
    ("A) raw ref vs raw candidate",          raw_ver, settings.CANDIDATE_TEST),
    ("A) raw ref vs raw second person",      raw_ver, settings.SECOND_PERSON),
    ("B) raw ref vs enhanced candidate",     raw_ver, settings.ENHANCED_CANDIDATE),
    ("B) raw ref vs enhanced second person", raw_ver, settings.ENHANCED_SECOND),
    ("C) enh ref vs enhanced candidate",     enh_ver, settings.ENHANCED_CANDIDATE),
    ("C) enh ref vs enhanced second person", enh_ver, settings.ENHANCED_SECOND),
]
print(f"\n{'Test':40s} {'Score':>7s}  Result")
for label, ver, path in rows:
    r = ver.compare(str(path))
    print(f"{label:40s} {r['score']:7.4f}  "
          f"{'SAME speaker' if r['same_speaker'] else 'DIFFERENT speaker'}")
print("\nBest setup = candidate scores high AND second person scores low.")
