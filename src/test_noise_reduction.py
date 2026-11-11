import settings
from noise_reduction import enhance_file

print("Enhancing:", settings.CANDIDATE_TEST)
print("Saved:", enhance_file(settings.CANDIDATE_TEST, settings.ENHANCED_CANDIDATE))
