import importlib

modules = [
    "torch",
    "torchaudio",
    "speechbrain",
    "librosa",
    "soundfile",
    "numpy",
    "scipy",
    "df"
]

print("\n===== VOICE VYOR AI SETUP CHECK =====\n")

for module in modules:
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "Installed")
        print(f"[OK] {module}: {version}")
    except Exception as error:
        print(f"[FAILED] {module}: {error}")

print("\n===== CHECK COMPLETE =====")
