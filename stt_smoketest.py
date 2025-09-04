import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

print("1) import faster_whisper ...", flush=True)
from faster_whisper import WhisperModel

MODEL = "tiny"   # เริ่มจาก tiny ให้โหลดเร็วสุด (~75–80MB)

print("2) instantiating model (this may download files the first time) ...", flush=True)
m = WhisperModel(MODEL, device="cpu", compute_type="int8")
print("3) model ready ✓", flush=True)