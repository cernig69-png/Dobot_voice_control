# -*- coding: utf-8 -*-
"""
STT ไทย/อังกฤษ แบบแม่นขึ้น (faster-whisper)
- ตรวจภาษาอัตโนมัติ (ไทย/อังกฤษ/ผสม)
- ปรับพารามิเตอร์ถอดเสียงให้เสถียร
- ใช้ CUDA อัตโนมัติถ้ามี
"""

import os, sys, time, queue, warnings
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")

# -------- Config --------
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 6
OUT_WAV = Path("record.wav")

# โมเดลหลายภาษา (ห้ามใช้รุ่น .en ถ้าจะเอาไทยด้วย)
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium/large-v3
USE_CUDA = os.getenv("USE_CUDA", "auto")          # auto/cpu/cuda

# คำศัพท์/บริบทเฉพาะ (ออปชัน) เช่น ชื่อแบรนด์ คำเทคนิค
INITIAL_PROMPT = os.getenv("INITIAL_PROMPT", "").strip() or None

# -------- Recorder --------
_q = queue.Queue()

def _audio_cb(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr, flush=True)
    _q.put(indata.copy())

def record_to_wav(seconds=RECORD_SECONDS):
    print(f"🎙️  อัดเสียง {seconds}s ... (พูดได้เลย)")
    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        callback=_audio_cb):
        start = time.time()
        while time.time() - start < seconds:
            try:
                frames.append(_q.get(timeout=1))
            except queue.Empty:
                pass
    if not frames:
        print("⚠️  ไม่ได้เสียงจากไมค์"); return None
    audio = np.concatenate(frames, axis=0)

    # Normalize ป้องกันคลิป/เสียงเบาเกิน
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    audio_i16 = np.int16(audio/peak*32767) if peak > 0 else np.int16(audio)

    wav_write(OUT_WAV, SAMPLE_RATE, audio_i16)
    print(f"✅  บันทึก: {OUT_WAV.resolve()}")
    return str(OUT_WAV)

# -------- STT --------
def _pick_device():
    if USE_CUDA.lower() == "cuda":
        return "cuda"
    if USE_CUDA.lower() == "cpu":
        return "cpu"
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def load_model():
    from faster_whisper import WhisperModel
    device = _pick_device()
    compute = "float16" if device == "cuda" else "int8"  # เร็ว/คุ้มทรัพยากร
    print(f"🧠  โหลดโมเดล: {MODEL_SIZE} (device={device}, compute={compute})")
    t0 = time.time()
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute)
    print(f"✅  พร้อมใช้งาน ({time.time()-t0:.1f}s)")
    return model

def transcribe_file(model, path):
    print("🔎  ถอดเสียง ...")

    # พารามิเตอร์ที่ช่วยความแม่นคง:
    kwargs = dict(
        # ภาษา=None ให้ตรวจอัตโนมัติ (รองรับสลับภาษาในประโยค)
        language=None,
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 100},
        # ค้นหาแบบ beam search + อุณหภูมิหลายค่า (กันหลุด/มั่ว)
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.5],
        # เกณฑ์หยุด/ไม่ใช่คำพูด
        no_speech_threshold=0.45,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.6,
        condition_on_previous_text=False,  # คลิปสั้นอิสระ ไม่ผูกเนื้อหาก่อนหน้า
        initial_prompt=INITIAL_PROMPT or None,
    )

    segments, info = model.transcribe(str(path), **kwargs)

    print(f"ℹ️  Detected language: {info.language} (p={info.language_probability:.2f})")
    texts = []
    for seg in segments:
        print(f"[{seg.start:6.2f} → {seg.end:6.2f}] {seg.text}")
        texts.append(seg.text)

    full = "".join(texts).strip()
    print("\n📄  ข้อความรวม:")
    print(full)
    return full

# -------- Main --------
if __name__ == "__main__":
    try:
        model = load_model()
        while True:
            input("\nกด Enter เพื่ออัดเสียง (Ctrl+C เพื่อออก): ")
            wav = record_to_wav(RECORD_SECONDS)
            if not wav:
                continue
            transcribe_file(model, wav)
    except KeyboardInterrupt:
        print("\n👋 ออกโปรแกรม")
