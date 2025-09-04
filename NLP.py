import queue
import sys
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel
from pathlib import Path

# ---------- ตั้งค่าหลัก ----------
SAMPLE_RATE = 16000      # อัตราสุ่มเสียง (Hz)
CHANNELS = 1             # โมโนพอ
MODEL_SIZE = "small"     # เลือกได้: tiny, base, small, medium, large-v3 (ใหญ่=แม่นขึ้นแต่ช้า/กินแรม)
LANGUAGE = "en"          # "th" = ไทย (จะ auto ก็ได้โดยใส่ None)
RECORD_SECONDS = 6       # ความยาวอัดแต่ละครั้ง (วินาที)
OUT_WAV = Path("record.wav")

# ---------- เตรียมคิวรับเสียง ----------
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    # แปลงเป็น float32 mono
    q.put(indata.copy())

def record_to_wav(duration=RECORD_SECONDS):
    print(f"🎙️ เริ่มอัดเสียง {duration} วินาที... (พูดได้เลย)")
    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        callback=audio_callback):
        start = time.time()
        while time.time() - start < duration:
            frames.append(q.get())
    audio = np.concatenate(frames, axis=0)

    # Normalize และแปลงเป็น int16 ก่อนเขียน WAV
    audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767) if np.max(np.abs(audio)) > 0 else np.int16(audio)
    wav_write(OUT_WAV, SAMPLE_RATE, audio_int16)
    print(f"✅ บันทึกไฟล์เสียงเป็น: {OUT_WAV.resolve()}")

def transcribe_wav(path=OUT_WAV, model_size=MODEL_SIZE, language=LANGUAGE):
    print("🧠 กำลังโหลดโมเดล (ครั้งแรกจะช้าหน่อย)...")
    # device="cpu" ใช้ได้ทุกเครื่อง; ถ้ามีการ์ดจอ NVIDIA ใช้ device="cuda" จะเร็วขึ้น
    model = WhisperModel(model_size, device="cpu", compute_type="int8")  # ลอง int8 ประหยัดแรม/เร็วขึ้น

    print("🔎 เริ่มถอดเสียง...")
    segments, info = model.transcribe(str(path), language=language, vad_filter=True)

    print(f"ℹ️  Language: {info.language}, Prob: {info.language_probability:.2f}")
    text = []
    for seg in segments:
        # พิมพ์เวลาบอกด้วย เผื่ออยากดูไทม์ไลน์
        print(f"[{seg.start:6.2f} → {seg.end:6.2f}] {seg.text}")
        text.append(seg.text)

    result = "".join(text).strip()
    print("\n📄 ข้อความรวม:")
    print(result)
    return result

if __name__ == "__main__":
    try:
        while True:
            # กด Enter เพื่อเริ่มอัด
            input("\nกด Enter เพื่ออัดเสียง (หรือ Ctrl+C เพื่อออก): ")
            record_to_wav(RECORD_SECONDS)
            transcribe_wav()
    except KeyboardInterrupt:
        print("\n👋 ออกโปรแกรมแล้ว")
