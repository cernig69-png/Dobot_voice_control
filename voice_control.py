# voice_control.py  (Enter เริ่มอัด / Enter หยุดอัด + robust Dobot connect)
import time
import wave
import re
from typing import Optional

import numpy as np
import sounddevice as sd
from pydobot import Dobot

# ใช้ฟังก์ชันถอดเสียงของคุณ ตามที่มีใน NLP.py
from NLP import transcribe_wav   # ต้องให้ transcribe_wav() อ่านไฟล์ชื่อ record.wav

PORT = "COM4"
STEP = 10
VALID = ["left", "right", "up", "down"]
THAI_MAP = {"ซ้าย": "left", "ขวา": "right", "ขึ้น": "up", "ลง": "down"}

# ==============================
# 1) อัดเสียงแบบ Enter เริ่ม/หยุด
# ==============================
def record_until_enter(filename: str = "record.wav", samplerate: int = 16000, channels: int = 1) -> str:
    """
    กด Enter ครั้งแรกเพื่อเริ่มอัด และ Enter อีกครั้งเพื่อหยุด
    บันทึกเป็น filename (ค่าเริ่มต้น record.wav)
    """
    input("\nกด Enter เพื่อเริ่มอัดเสียง... (แล้วพูดคำสั่ง)")
    print("⏺️ เริ่มอัดแล้ว! (พูดคำสั่งได้เลย)  >> กด Enter อีกครั้งเพื่อหยุด")
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=callback)
    stream.start()

    # รอจนกด Enter อีกครั้ง
    input()
    print("⏹️ หยุดอัด")

    stream.stop()
    stream.close()

    # รวมบล็อกเสียงทั้งหมด แล้วบันทึกเป็น WAV 16-bit PCM
    audio_data = np.concatenate(frames, axis=0) if frames else np.zeros((1, channels), dtype=np.float32)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    print(f"💾 บันทึกเป็น {filename}")
    return filename

# ==============================
# 2) Utils: Dobot
# ==============================
def normalize_command(text: str) -> Optional[str]:
    """
    คืน 'left/right/up/down' ถ้าเจอในประโยค (รองรับไทยพื้นฐาน)
    """
    t = text.strip().lower()
    for th, en in THAI_MAP.items():
        if re.search(rf"(^|[\s.,!?:;]){re.escape(th)}($|[\s.,!?:;])", t):
            return en
    for en in VALID:
        if re.search(rf"(^|[\s.,!?:;]){en}($|[\s.,!?:;])", t):
            return en
    return None

def get_xyzr(device):
    pose = device.pose()      # คืน 8 ค่า
    return pose[:4]           # x, y, z, r

def move_abs(device, x, y, z, r, dwell=1.2):
    device.move_to(x, y, z, r)
    time.sleep(dwell)

def move_relative(device, direction, step=STEP):
    vec = {
        "left":  (0, -1,  0, 0),
        "right": (0,  1,  0, 0),
        "up":    (0,  0,  1, 0),
        "down":  (0,  0, -1, 0),
    }[direction]
    dx, dy, dz, dr = vec
    x0, y0, z0, r0 = get_xyzr(device)
    x1, y1, z1, r1 = x0 + dx*step, y0 + dy*step, z0 + dz*step, r0 + dr*step
    move_abs(device, x1, y1, z1, r1)
    print(f"✅ Moved {direction} {step} units")

def connect_dobot(port=PORT, retries=2, settle=0.8):
    """
    เชื่อมต่อ Dobot แบบทนทาน:
    - เว้นจังหวะหลังเปิดพอร์ต
    - ล้างบัฟเฟอร์ in/out
    - warm-up pose
    - retry อัตโนมัติ
    """
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            print(f"🚀 Connecting to Dobot... (attempt {attempt})")
            device = Dobot(port=port, verbose=True)
            time.sleep(settle)
            try:
                device._ser.reset_input_buffer()
                device._ser.reset_output_buffer()
            except Exception:
                pass
            _ = get_xyzr(device)  # warm-up
            print("✅ Connected! Current pose:", _)
            return device
        except Exception as e:
            last_err = e
            print(f"⚠️ Connect failed: {e}")
            try:
                device.close()
            except Exception:
                pass
            time.sleep(1.0)
    raise RuntimeError(f"ไม่สามารถเชื่อมต่อ Dobot ได้ที่ {port}: {last_err}")

# ==============================
# 3) Main loop
# ==============================
def main():
    try:
        device = connect_dobot(PORT, retries=2, settle=0.8)
    except Exception as e:
        print("❌", e)
        return

    print("🎧 Voice control ready. พูด: Left / Right / Up / Down (หรือ ซ้าย / ขวา / ขึ้น / ลง)")
    print("ℹ️ ขั้นตอน: กด Enter เพื่อเริ่มอัด → พูดคำสั่ง → กด Enter เพื่อหยุดอัด")

    try:
        while True:
            # กด Enter เริ่ม/หยุดอัด → บันทึกเป็น record.wav
            wavfile = record_until_enter(filename="record.wav", samplerate=16000, channels=1)

            # ส่งเข้าโมดูลถอดเสียงเดิมของคุณ (อ่านจาก record.wav)
            text = transcribe_wav()  # เวอร์ชันของคุณคาดว่าไม่รับพาธ และอ่าน record.wav ตรง ๆ
            if not text:
                print("🙈 ไม่ได้ยินคำสั่ง ลองใหม่อีกครั้ง")
                continue

            print("🗣️ ได้ยินว่า:", text)
            direction = normalize_command(text)
            if not direction:
                print("🤔 ไม่พบคำสั่งที่รู้จัก (Left/Right/Up/Down หรือ ซ้าย/ขวา/ขึ้น/ลง)")
                continue

            move_relative(device, direction, STEP)

    except KeyboardInterrupt:
        print("\n👋 ปิดการทำงาน")
    finally:
        try:
            device.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
