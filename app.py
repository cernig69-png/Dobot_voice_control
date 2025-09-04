# -*- coding: utf-8 -*-
"""
ควบคุม Dobot Magician ด้วยเสียง (ภาษาไทย)
- STT: faster-whisper
- Audio: sounddevice
- Robot: pydobot (serial)

วิธีใช้:
1) ตั้งค่า COM_PORT ให้ตรงพอร์ตจริง (เช่น "COM4")
2) pip install faster-whisper sounddevice scipy numpy pyserial pydobot
3) python -u app.py
คำสั่งเสียงตัวอย่าง:
- "เริ่มควบคุม" (ปลดล็อก) / "หยุดควบคุม" (ล็อก)
- "ขึ้น 10", "ลง 5", "ซ้าย 10", "ขวา 5", "หน้า 10", "ถอย 10"
- "ดูด" / "ปล่อย" (vacuum)
- "โฮม" / "หยุดฉุกเฉิน"
"""

import warnings; warnings.filterwarnings("ignore", category=UserWarning, module="ctranslate2")
print("🚀 app.py start", flush=True)

# ================== CONFIG ==================
COM_PORT = "COM4"     # ใส่พอร์ตจริง เช่น "COM4" (แนะนำใส่ให้ตายตัว)
BAUDRATE = 115200

MODEL_SIZE = "tiny"   # tiny/base/small/medium/large-v3 (ใหญ่=แม่นขึ้นแต่ช้ากว่า)
LANGUAGE   = "th"     # None = ให้โมเดลเดาภาษาเอง

SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK_SECONDS = 4.0   # อัดทีละกี่วินาที
STEP_MM       = 5.0   # ระยะก้าวเริ่มต้นต่อคำสั่ง (mm) ปลอดภัยไว้ก่อน

# ขอบเขตความปลอดภัยพื้นที่ทำงาน (mm) — ปรับให้เข้ากับโต๊ะคุณ
X_MIN, X_MAX = 100, 260
Y_MIN, Y_MAX = -150, 150
Z_MIN, Z_MAX = -5, 130

# HOME สูงไว้ก่อนกันชนโต๊ะ
HOME_POS = (200.0, 0.0, 80.0, 0.0)   # (x,y,z,r)
# ============================================

# ------------- Imports แบบ Lazy เพื่อตรวจจุดค้าง -------------
def step(msg): print(msg, flush=True)

def load_stt_model():
    step("🧠 โหลดโมเดล STT ... (ครั้งแรกอาจดาวน์โหลดไฟล์)")
    from faster_whisper import WhisperModel
    m = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    step("✅ STT พร้อม")
    return m

def record_once(seconds=CHUNK_SECONDS):
    # อัดเสียงระยะสั้นจากไมค์ → record.wav
    import sounddevice as sd
    import numpy as np
    from scipy.io.wavfile import write as wav_write
    from pathlib import Path
    import queue, time

    step(f"🎧 อัดเสียง {seconds:.1f}s ... (พูดได้เลย)")
    q = queue.Queue()
    def cb(indata, frames, t, status):
        if status: print("AUDIO:", status, flush=True)
        q.put(indata.copy())

    frames = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", callback=cb):
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                frames.append(q.get(timeout=1))
            except queue.Empty:
                pass

    if not frames:
        step("⚠️ ไม่ได้รับสัญญาณเสียง")
        return None

    audio = np.concatenate(frames, axis=0)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    audio_i16 = np.int16(audio/peak*32767) if peak>0 else np.int16(audio)

    out_wav = Path("record.wav")
    wav_write(out_wav, SAMPLE_RATE, audio_i16)
    step(f"✅ บันทึกไฟล์ {out_wav.resolve()}")
    return str(out_wav)

def transcribe(path, stt):
    step("🔎 ถอดเสียง ...")
    texts = []
    segments, info = stt.transcribe(path, language=LANGUAGE, vad_filter=True)
    for seg in segments:
        print(f"[{seg.start:.2f}→{seg.end:.2f}] {seg.text}", flush=True)
        texts.append(seg.text)
    text = "".join(texts).strip()
    print("📄 ข้อความ:", text, flush=True)
    return text

# --------------------- Dobot helpers ---------------------
import serial.tools.list_ports
try:
    from pydobot import Dobot
except Exception as e:
    Dobot = None
    print("⚠️ pydobot import ไม่สำเร็จ:", e, flush=True)

def autodetect_port():
    ports = list(serial.tools.list_ports.comports())
    # เลือกอย่างง่าย: ถ้ามีอุปกรณ์เดียว ก็เอาเลย
    if not ports: return None
    # พยายามเลือกตัวที่เป็น USB-to-UART
    for p in ports:
        d = p.description.upper()
        if "USB" in d or "UART" in d or "SILICON" in d or "CP210" in d or "CH340" in d or "FTDI" in d:
            return p.device
    return ports[0].device

def clamp(v, vmin, vmax): return max(vmin, min(v, vmax))

def safe_bound(x, y, z):
    return (clamp(x, X_MIN, X_MAX),
            clamp(y, Y_MIN, Y_MAX),
            clamp(z, Z_MIN, Z_MAX))

robot = None
def connect_robot():
    """มินิมอล: ไม่ตั้ง speed/acc ตอนต่อ เพื่อลดโอกาส error index out of range"""
    global robot
    if Dobot is None:
        raise RuntimeError("pydobot ใช้งานไม่ได้ (import ไม่สำเร็จ)")

    port = COM_PORT or autodetect_port()
    ports = [f"{p.device} | {p.description}" for p in serial.tools.list_ports.comports()]
    print("🔌 COM ports:", ", ".join(ports) if ports else "(none)", flush=True)
    if port is None:
        raise RuntimeError("หา COM พอร์ตไม่เจอ กรุณาต่อ USB และดูหมายเลขใน Device Manager")

    print(f"🤖 เชื่อมต่อ Dobot ที่ {port} ...", flush=True)
    robot = Dobot(port=port, verbose=False)

    # ทดสอบ pose ก่อน
    p = get_pose()
    print(f"✅ เชื่อมต่อสำเร็จ pose={p}", flush=True)

    # ยก Z ขึ้นถ้าต่ำเกินไป
    x,y,z,r = p
    try:
        safe_z = max(z, 50.0)
        goto_xyzr(x, y, safe_z, r, wait=True)
    except Exception:
        pass

    # กลับ HOME สูง ๆ
    try:
        goto_xyzr(*HOME_POS, wait=True)
        print("🏠 ไป HOME เรียบร้อย", flush=True)
    except Exception as e:
        print("⚠️ ไป HOME ไม่ได้:", e, flush=True)

def get_pose():
    for attr in ("pose", "get_pose", "get_pose_as_dict"):
        f = getattr(robot, attr, None)
        if callable(f):
            p = f()
            if isinstance(p, dict):
                return (p.get("x"), p.get("y"), p.get("z"), p.get("r"))
            if isinstance(p, (list, tuple)) and len(p) >= 4:
                return (p[0], p[1], p[2], p[3])
    raise RuntimeError("หาเมธอดอ่าน pose ไม่เจอใน pydobot เวอร์ชันนี้")

def goto_xyzr(x, y, z, r=0.0, wait=True):
    x, y, z = safe_bound(x, y, z)
    # รองรับหลายชื่อเมธอดตามเวอร์ชัน
    for name in ("move_to", "go", "move_to_xyz"):
        f = getattr(robot, name, None)
        if callable(f):
            try:
                return f(x, y, z, r, wait=wait)
            except TypeError:
                return f(x, y, z, r)
    # Fallback: ถ้ามี PTP ตรง
    f = getattr(robot, "set_ptp_cmd", None)
    if callable(f):
        return f(x, y, z, r, mode=1, wait=wait)

def move_rel(dx=0, dy=0, dz=0, dr=0, wait=True):
    x, y, z, r = get_pose()
    return goto_xyzr(x + dx, y + dy, z + dz, r + dr, wait=wait)

def suction(on: bool):
    for name in ("suck", "set_end_effector_suction_cup"):
        f = getattr(robot, name, None)
        if callable(f):
            try:
                return f(on)
            except TypeError:
                try:
                    return f(enable=on, on=on)
                except Exception:
                    pass
    print("⚠️ ไม่พบเมธอดดูด/ปล่อย suction ใน pydobot เวอร์ชันนี้")

def go_home():
    return goto_xyzr(*HOME_POS, wait=True)

# -------------------- NLP → Action --------------------
import re
TH_NUM = {"ศูนย์":0,"หนึ่ง":1,"สอง":2,"สาม":3,"สี่":4,"ห้า":5,"หก":6,"เจ็ด":7,"แปด":8,"เก้า":9,"สิบ":10}
def parse_number(text):
    m = re.search(r"(\d+)", text)
    if m: return int(m.group(1))
    for k,v in TH_NUM.items():
        if k in text: return v
    return None

armed = False
estop = False

def nlp_to_action(text):
    t = text.strip().replace(" ", "")
    t_raw = text

    if "หยุดฉุกเฉิน" in t or "emergencystop" in t_raw.lower():
        return {"type":"ESTOP"}
    if "เริ่มควบคุม" in t or "ปลดล็อก" in t:
        return {"type":"ARM", "on": True}
    if "หยุดควบคุม" in t or "ล็อก" in t:
        return {"type":"ARM", "on": False}

    if "ดูด" in t or "vacuumon" in t_raw.lower():
        return {"type":"SUCTION", "on": True}
    if "ปล่อย" in t or "vacuumoff" in t_raw.lower():
        return {"type":"SUCTION", "on": False}

    if "โฮม" in t or "home" in t_raw.lower() or "กลับบ้าน" in t:
        return {"type":"HOME"}

    step = STEP_MM
    n = parse_number(t_raw)
    if n is not None and n > 0:
        step = float(n)

    if "ขึ้น" in t:    return {"type":"MOVE", "dx":0, "dy":0, "dz": +step}
    if "ลง" in t:      return {"type":"MOVE", "dx":0, "dy":0, "dz": -step}
    if "ซ้าย" in t:    return {"type":"MOVE", "dx":0, "dy": +step, "dz":0}
    if "ขวา" in t:     return {"type":"MOVE", "dx":0, "dy": -step, "dz":0}
    if "หน้า" in t or "ไปข้างหน้า" in t:
                       return {"type":"MOVE", "dx": +step, "dy":0, "dz":0}
    if "หลัง" in t or "ถอย" in t:
                       return {"type":"MOVE", "dx": -step, "dy":0, "dz":0}
    if "หมุนซ้าย" in t:
                       return {"type":"MOVE", "dx":0, "dy":0, "dz":0, "dr": +5.0}
    if "หมุนขวา" in t:
                       return {"type":"MOVE", "dx":0, "dy":0, "dz":0, "dr": -5.0}

    return {"type":"UNKNOWN", "raw": t_raw}

def execute_action(act):
    global armed, estop
    t = act.get("type")

    if t == "ESTOP":
        estop = True
        print("🛑 E-STOP! ระงับการเคลื่อนที่", flush=True)
        return
    if t == "ARM":
        armed = bool(act.get("on"))
        print(f"🔒 โหมดควบคุม = {armed}", flush=True)
        return
    if estop:
        print("⛔ อยู่ในสถานะ E-STOP (รีสตาร์ทโปรแกรมเพื่อเริ่มใหม่)", flush=True)
        return
    if not armed:
        print("🔔 ยังไม่ปลดล็อก (พูดว่า 'เริ่มควบคุม' ก่อน)", flush=True)
        return

    if t == "SUCTION":
        on = bool(act.get("on"))
        suction(on)
        print(f"🧲 Suction = {on}", flush=True)
        return

    if t == "HOME":
        print("🏠 กลับ HOME", flush=True)
        go_home()
        return

    if t == "MOVE":
        dx = float(act.get("dx",0)); dy = float(act.get("dy",0))
        dz = float(act.get("dz",0)); dr = float(act.get("dr",0))
        print(f"➡️ MOVE: dx={dx}, dy={dy}, dz={dz}, dr={dr}", flush=True)
        move_rel(dx, dy, dz, dr, wait=True)
        x,y,z,r = get_pose()
        print(f"📌 pose -> x={x:.1f}, y={y:.1f}, z={z:.1f}, r={r:.1f}", flush=True)
        return

    if t == "UNKNOWN":
        print(f"❓ ไม่เข้าใจคำสั่ง: {act.get('raw')}", flush=True)

# ------------------------- Main -------------------------
def main():
    # 1) โหลดโมเดลก่อน (โชว์สถานะชัด)
    stt = load_stt_model()

    # 2) เชื่อมต่อหุ่น — ถ้าพัง จะบอกสาเหตุแล้วจบ
    print("🤖 เชื่อมต่อหุ่น ...", flush=True)
    try:
        connect_robot()
    except Exception as e:
        print("❌ เชื่อมต่อ Dobot ไม่สำเร็จ:", e, flush=True)
        print("👉 ตรวจสอบ COM_PORT / สาย USB / ไดรเวอร์ แล้วรันใหม่", flush=True)
        return

    print("\n✅ พร้อมรับคำสั่งเสียงแล้ว")
    print("   - 'เริ่มควบคุม' → ปลดล็อก", flush=True)
    print("   - 'ขึ้น 10', 'ซ้าย 5', 'หน้า 10', 'ดูด', 'ปล่อย', 'โฮม'", flush=True)
    print("   - 'หยุดฉุกเฉิน' เพื่อหยุดทันที", flush=True)

    while True:
        try:
            input("\nกด Enter เพื่ออัดเสียง (Ctrl+C เพื่อออก): ")
        except KeyboardInterrupt:
            print("\n👋 ออกโปรแกรม"); break

        path = record_once(CHUNK_SECONDS)
        if not path:
            continue

        text = transcribe(path, stt)
        if not text:
            print("⚠️ ไม่ได้ยินคำพูด ลองใหม่", flush=True)
            continue

        act = nlp_to_action(text)
        execute_action(act)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 ออกโปรแกรม", flush=True)
    except Exception as e:
        import sys
        print("❌ ERROR:", e, file=sys.stderr, flush=True)
