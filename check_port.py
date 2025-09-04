# check_port.py
from pydobot import Dobot
import serial
import time

PORT = "COM4"

def check_connection(port=PORT):
    try:
        device = Dobot(port=port, verbose=False)
        pose = device.pose()
        device.close()
        print(f"✅ Connected to Dobot at {port}, pose={pose[:4]}")
        return True
    except (serial.SerialException, Exception) as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ Dobot ได้ที่ {port}: {e}")
        return False

if __name__ == "__main__":
    if check_connection():
        print("พร้อมใช้งาน")
    else:
        print("ตรวจสอบสาย/ไดรเวอร์/พอร์ต ก่อนเริ่มใหม่")
