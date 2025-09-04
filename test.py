from pydobot import Dobot
import time

PORT = "COM4"
STEP = 5          # หน่วยของ Dobot (มักเป็นมม.)
DWELL = 1.5       # เวลารอแต่ละก้าว (วินาที) ปรับได้ตามความเร็ว

def get_xyzr(device):
    pose = device.pose()          # คืนค่า 8 ตัว: x,y,z,r,j1,j2,j3,j4
    x, y, z, r = pose[:4]         # เอาแค่ 4 ตัวแรก
    return x, y, z, r

def move_abs(device, x, y, z, r):
    device.move_to(x, y, z, r)
    time.sleep(DWELL)

def main():
    device = Dobot(port=PORT, verbose=True)
    try:
        # อ่านจุดเริ่ม
        x0, y0, z0, r0 = get_xyzr(device)
        print("Start pose:", x0, y0, z0, r0)

        # ทดสอบ +X
        move_abs(device, x0 + STEP, y0, z0, r0)
        move_abs(device, x0, y0, z0, r0)

        # ทดสอบ -X
        move_abs(device, x0 - STEP, y0, z0, r0)
        move_abs(device, x0, y0, z0, r0)

        # ทดสอบ +Y
        move_abs(device, x0, y0 + STEP, z0, r0)
        move_abs(device, x0, y0, z0, r0)

        # ทดสอบ -Y
        move_abs(device, x0, y0 - STEP, z0, r0)
        move_abs(device, x0, y0, z0, r0)

        print("Done. Back to start pose.")
    finally:
        device.close()

if __name__ == "__main__":
    main()
