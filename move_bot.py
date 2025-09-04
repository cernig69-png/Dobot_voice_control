# move_bot.py
from pydobot import Dobot
import sys
import time

PORT = "COM4"
DWELL = 1.2  # เวลารอหลังสั่งขยับ (วินาที)

# แผนที่ทิศทาง -> (dx, dy, dz, dr)
# นิยามนี้: Left = -Y, Right = +Y, Up = +Z, Down = -Z
DIRECTION_DELTAS = {
    "left":  (0, -1,  0, 0),
    "right": (0,  1,  0, 0),
    "up":    (0,  0,  1, 0),
    "down":  (0,  0, -1, 0),
}

def get_xyzr(device):
    pose = device.pose()  # คืน 8 ค่า: x,y,z,r,j1,j2,j3,j4
    x, y, z, r = pose[:4]
    return x, y, z, r

def move_abs(device, x, y, z, r):
    device.move_to(x, y, z, r)
    time.sleep(DWELL)

def main():
    if len(sys.argv) < 2:
        print("Usage: python move_bot.py <Left|Right|Up|Down> [step]")
        sys.exit(1)

    direction = sys.argv[1].strip().lower()
    step = float(sys.argv[2]) if len(sys.argv) >= 3 else 10.0

    if direction not in DIRECTION_DELTAS:
        print(f"Unknown direction: {direction}")
        sys.exit(1)

    dxu, dyu, dzu, dru = DIRECTION_DELTAS[direction]

    device = Dobot(port=PORT, verbose=True)
    try:
        x0, y0, z0, r0 = get_xyzr(device)
        print("Start pose:", x0, y0, z0, r0)

        # คำนวณเป้าหมายแบบ absolute จาก delta * step
        x1 = x0 + dxu * step
        y1 = y0 + dyu * step
        z1 = z0 + dzu * step
        r1 = r0 + dru * step

        move_abs(device, x1, y1, z1, r1)
        print(f"Moved {direction} by {step} units.")
    finally:
        device.close()

if __name__ == "__main__":
    main()
