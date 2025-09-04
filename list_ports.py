import serial.tools.list_ports

print("Found ports:")
for p in serial.tools.list_ports.comports():
    print("-", p.device, "|", p.description)