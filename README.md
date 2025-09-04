# Dobot_voice_control

Voice Control Dobot - README

Python Version

-   แนะนำให้ใช้ Python 3.10 - 3.11
-   ไม่ควรใช้ Python 3.12+ เพราะบางไลบรารี (เช่น pydobot,
    faster-whisper) อาจยังไม่รองรับเต็มที่

Required Files

1.  voice_control.py
    -   ไฟล์หลักสำหรับควบคุม Dobot ด้วยเสียง (Enter เริ่ม/หยุดอัดเสียง)
2.  NLP.py
    -   ไฟล์ที่มีฟังก์ชัน transcribe_wav() สำหรับถอดเสียงจากไฟล์
        record.wav
3.  test.py
    -   ไฟล์ทดสอบเบื้องต้นเพื่อเชื่อมต่อและตรวจสอบการขยับ Dobot

Optional / Supporting Files

-   requirements.txt (ควรสร้างเพื่อรวมรายการไลบรารี เช่น pydobot,
    sounddevice, numpy, faster-whisper, scipy)
-   record.wav (ไฟล์เสียงที่บันทึกจะถูกสร้างอัตโนมัติทุกครั้ง)

วิธีสร้าง Virtual Environment (venv)

1.  เปิด Command Prompt หรือ PowerShell ไปยังโฟลเดอร์โปรเจกต์

2.  สร้าง venv ด้วยคำสั่ง:

        python -m venv .venv

3.  เปิดใช้งาน venv:

    -   บน Windows (PowerShell):

            .venv\Scripts\activate

    -   บน Linux/Mac:

            source .venv/bin/activate

4.  ติดตั้ง dependencies จาก requirements.txt:

        pip install -r requirements.txt

วิธีใช้งานโดยสรุป

1.  ติดตั้ง Python 3.10 หรือ 3.11

2.  สร้าง virtual environment และติดตั้ง dependencies จาก
    requirements.txt

3.  เชื่อมต่อ Dobot ที่พอร์ต COM4 (แก้ใน voice_control.py ถ้าไม่ใช่)

4.  รันสคริปต์:

        python voice_control.py

5.  ขั้นตอนใช้งาน:

    -   กด Enter เพื่อเริ่มอัดเสียง
    -   พูดคำสั่ง: Left / Right / Up / Down หรือ ซ้าย / ขวา / ขึ้น / ลง
    -   กด Enter อีกครั้งเพื่อหยุดอัด
    -   Dobot จะขยับ 10 หน่วยตามทิศทางที่สั่ง

หมายเหตุ

-   หากเจอ error index out of range ตอนเชื่อมต่อ ให้รันใหม่อีกครั้ง
    เนื่องจากเป็นปัญหาการ handshake ของพอร์ต
-   ควรปิดโปรแกรมอื่น ๆ ที่อาจใช้พอร์ตเดียวกัน (เช่น Serial Monitor)
