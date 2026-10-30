# Maternal Health Fitband
**Proactive, Continuous Care for Expectant Mothers**

---

## Project Overview

A wearable band for pregnant women that continuously monitors:
- ❤️ Heart Rate & SpO₂ (via PPG — MAX30102)
- 🏃 Activity & Sleep (via Accelerometer — MPU6050)
- 🌡️ Skin Temperature (via MLX90614)

---

## Repository Structure

```
maternal_fitband/
├── firmware/
│   └── maternal_fitband.ino    ← ESP32 Arduino firmware
├── data/
│   ├── serial_reader.py        ← Live USB serial reader + plotter
│   ├── analyze.py              ← Post-session analytics
│   └── readings.csv            ← Auto-generated during testing
├── design flow.pdf             ← System design flow diagram
├── Maternal_Fitband_Project_need_statement.pdf
└── README.md
```

---

## Prototype Setup (USB-Powered)

> **No battery needed.** The ESP32 is powered directly from your laptop via USB.

### Hardware

| Component | Part | I2C Address |
|---|---|---|
| Microcontroller | ESP32 DevKit v1 | — |
| PPG Sensor | MAX30102 | 0x57 |
| Accelerometer | MPU6050 GY-521 | 0x68 |

### Wiring

| Sensor | Pin | ESP32 GPIO |
|---|---|---|
| MAX30102 + MPU6050 | SDA | GPIO 21 |
| MAX30102 + MPU6050 | SCL | GPIO 22 |
| All sensors | VCC | 3.3V |
| All sensors | GND | GND |

---

## Quick Start

### 1. Flash the Firmware

1. Install [Arduino IDE](https://www.arduino.cc/en/software) + ESP32 board support
2. Install libraries via Library Manager:
   - `SparkFun MAX3010x`
   - `MPU6050` by Electronic Cats
   - `ArduinoJson`
3. Open `firmware/maternal_fitband.ino`
4. Select **Tools → Board → ESP32 Dev Module**
5. Select your COM port and click **Upload**

### 2. Verify in Serial Monitor
Open Arduino IDE Serial Monitor at **115200 baud**. You should see:
```json
{"ts":4,"hr":72,"spo2":98,"hr_valid":true,"spo2_valid":true,"activity":"sedentary"}
```

### 3. Run the Live Monitor on Your Laptop

```bash
# Install Python dependencies (once)
pip install pyserial matplotlib pandas

# Linux / Mac
python data/serial_reader.py --port /dev/ttyUSB0

# Windows
python data/serial_reader.py --port COM3
```

A live chart window opens showing real-time HR, SpO₂, and activity.

### 4. Analyze a Recorded Session

```bash
python data/analyze.py
```

---

## Full Roadmap

See the [project roadmap](maternal_fitband_roadmap.md) for:
- Complete sensor wiring diagrams
- PPG signal processing details
- Accelerometer activity classification code
- Full data analytics pipeline (cloud, ML)
- 24-week phased project schedule