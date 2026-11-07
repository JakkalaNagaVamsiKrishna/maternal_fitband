# Maternal Health Fitband — Complete Hardware Implementation Guide
**Deploying the ESP32 + MAX30102 + MPU6050 with TinyML & BLE Wireless**

---

## 1. What This Folder Contains
This folder brings together all the required production files for physical hardware implementation, resolving the missing components and inconsistencies from the original GitHub repository:

```text
hardware_implementation/
├── firmware/
│   ├── maternal_fitband_esp32.ino   # Production ESP32 firmware (MAX30102 + MPU6050 + BLE + TinyML)
│   └── maternal_models_esp32.h      # Embedded TinyML C++ header (Inference on ESP32 in <2.5 ms)
├── models/                          # All 9 pre-trained clinical AI models (.joblib)
│   ├── best_glucose_regressor.joblib
│   ├── best_risk_classifier.joblib
│   ├── maternal_heart_rate_estimator.joblib
│   ├── maternal_activity_classifier.joblib
│   ├── maternal_sleep_classifier.joblib
│   ├── maternal_anomaly_detector.joblib
│   ├── pulse_amplitude_estimator.joblib
│   ├── feature_scaler.joblib
│   └── mpu_act_scaler.joblib
├── live_dashboard.py                # USB serial live monitoring dashboard with clinical alerts
├── wireless_ble_receiver.py         # Wireless Bluetooth Low Energy (BLE) receiver
├── evaluate_inference.py            # Live simulation of maternal profiles (Mother A, B, C)
├── evaluate_mpu_inference.py        # Live simulation of activity & sleep postures
├── verify_predictions.py            # ISO 15197 lab ground-truth vs. model validation
└── HARDWARE_SETUP_GUIDE.md          # This complete setup documentation
```

---

## 2. Inconsistencies Fixed From the GitHub Repo

| Area | Issue in Original GitHub Repo | Fixed in This Implementation Package |
| :--- | :--- | :--- |
| **MPU6050 Driver** | Mixed references to different MPU6050 libraries | Standardized on **`Adafruit_MPU6050`** and **`Adafruit_Sensor`** |
| **Serial Telemetry Format** | Firmware emitted CSV while `serial_reader.py` expected JSON | Emits standardized **JSON packets** matching Python dashboards |
| **Wireless Transfer** | Missing (firmware only streamed over USB wire) | Full **Bluetooth Low Energy (BLE)** advertising (`MaternalFitband`) |
| **Vitals & SpO₂** | SpO₂ calculation formula was missing in code | Real-time Beer-Lambert **Ratio-of-Ratios** calculation added |
| **AI Inference** | Repo had zero machine learning models | **On-chip TinyML header** + **9 trained scikit-learn models** |
| **Maternal Safety Alerts** | No sleep posture or overexertion checks | **Supine back-sleeping alert**, **Overexertion alert**, **Tachycardia alert** |

---

## 3. Hardware Wiring & Pinout

Both the **MAX30102** and **MPU6050** share the ESP32's hardware I2C bus:

| Sensor Pin | ESP32 Pin | Description |
| :--- | :--- | :--- |
| **MAX30102 VCC** | **3.3V** | Power (Do NOT connect to 5V to avoid sensor damage) |
| **MAX30102 GND** | **GND** | Ground |
| **MAX30102 SDA** | **GPIO 21** | I2C Data line (Address `0x57`) |
| **MAX30102 SCL** | **GPIO 22** | I2C Clock line |
| **MPU6050 VCC** | **3.3V** | Power |
| **MPU6050 GND** | **GND** | Ground |
| **MPU6050 SDA** | **GPIO 21** | I2C Data line (Address `0x68`) |
| **MPU6050 SCL** | **GPIO 22** | I2C Clock line |

> **Note:** Breadboard wiring can share the single 3.3V and GND pins of the ESP32 between both sensor modules.

---

## 4. Flashing the Firmware to the ESP32

### Step A: Install Arduino IDE Libraries
1. Open **Arduino IDE** (version 2.x recommended).
2. Go to **Tools → Manage Libraries...** (or `Ctrl+Shift+I` / `Cmd+Shift+I`).
3. Search and install:
   - **`SparkFun MAX3010x Pulse and Proximity Sensor Library`**
   - **`Adafruit MPU6050`** (Click "Install All" to also install *Adafruit Unified Sensor* and *Adafruit BusIO*).

### Step B: Board Setup
1. Go to **Tools → Board → esp32 → ESP32 Dev Module**.
2. Connect your ESP32 to your computer using a micro-USB / USB-C data cable.
3. Select your serial port under **Tools → Port**:
   - Mac: `/dev/cu.usbserial-XXXX` or `/dev/cu.wchusbserialXXXX`
   - Windows: `COM3`, `COM4`, etc.

### Step C: Flash the Code
1. Open `hardware_implementation/firmware/maternal_fitband_esp32.ino`.
2. Ensure `maternal_models_esp32.h` is kept in the same folder.
3. Click the **Upload (➡️)** button.
4. Once flashed, open **Tools → Serial Monitor** at **115200 baud** to see initialization messages.

---

## 5. Running the Software & Live Testing

### Option 1: Live USB Serial Monitor
Connect the ESP32 to your computer and run:
```bash
cd hardware_implementation

# Mac / Linux:
python3 live_dashboard.py --port /dev/tty.usbserial-XXXX

# Windows:
python3 live_dashboard.py --port COM3
```
*   Displays live Heart Rate, SpO₂, Glucose, Exercise Mode, and Sleep Stages.
*   Automatically logs every epoch into `readings.csv`.

---

### Option 2: True Wireless Reception via Bluetooth (BLE)
Your ESP32 automatically advertises as **`MaternalFitband`** over Bluetooth Low Energy.

#### A. From Your Laptop:
```bash
pip install bleak
python3 wireless_ble_receiver.py
```
The script will discover the band wirelessly and print live telemetry in real time.

#### B. From Your Phone (iOS or Android):
1. Download the free **nRF Connect** or **LightBlue** app from the App Store / Google Play Store.
2. Open the app and scan for Bluetooth devices.
3. Connect to **`MaternalFitband`**.
4. Open the Nordic UART TX characteristic (`6E400003-...`) and enable **Notify**.
5. You will see live JSON telemetry streaming directly onto your phone screen!

---

### Option 3: Offline AI Inference & Validation
You can run the full suite of machine learning simulations anytime without hardware connected:

*   **Clinical Profiles Simulation:**
    ```bash
    python3 evaluate_inference.py
    ```
    Tests Mother A (normal), Mother B (high stress anomaly alert), and Mother C (night fasting).

*   **Movement & Sleep Posture Simulation:**
    ```bash
    python3 evaluate_mpu_inference.py
    ```
    Validates Gentle Walking, Overexertion Alerts, Deep Sleep (Left-Lateral posture), and Restless tossing.

*   **Lab Accuracy vs. ISO 15197 Validation:**
    ```bash
    python3 verify_predictions.py
    ```
    Validates non-invasive glucose against true clinical blood lab values across unseen test patients.

*   **Clinical AI Companion & Ob/Gyn Consultation Report:**
    ```bash
    # Run the verification suite across all 4 clinical scenarios:
    python3 maternal_ai_wrapper.py --all-scenarios

    # Analyze a recorded session log and export doctor's SBAR report:
    python3 maternal_ai_wrapper.py --csv readings.csv --export consultation_report.md

    # Run live serial dashboard with real-time AI summary:
    python3 live_dashboard.py --simulate --ai
    ```
    Translates biometrics into 4 pillars: personalized diet, sleep coach, safe exertion, and SBAR Ob/Gyn reports.
