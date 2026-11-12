/*
 * Maternal FitBand — Dedicated USB 100Hz Raw Telemetry Firmware
 * ==============================================================
 * Connects MAX30102/MAX30105 Optical PPG + MPU6050 6-DoF IMU.
 * Emits raw time-series samples at ~100 Hz as newline-delimited JSON
 * directly over USB Serial (115200 baud).
 *
 * All BLE and Wi-Fi code is intentionally excluded for ultra-low latency,
 * maximum loop rate, and complete connection stability during live demos.
 *
 * Hardware Wiring:
 *   ESP32 GPIO 21 (SDA) -> MAX30102 SDA & MPU6050 SDA
 *   ESP32 GPIO 22 (SCL) -> MAX30102 SCL & MPU6050 SCL
 *   ESP32 3.3V          -> MAX30102 VCC & MPU6050 VCC
 *   ESP32 GND           -> MAX30102 GND & MPU6050 GND
 */

#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

MAX30105 ppgSensor;
Adafruit_MPU6050 mpu;

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const uint32_t SERIAL_BAUD = 115200;
const float GRAVITY = 9.80665f;

bool ppgOk = false;
bool mpuOk = false;

uint32_t sampleCount = 0;
uint32_t lastHeartbeatMs = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

  Serial.println("\n# [Maternal FitBand] USB High-Speed Telemetry Streamer (100 Hz)");
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000); // 400kHz Fast I2C

  // 1. Initialize Optical PPG Sensor (MAX30102 / MAX30105)
  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("# [WARN] MAX30102 not detected on I2C. Check wiring.");
    ppgOk = false;
  } else {
    // Configuration: LED brightness 0x1F, 4x sample avg, mode 2 (Red + IR),
    // 100 Hz sample rate, 411 pulse width, 4096 ADC range
    ppgSensor.setup(0x1F, 4, 2, 100, 411, 4096);
    ppgSensor.enableFIFORollOver();
    ppgOk = true;
    Serial.println("# [OK] MAX30102 initialized @ 100 Hz");
  }

  // 2. Initialize MPU6050 Accelerometer
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("# [WARN] MPU6050 not detected at 0x68. Trying 0x69...");
    if (!mpu.begin(0x69, &Wire)) {
      Serial.println("# [WARN] MPU6050 not found. Check wiring.");
      mpuOk = false;
    } else {
      mpuOk = true;
    }
  } else {
    mpuOk = true;
  }

  if (mpuOk) {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("# [OK] MPU6050 initialized (4G / 500dps)");
  }

  Serial.println("# [READY] Streaming raw JSON lines: {\"t\", \"ir\", \"red\", \"ax\", \"ay\", \"az\"}");
}

void loop() {
  uint32_t nowMs = millis();

  // Read latest MPU6050 values
  float ax = 0.0f, ay = 0.0f, az = 1.0f;
  if (mpuOk) {
    sensors_event_t a, g, temp;
    if (mpu.getEvent(&a, &g, &temp)) {
      ax = a.acceleration.x / GRAVITY;
      ay = a.acceleration.y / GRAVITY;
      az = a.acceleration.z / GRAVITY;
    }
  }

  if (ppgOk) {
    ppgSensor.check(); // Service FIFO buffer
    while (ppgSensor.available()) {
      uint32_t red = ppgSensor.getRed();
      uint32_t ir  = ppgSensor.getIR();

      // Emit one compact JSON line per optical sample
      Serial.print("{\"t\":");
      Serial.print(nowMs);
      Serial.print(",\"ir\":");
      Serial.print(ir);
      Serial.print(",\"red\":");
      Serial.print(red);
      Serial.print(",\"ax\":");
      Serial.print(ax, 3);
      Serial.print(",\"ay\":");
      Serial.print(ay, 3);
      Serial.print(",\"az\":");
      Serial.print(az, 3);
      Serial.println("}");

      sampleCount++;
      ppgSensor.nextSample();
    }
  } else {
    // Fallback if PPG not present: stream at ~50 Hz
    Serial.print("{\"t\":");
    Serial.print(nowMs);
    Serial.print(",\"ir\":512,\"red\":512");
    Serial.print(",\"ax\":");
    Serial.print(ax, 3);
    Serial.print(",\"ay\":");
    Serial.print(ay, 3);
    Serial.print(",\"az\":");
    Serial.print(az, 3);
    Serial.println("}");

    delay(20);
  }
}
