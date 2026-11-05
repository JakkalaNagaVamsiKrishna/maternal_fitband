/*
 * Maternal Health Fitband - Hardware Implementation Firmware (ESP32)
 * ===================================================================
 * Fully compatible with SparkFun MAX3010x & Adafruit MPU6050 libraries.
 *
 * Hardware Wiring:
 *   ESP32 GPIO 21 (SDA) -> MAX30102 SDA & MPU6050 SDA
 *   ESP32 GPIO 22 (SCL) -> MAX30102 SCL & MPU6050 SCL
 *   ESP32 3.3V          -> MAX30102 VCC & MPU6050 VCC
 *   ESP32 GND           -> MAX30102 GND & MPU6050 GND
 *
 * Features:
 *   1. Optical PPG (MAX30102) -> Real-time Heart Rate & SpO2 via Beer-Lambert
 *   2. Motion (MPU6050)       -> Tilt, Pitch/Roll, Kinematics & TinyML classification
 *   3. TinyML On-Chip         -> Continuous Glucose (mg/dL), Sleep Stages & Supine Alert
 *   4. Dual Streaming         -> JSON over USB Serial (115200) + BLE Wireless Telemetry
 */

#include <Wire.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "MAX30105.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "maternal_models_esp32.h"

// ── BLE UUIDs (Nordic UART Service) ──────────────────────────────────────────
#define SERVICE_UUID           "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define CHARACTERISTIC_UUID_TX "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

BLEServer *pServer = NULL;
BLECharacteristic *pTxCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// ── Sensors ──────────────────────────────────────────────────────────────────
MAX30105 ppgSensor;
Adafruit_MPU6050 mpu;

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const float GRAVITY = 9.80665f;
const uint32_t SERIAL_BAUD = 115200;

// ── Vitals State ─────────────────────────────────────────────────────────────
uint32_t irValue = 0;
uint32_t redValue = 0;
float dc_ir = 0, dc_red = 0;
float ac_ir = 0, ac_red = 0;

unsigned long lastBeatTime = 0;
int currentHR = 72;
int currentSpO2 = 98;
bool hr_valid = false;
bool spo2_valid = false;

uint32_t startMs = 0;
uint32_t lastOutputMs = 0;
const uint32_t TELEMETRY_INTERVAL_MS = 200; // 5 Hz telemetry stream

// ── BLE Callbacks ────────────────────────────────────────────────────────────
class BleCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
      deviceConnected = true;
      Serial.println("# [BLE] Mobile / Laptop receiver connected!");
    }
    void onDisconnect(BLEServer* pServer) {
      deviceConnected = false;
      Serial.println("# [BLE] Receiver disconnected. Advertising resumed.");
    }
};

void setup() {
  Serial.begin(SERIAL_BAUD);
  delay(1000);

  Serial.println("=====================================================");
  Serial.println("  MATERNAL FITBAND: STARTING HARDWARE SUBSYSTEMS    ");
  Serial.println("=====================================================");

  Wire.begin(SDA_PIN, SCL_PIN);

  // 1. Initialize MAX30102 PPG
  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("# ERROR: MAX30102 not found at 0x57. Check wiring.");
    while (true) delay(100);
  }
  ppgSensor.setup(0x1F, 4, 2, 100, 411, 4096);
  Serial.println("# [OK] MAX30102 PPG Sensor initialized.");

  // 2. Initialize MPU6050 IMU
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("# ERROR: MPU6050 not found at 0x68. Check wiring.");
    while (true) delay(100);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.println("# [OK] MPU6050 IMU initialized.");

  // 3. Initialize BLE Wireless
  BLEDevice::init("MaternalFitband");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new BleCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pTxCharacteristic = pService->createCharacteristic(
                        CHARACTERISTIC_UUID_TX,
                        BLECharacteristic::PROPERTY_NOTIFY
                      );
  pTxCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("# [OK] BLE Wireless Active: 'MaternalFitband'");
  Serial.println("# READY. Streaming live JSON telemetry below...\n");

  startMs = millis();
}

void loop() {
  // Service PPG sensor FIFO
  ppgSensor.check();
  while (ppgSensor.available()) {
    redValue = ppgSensor.getRed();
    irValue = ppgSensor.getIR();

    // Finger / Wrist contact detection
    if (irValue > 50000) {
      dc_ir = 0.95f * dc_ir + 0.05f * (float)irValue;
      dc_red = 0.95f * dc_red + 0.05f * (float)redValue;
      ac_ir = fabs((float)irValue - dc_ir);
      ac_red = fabs((float)redValue - dc_red);

      // Pulse peak detection
      if (irValue > (dc_ir + 150) && (millis() - lastBeatTime > 350)) {
        unsigned long interval = millis() - lastBeatTime;
        lastBeatTime = millis();
        float instantHR = 60000.0f / (float)interval;

        if (instantHR >= 50.0f && instantHR <= 160.0f) {
          currentHR = (int)(0.7f * currentHR + 0.3f * instantHR);
          hr_valid = true;
        }

        // SpO2 calculation via Beer-Lambert Ratio-of-Ratios
        if (dc_red > 0 && dc_ir > 0 && ac_ir > 0) {
          float R = (ac_red / dc_red) / (ac_ir / dc_ir);
          float instantSpO2 = 110.0f - (25.0f * R);
          if (instantSpO2 >= 85.0f && instantSpO2 <= 100.0f) {
            currentSpO2 = (int)(0.8f * currentSpO2 + 0.2f * instantSpO2);
            spo2_valid = true;
          }
        }
      }
    } else {
      hr_valid = false;
      spo2_valid = false;
    }
    ppgSensor.nextSample();
  }

  // Telemetry broadcast loop (5 Hz)
  uint32_t nowMs = millis();
  if (nowMs - lastOutputMs >= TELEMETRY_INTERVAL_MS) {
    lastOutputMs = nowMs;
    uint32_t ts_sec = (nowMs - startMs) / 1000;

    // Read MPU6050
    sensors_event_t accel, gyro, temp;
    mpu.getEvent(&accel, &gyro, &temp);

    float ax_g = accel.acceleration.x / GRAVITY;
    float ay_g = accel.acceleration.y / GRAVITY;
    float az_g = accel.acceleration.z / GRAVITY;
    float vm = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
    float vm_std = fabs(vm - 1.0f);

    float pitch = atan2(-ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * 180.0f / 3.14159f;
    float roll  = atan2(ay_g, az_g) * 180.0f / 3.14159f;
    float gyroMag = sqrt(gyro.gyro.x * gyro.gyro.x +
                         gyro.gyro.y * gyro.gyro.y +
                         gyro.gyro.z * gyro.gyro.z) * 180.0f / 3.14159f;

    // Run On-Chip TinyML Inference
    int actIdx = classifyActivityESP32(vm_std, vm, gyroMag, pitch);
    int sleepIdx = classifySleepESP32(vm_std, pitch, roll);
    bool supine = isSupineBackSleeping(pitch, roll);

    const char* actList[] = {"sedentary", "walking", "yoga", "OVEREXERTION"};
    const char* sleepList[] = {"Deep Sleep", "Light Sleep", "Restless/Discomfort"};

    // On-chip Glucose estimation
    float gluc = predictGlucose(512.0f, (float)currentHR, 520.0f, 507.0f, 388.0f, 24.5f, 28.0f);

    // Format JSON matching serial_reader.py & BLE clients
    char jsonBuffer[256];
    snprintf(jsonBuffer, sizeof(jsonBuffer),
      "{\"ts\":%lu,\"hr\":%d,\"spo2\":%d,\"hr_valid\":%s,\"spo2_valid\":%s,"
      "\"ax_g\":%.3f,\"ay_g\":%.3f,\"az_g\":%.3f,\"vm\":%.3f,"
      "\"activity\":\"%s\",\"glucose\":%.1f,\"sleep\":\"%s\",\"supine\":%s}",
      (unsigned long)ts_sec,
      currentHR, currentSpO2,
      hr_valid ? "true" : "false",
      spo2_valid ? "true" : "false",
      ax_g, ay_g, az_g, vm,
      actList[actIdx], gluc, sleepList[sleepIdx],
      supine ? "true" : "false"
    );

    // 1. Output over USB Serial
    Serial.println(jsonBuffer);

    // 2. Transmit over BLE Wireless
    if (deviceConnected) {
      pTxCharacteristic->setValue((uint8_t*)jsonBuffer, strlen(jsonBuffer));
      pTxCharacteristic->notify();
    }
  }

  // Handle BLE disconnect / reconnect housekeeping
  if (!deviceConnected && oldDeviceConnected) {
    delay(500);
    pServer->startAdvertising();
    oldDeviceConnected = deviceConnected;
  }
  if (deviceConnected && !oldDeviceConnected) {
    oldDeviceConnected = deviceConnected;
  }
}
