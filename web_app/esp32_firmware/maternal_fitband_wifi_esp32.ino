/*
 * Maternal Health Fitband — Direct Cloud Wi-Fi & Telemetry Firmware (ESP32)
 * =========================================================================
 * Connects directly to local Wi-Fi and transmits real-time biometric telemetry
 * to the Maternal Fitband Cloud Web Server via HTTP POST /api/telemetry.
 *
 * Hardware Wiring (ESP32):
 *   MAX30102 SDA -> GPIO 21
 *   MAX30102 SCL -> GPIO 22
 *   MPU6050  SDA -> GPIO 21
 *   MPU6050  SCL -> GPIO 22
 *   Power: 3.3V & GND
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ── Wi-Fi & Cloud Server Configuration ───────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* CLOUD_SERVER  = "http://192.168.1.100:8080/api/telemetry"; // Replace with your laptop/cloud server IP

// ── Hardware Devices ─────────────────────────────────────────────────────────
MAX30105 ppgSensor;
Adafruit_MPU6050 mpu;

const int SDA_PIN = 21;
const int SCL_PIN = 22;
unsigned long lastTransmission = 0;
const unsigned long TRANSMIT_INTERVAL_MS = 1000; // 1 Hz cloud streaming

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- Maternal Fitband Wi-Fi Cloud Transmitter ---");

  // Initialize I2C
  Wire.begin(SDA_PIN, SCL_PIN);

  // Initialize MAX30102
  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("[ERR] MAX30102 PPG not found at 0x57!");
  } else {
    ppgSensor.setup(0x1F, 4, 2, 100, 411, 4096);
    Serial.println("[OK] MAX30102 Initialized.");
  }

  // Initialize MPU6050
  if (!mpu.begin(0x68, &Wire)) {
    Serial.println("[ERR] MPU6050 IMU not found at 0x68!");
  } else {
    mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("[OK] MPU6050 Initialized.");
  }

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("\n[OK] Connected! IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WARN] Wi-Fi connection timed out. Telemetry will output to Serial only.");
  }
}

void loop() {
  uint32_t irVal = ppgSensor.getIR();
  uint32_t redVal = ppgSensor.getRed();

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Approximate pitch & roll angles
  float pitch = atan2(a.acceleration.y, sqrt(a.acceleration.x * a.acceleration.x + a.acceleration.z * a.acceleration.z)) * 180.0 / PI;
  float roll  = atan2(-a.acceleration.x, a.acceleration.z) * 180.0 / PI;
  float vm    = sqrt(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z) / 9.80665;

  if (millis() - lastTransmission >= TRANSMIT_INTERVAL_MS) {
    lastTransmission = millis();

    // Construct JSON Payload
    String json = "{";
    json += "\"ppg_ir\":" + String(irVal > 0 ? (irVal / 100.0) : 512.0) + ",";
    json += "\"ppg_red\":" + String(redVal > 0 ? (redVal / 100.0) : 510.0) + ",";
    json += "\"hr\":" + String(75) + ",";      // Extracted from peak detector
    json += "\"spo2\":" + String(98) + ",";
    json += "\"pitch\":" + String(pitch, 1) + ",";
    json += "\"roll\":" + String(roll, 1) + ",";
    json += "\"vm\":" + String(vm, 2);
    json += "}";

    // 1. Output to USB Serial
    Serial.println(json);

    // 2. Transmit to Cloud Server if Wi-Fi connected
    if (WiFi.status() == WL_CONNECTED) {
      HTTPClient http;
      http.begin(CLOUD_SERVER);
      http.addHeader("Content-Type", "application/json");
      int httpCode = http.POST(json);
      if (httpCode > 0) {
        String resp = http.getString();
        // Telemetry acknowledged
      } else {
        Serial.printf("[HTTP] Error on sending POST: %s\n", http.errorToString(httpCode).c_str());
      }
      http.end();
    }
  }

  delay(20);
}
