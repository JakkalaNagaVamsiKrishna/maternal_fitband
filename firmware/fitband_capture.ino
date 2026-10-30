// capturing data from the sensors using the ESP32
#include "MAX30105.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

MAX30105 ppgSensor;
Adafruit_MPU6050 mpu;

const int SDA_PIN = 21;
const int SCL_PIN = 22;

const float GRAVITY = 9.80665f;
const uint32_t SERIAL_BAUD = 115200;

uint32_t startMs;
uint32_t lastOutputMs = 0;

void setup() {
	Serial.begin(SERIAL_BAUD);
	delay(1000);

	Wire.begin(SDA_PIN, SCL_PIN);

	// Initialize the PPG sensor
	if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
		Serial.println("# ERROR: MAX30102 not found. Check wiring.");
		while (true)
			delay(100);
	}

	// LED brightness, averaging, LED mode,
	// sample rate, pulse width, ADC range
	ppgSensor.setup(0x1F, 4, 2, 100, 411, 4096);

	// Initialize the accelerometer
	if (!mpu.begin(0x68, &Wire)) {
		Serial.println("# ERROR: MPU6050 not found. Check wiring.");
		while (true)
			delay(100);
	}

	mpu.setAccelerometerRange(MPU6050_RANGE_2_G);
	mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

	// CSV header compatible with your Python pipeline
	Serial.println("timestamp_ms,elapsed_seconds,ppg_red,ppg_ir,"
				   "accel_x_g,accel_y_g,accel_z_g,synthetic");

	startMs = millis();
}

void loop() {
	// Keep the sensor FIFO serviced frequently
	ppgSensor.check();

	while (ppgSensor.available()) {
		uint32_t red = ppgSensor.getRed();
		uint32_t ir = ppgSensor.getIR();

		// Read acceleration
		sensors_event_t accel, gyro, temp;
		mpu.getEvent(&accel, &gyro, &temp);

		// Convert acceleration from m/s^2 to units of g
		float ax = accel.acceleration.x / GRAVITY;
		float ay = accel.acceleration.y / GRAVITY;
		float az = accel.acceleration.z / GRAVITY;

		// Timestamp when this sample is read by the ESP32
		uint32_t nowMs = millis();
		float elapsed = (nowMs - startMs) / 1000.0f;

		// Emit one CSV row
		Serial.printf("%lu,%.3f,%lu,%lu,%.5f,%.5f,%.5f,false\n",
					  (unsigned long)nowMs, elapsed, (unsigned long)red,
					  (unsigned long)ir, ax, ay, az);

		ppgSensor.nextSample();
		lastOutputMs = nowMs;
	}

	delay(1);
}
