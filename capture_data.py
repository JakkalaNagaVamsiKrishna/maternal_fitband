# capturing data from the sensors and storing it in a csv file 

import serial 
import csv 
import time 

SERIAL_PORT = "/dev/ttyUSB0" 

BAUD_RATE = 115200 
OUTPUT_FILE = "real_sensor_data.csv" 

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2) 

time.sleep(2) 

print("Connected to ESP32.") 
print("Recording sensor data...") 
print("Press Ctrl+C to stop.") 

try: 
    with open(OUTPUT_FILE, "w", newline="") as file: 
        writer = csv.writer(file) 
        header_written = False 

        while True : 
            line = ser.readline().decode("utf-8", errors="replace").strip() 

            if not line: 
                continue 

            if line.startswith("#"): 
                print(line) 
                continue 

            values = line.split(",") 

            if len(values) != 8: 
                conntinue 

            if values[0] == "timestamp_ms": 
                writer.writerow(values) 
                header_writtem = True 
                print("CSV header recieved.") 
                continue 

            if header_written: 
                writer.writerow(values)
                file.flush() 
                print("Recorded:", line) 

except KeybordInterrupt: 
    print("\n Recording Stopped.") 

finally: 
    ser.close() 
    print(f"Data saved to {OUTPUT_FILE}") 