# simulator to generate synthetic data for testing 
# uses some standar probabilistic randomized functions like guass and non-linear functions like sin and cos 
import csv
import math
import random
import time
from datetime import datetime, timezone 

OUTPUT_FILE = "simulated_sensor_data.csv"
SAMPLE_RATE_HZ = 25
DURATION_SECONDS = 30 

def generate_sample(t): 
    heart_rate = 80 + 3 * math.sin(t/8) 
    frequency = heart_rate/60  # BPM 
    pulse = max(0, math.sin(2*math.pi * frequency * t)) 

    red = 1000 + 80 * pulse + random.gauss(0, 5) 
    infrared = 100 + 110 * pulse + random.gauss(0, 5) 

    ax = 0.03 * math.sin(t) + random.gauss(0, 0.01) 
    ay = 0.02 * math.cos(t) + random.gauss(0, 0.01) 
    az = 0.1 + random.gauss(0, 0.02) 


    return {
        "timestamp" : datetime.now(timezone.utc).isoformat(), 
        "elapsed_seconds": round(t, 3), 
        "ppg_red": round(red, 2), 
        "ppg_ir": round(infrared, 2), 
        "accel_x_g": round(ax, 4), 
        "accel_y_g": round(ay, 4), 
        "accel_z_g": round(az, 4), 
        "synthetic": True,    
        }


with open(OUTPUT_FILE, "w", newline = "") as file: 
    writer = csv.DictWriter(file, fieldnames = ["timestamp", "elapsed_seconds", "ppg_red", "ppg_ir", "accel_x_g", "accel_y_g", "accel_z_g", "synthetic",],)
    writer.writeheader() 

    for i in range(SAMPLE_RATE_HZ * DURATION_SECONDS): 
        sample = generate_sample(i / SAMPLE_RATE_HZ)
        writer.writerow(sample) 


print(f"Generated {SAMPLE_RATE_HZ * DURATION_SECONDS} samples")
print(f"Saved to {OUTPUT_FILE}") 
