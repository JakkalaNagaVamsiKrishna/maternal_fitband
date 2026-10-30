# converting the PPG signals into heart rate using some linear and non-linear conversions and also validate the signals  

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks 

INPUT_FILE = "simulated_sensor_data.csv" 
SAMPLE_RATE_HZ = 25 

df = pd.read_csv(INPUT_FILE)
required = {"elapsed_seconds", "ppg_ir"} 
missing = required - set(df.columns) 

if missing: 
    raise ValueError(f"Missing reuired columns:  {sorted(missing)}")


df = df.sort_values("elapsed_seconds").reset_index(drop = True) 

if df["elapsed_seconds"].duplicated().any(): 
    raise ValueError("Duplicate elapsed timestamps found") 


time_s = df["elapsed_seconds"].to_numpy(dtype=float) 
ppg = df["ppg_ir"].to_numpy(dtype = float) 

if len(ppg) < SAMPLE_RATE_HZ * 5: 
    raise ValueError("Collect at least 5 seconds of data.") 

if not np.isfinite(time_s).all() or not np.isfinite(ppg).all(): 
    raise ValueError("Data contains invalid numeric values") 


dt = np.diff(time_s) 
if np.any(dt <= 0): 
    raise ValueError("Timestamps must increase") 

actual_rate = 1/np.median(dt) 
print(f"Estimated sampling rate : {actual_rate:.2f} Hz") 

min_peak_distance = int(actual_rate * 60/180) 

peaks, properties = find_peaks(ppg, distance = max(1, min_peak_distance), prominence = 25) 

peak_times = time_s[peaks] 
intervals = np.diff(peak_times) 

valid = (intervals >= 60/180) & (intervals <= 60/40) 
valid_intervals = intervals[valid] 

if len(valid_intervals) == 0: 
    print("No usable pulse intervals detected") 
    print("Try adjusting the peak prominence or inspect the plot.") 
else: 
    bpm_values = 60/valid_intervals 
    estimated_bpm = float(np.median(bpm_values)) 

    print(f"Candidate peaks detected: {len(peaks)}") 
    print(f"Usable pulse intervals: {len(valid_intervals)}") 
    print(f"Estimated median heart rate: {estimated_bpm:.1f} BPM") 

    results = pd.DataFrame({"interval_seconds": valid_intervals, "estimated_bpm": bpm_values}) 
    results.to_csv("heart_rate_results.csv", index = False) 
    print("saved interval results to heart_rate_results.csv") 


plt.figure(figsize=(12, 5)) 
plt.plot(time_s, ppg, label="IR PPG") 
plt.plot(time_s[peaks], ppg[peaks], "x", label="Candidate pulse peaks") 
plt.xlabel("Elapsed time (seconds)") 
plt.ylabel("IR signal value") 
plt.title("PPG pulse-peak detection (synthetic data)") 
plt.grid(True) 
plt.legend() 
plt.tight_layout() 
plt.savefig("heart_rate_peaks.png", dpi = 150) 
plt.show() 