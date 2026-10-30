import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy.signal import butter, sosfiltfilt, find_peaks 

INPUT_FILE = "results/simulated_sensor_data.csv"
OUTPUT_FILE = "results/preprocessed_ppg.csv" 

LOW_CUTOFF = 0.5 
HIGH_CUTOFF = 3.0 
FILTER_ORDER = 3 

df = pd.read_csv(INPUT_FILE) 

required = {"elapsed_seconds", "ppg_ir"} 

if not required.issubset(df.columns): 
    raise ValueError("Required PPG columns are missing. ") 

df = df.sort_values("elapsed_seconds").reset_index(drop=True) 

time_s = df["elapsed_seconds"].to_numpy(dtype=float) 
ppg = df["ppg_ir"].to_numpy(dtype = float) 

if not np.isfinite(time_s).all() or not np.isfinite(ppg).all(): 
    raise ValueError("Invalid numerical values Found")


dt = np.diff(time_s) 

if np.any(dt <= 0): 
    raise ValueError("Timestamps must increase") 

actual_rate = 1/np.median(dt) 

print(f"Sampling rate: {actual_rate:.2f} Hz") 

nyquist = actual_rate/2 

if not (0 < LOW_CUTOFF < HIGH_CUTOFF < nyquist): 
    raise ValueError("Invalid cutoff frequencies for sampling rate.") 


sos = butter(
    FILTER_ORDER,
    [LOW_CUTOFF, HIGH_CUTOFF],
    btype="bandpass", 
    fs = actual_rate, 
    output="sos"
    ) 


filtered_ppg = sosfiltfilt(sos, ppg) 

print("PPG filtering completed. ")

min_distance = max(1, int(actual_rate * 60/180)) 

raw_peaks, _ = find_peaks(
    ppg, 
    distance=min_distance, 
    prominence=25
) 

filtered_peaks, _ = find_peaks(
    filtered_ppg, 
    distance = min_distance, 
    prominence = 1
)

print(f"Raw signal peaks: {len(raw_peaks)}")
print(f"Filtered_signal peaks: {len(filtered_peaks)}") 

output_df = pd.DataFrame({
    "elapsed_seconds:": time_s, 
    "ppg_ir_raw": ppg, 
    "ppg_ir_filtered": filtered_ppg
}) 

output_df.to_csv(OUTPUT_FILE, index=False) 

print(f"Saved processed data to {OUTPUT_FILE}")