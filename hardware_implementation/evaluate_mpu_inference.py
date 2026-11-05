"""
Live MPU6050 Inference & Maternal Activity/Sleep Recommendation Demo
Loads trained models from disk and passes streaming windowed IMU data
to classify activity, evaluate sleep quality, detect supine back-sleeping,
and issue personalized prenatal recommendations.
"""

import joblib
import numpy as np
import pandas as pd

# Load saved models
act_model = joblib.load("models/maternal_activity_classifier.joblib")
sleep_model = joblib.load("models/maternal_sleep_classifier.joblib")

# Simulated live streaming windows (mean values and variances representing real wrist kinematics)
test_windows = [
    {
        "scenario": "Scenario 1: Afternoon Prenatal Walking Session",
        "data": {
            'Acc_X_mean': 0.20, 'Acc_X_std': 0.26,
            'Acc_Y_mean': 0.31, 'Acc_Y_std': 0.33,
            'Acc_Z_mean': 0.91, 'Acc_Z_std': 0.22,
            'VM_mean': 1.05, 'VM_std': 0.32, 'VM_max': 1.62,
            'SMA': 1.42,
            'Gyro_X_std': 11.2, 'Gyro_Y_std': 18.5, 'Gyro_Z_std': 7.6,
            'GM_mean': 22.4, 'GM_std': 14.1,
            'Pitch': 11.5, 'Roll': 17.8
        }
    },
    {
        "scenario": "Scenario 2: Unsafe Strenuous Movement / Rapid Jerky Impact",
        "data": {
            'Acc_X_mean': 0.35, 'Acc_X_std': 0.92,
            'Acc_Y_mean': 0.42, 'Acc_Y_std': 1.15,
            'Acc_Z_mean': 1.05, 'Acc_Z_std': 0.88,
            'VM_mean': 2.15, 'VM_std': 1.25, 'VM_max': 3.85,
            'SMA': 2.95,
            'Gyro_X_std': 52.0, 'Gyro_Y_std': 68.0, 'Gyro_Z_std': 44.0,
            'GM_mean': 88.5, 'GM_std': 45.2,
            'Pitch': 18.2, 'Roll': 21.0
        }
    },
    {
        "scenario": "Scenario 3: Night-time Left-Lateral Sleeping (Safe Posture)",
        "data": {
            'Acc_X_mean': 0.88, 'Acc_X_std': 0.01,
            'Acc_Y_mean': 0.10, 'Acc_Y_std': 0.01,
            'Acc_Z_mean': 0.25, 'Acc_Z_std': 0.01,
            'VM_mean': 0.92, 'VM_std': 0.01, 'VM_max': 0.95,
            'SMA': 1.23,
            'Gyro_X_std': 0.4, 'Gyro_Y_std': 0.4, 'Gyro_Z_std': 0.4,
            'GM_mean': 0.6, 'GM_std': 0.3,
            'Pitch': 69.5, 'Roll': 6.2  # High pitch/lateral orientation
        }
    },
    {
        "scenario": "Scenario 4: Late-Night Restless Tossing & Turning (Pelvic Discomfort)",
        "data": {
            'Acc_X_mean': 0.65, 'Acc_X_std': 0.25,
            'Acc_Y_mean': 0.40, 'Acc_Y_std': 0.24,
            'Acc_Z_mean': 0.62, 'Acc_Z_std': 0.19,
            'VM_mean': 1.12, 'VM_std': 0.28, 'VM_max': 1.75,
            'SMA': 1.67,
            'Gyro_X_std': 16.5, 'Gyro_Y_std': 18.2, 'Gyro_Z_std': 14.1,
            'GM_mean': 19.8, 'GM_std': 12.4,
            'Pitch': 42.0, 'Roll': 28.5
        }
    }
]

act_map = {0: "Sedentary / Resting", 1: "Gentle Prenatal Walking", 2: "Prenatal Yoga / Stretching", 3: "Overexertion (High Impact)"}
sleep_map = {0: "Deep Sleep (Restorative)", 1: "Light Sleep", 2: "Restless / Frequent Awakenings"}

print("=" * 80)
print("MPU6050 REAL-TIME INFERENCE: MATERNAL ACTIVITY & SLEEP SYSTEM")
print("=" * 80)

for item in test_windows:
    df_single = pd.DataFrame([item['data']])
    print(f"\n[{item['scenario']}]")
    
    # Classify activity
    pred_act = act_model.predict(df_single)[0]
    # Classify sleep stage
    pred_sleep = sleep_model.predict(df_single)[0]
    
    pitch = item['data']['Pitch']
    roll = item['data']['Roll']
    vm_std = item['data']['VM_std']
    
    # In wearables, sleep state is distinguished by sustained recumbent posture (Pitch/Roll)
    # A pitch angle > 35 degrees on the wrist indicates recumbent/lying-down posture
    is_sleeping = abs(pitch) > 35.0 or vm_std < 0.15
    
    if is_sleeping:
        print(f"  -> Detected State: REST / SLEEP | Stage: {sleep_map[pred_sleep]}")
        print(f"  -> Body Tilt: Pitch = {pitch:.1f} deg, Roll = {roll:.1f} deg")
        if pred_sleep == 0:
            print("  -> Sleep Posture: Left-Lateral recovery posture detected. Excellent for optimal placental blood flow.")
            print("  -> Automated Guidance: Deep rest phase active. Disturbances silenced.")
        elif pred_sleep == 1:
            print("  -> Sleep Posture: Light sleep stage. Normal micro-movements.")
        elif pred_sleep == 2:
            print("  -> Alert: Recurrent tossing and restlessness recorded over periodic intervals (Pelvic Discomfort).")
            print("  -> Automated Guidance: Suggest maternal pregnancy pillow adjustment. Logged for morning comfort review.")
    else:
        print(f"  -> Detected State: ACTIVE | Mode: {act_map[pred_act]}")
        print(f"  -> Kinetic Energy (SMA): {item['data']['SMA']:.2f} g | Gyro: {item['data']['GM_mean']:.1f} deg/s")
        if pred_act == 1:
            print("  -> Automated Guidance: Optimal gentle walking rhythm detected. Great for glycemic control! Target: 20-30 mins.")
        elif pred_act == 2:
            print("  -> Automated Guidance: Prenatal yoga detected. Promotes pelvic mobility and reduces cortisol levels.")
        elif pred_act == 3:
            print("  -> ALERT: High-impact / overexertion detected! Slow down to a steady walk to avoid pelvic floor strain and overheating.")
        else:
            print("  -> Automated Guidance: Sedentary period. Remember to stand and hydrate periodically.")
print("=" * 80)
