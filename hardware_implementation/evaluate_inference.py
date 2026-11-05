"""
Inference Demonstration for Maternal Fitband
Simulates real-time sensor inputs from a pregnant mother and passes them through
the trained ML pipeline to generate predictions, clinical alerts, and lifestyle recommendations.
"""

import joblib
import numpy as np
import pandas as pd

# Load saved models
scaler = joblib.load("models/feature_scaler.joblib")
glucose_model = joblib.load("models/best_glucose_regressor.joblib")
risk_model = joblib.load("models/best_risk_classifier.joblib")
bp_model = joblib.load("models/pulse_amplitude_estimator.joblib")
anomaly_model = joblib.load("models/maternal_anomaly_detector.joblib")

# Sample simulated profiles representing different states during pregnancy:
profiles = [
    {
        "description": "Mother A: Postprandial (After lunch), Mild rest, Normal pregnancy profile",
        "PPG_Signal": 512, "Heart_Rate": 76.0, "Systolic_Peak": 521.0, "Diastolic_Peak": 507.0,
        "Pulse_Area": 390.0, "Age": 28, "Gender": 0, "Height": 162, "Weight": 65
    },
    {
        "description": "Mother B: Elevated Stress & Tachycardia (Potential hypertensive / gestational surge)",
        "PPG_Signal": 515, "Heart_Rate": 92.0, "Systolic_Peak": 526.0, "Diastolic_Peak": 505.0,
        "Pulse_Area": 420.0, "Age": 34, "Gender": 0, "Height": 165, "Weight": 78
    },
    {
        "description": "Mother C: Night-time Resting / Fasting State",
        "PPG_Signal": 509, "Heart_Rate": 64.0, "Systolic_Peak": 518.0, "Diastolic_Peak": 507.0,
        "Pulse_Area": 375.0, "Age": 26, "Gender": 0, "Height": 158, "Weight": 58
    }
]

print("=" * 80)
print("MATERNAL FITBAND INFERENCE & CLINICAL RECOMMENDATION SYSTEM")
print("=" * 80)

risk_labels = {0: "Normal (<110 mg/dL)", 1: "Borderline (110-130 mg/dL)", 2: "High Glycemic Risk (>130 mg/dL)"}

for i, p in enumerate(profiles, 1):
    df_single = pd.DataFrame([p])
    
    # Feature calculations
    df_single['Pulse_Amplitude'] = df_single['Systolic_Peak'] - df_single['Diastolic_Peak']
    df_single['Peak_Ratio'] = df_single['Diastolic_Peak'] / (df_single['Systolic_Peak'] + 1e-5)
    df_single['Normalized_Pulse_Area'] = df_single['Pulse_Area'] / (df_single['Pulse_Amplitude'] + 1e-5)
    df_single['BMI'] = df_single['Weight'] / ((df_single['Height'] / 100) ** 2)
    df_single['Cardiac_Index_Proxy'] = (df_single['Heart_Rate'] * df_single['Pulse_Amplitude']) / 100.0
    
    features = [
        'PPG_Signal', 'Heart_Rate', 'Systolic_Peak', 'Diastolic_Peak',
        'Pulse_Area', 'Pulse_Amplitude', 'Peak_Ratio',
        'Normalized_Pulse_Area', 'BMI', 'Cardiac_Index_Proxy',
        'Age', 'Gender', 'Height', 'Weight'
    ]
    
    # Predictions
    pred_glucose = glucose_model.predict(df_single[features])[0]
    pred_risk = risk_model.predict(df_single[features])[0]
    
    bp_features = ['PPG_Signal', 'Heart_Rate', 'Pulse_Area', 'Age', 'Height', 'Weight', 'BMI']
    pred_pulse_amp = bp_model.predict(df_single[bp_features])[0]
    is_anomaly = anomaly_model.predict(df_single[features])[0] == -1
    
    print(f"\n[Case {i}] {p['description']}")
    print(f"  Vitals Input: HR = {p['Heart_Rate']} bpm | BMI = {df_single['BMI'].values[0]:.1f} | PPG = {p['PPG_Signal']}")
    print(f"  -> Predicted Glucose Level: {pred_glucose:.1f} mg/dL")
    print(f"  -> Glycemic Risk Tier:      {risk_labels[pred_risk]}")
    print(f"  -> Est. Pulse Amplitude:    {pred_pulse_amp:.1f} (Vascular compliance)")
    print(f"  -> Physiological Anomaly:   {'WARNING: Anomaly Detected!' if is_anomaly else 'Normal stable rhythm'}")
    
    # Actionable guidance generation
    print("  -> Automated Fitband Guidance:")
    if pred_risk == 2:
        print("     * Diet: Alert: elevated glycemic trend. Prioritize low-GI meals (whole grains, high fiber). Reduce refined sugars.")
        print("     * Exercise: Light 15-minute post-meal stroll recommended once heart rate stabilizes.")
        print("     * Prebirth Care: Log shared to Ob/Gyn portal for oral glucose tolerance follow-up.")
    elif pred_risk == 1:
        print("     * Diet: Balanced meal recommendation; maintain hydration (at least 2.5L water today).")
        print("     * Exercise: 25 minutes prenatal yoga or brisk walking.")
        print("     * Sleep: Target 8 hours; maintain left-lateral sleeping position for optimal fetal blood flow.")
    else:
        print("     * Diet: Glucose within target range. Maintain current balanced diet.")
        print("     * Exercise: Standard moderate prenatal physical activity.")
        print("     * Sleep: Healthy recovery trend detected.")
print("=" * 80)
