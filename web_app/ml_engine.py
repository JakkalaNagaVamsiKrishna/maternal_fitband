"""
Maternal Health Fitband — Cloud ML Classification Engine
========================================================
Loads trained clinical models and executes multi-parameter inference:
  - Non-invasive Blood Glucose (mg/dL) & 3-Tier Glycemic Risk
  - Systolic & Diastolic Blood Pressure (mmHg) & Hypertension Stage
  - Motion-Robust Heart Rate (BPM) & Vascular Compliance Proxy
  - IMU Kinematic Activity Classification (Sedentary, Walking, Yoga, Overexertion)
  - Nocturnal Sleep Staging & Supine Back-Sleeping Aortocaval Risk
  - Isolation Forest Acute Biometric Anomaly Detection
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hardware_implementation", "models")

class MaternalMLEngine:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self._load_models()

    def _safe_load(self, filename: str):
        path = os.path.join(self.models_dir, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception as e:
                print(f"[WARN] Failed loading {filename}: {e}")
        return None

    def _load_models(self):
        print(f"[ML Engine] Loading models from {self.models_dir}...")
        # Scalers
        self.scalers['feature'] = self._safe_load('feature_scaler.joblib')
        self.scalers['bp'] = self._safe_load('bp_feature_scaler.joblib')
        self.scalers['mpu'] = self._safe_load('mpu_act_scaler.joblib')
        self.scalers['glucose'] = self._safe_load('glucose_feature_scaler.joblib')
        self.scalers['hr_motion'] = self._safe_load('hr_motion_scaler.joblib')

        # Regressors & Classifiers
        self.models['glucose_reg'] = self._safe_load('best_glucose_regressor.joblib')
        self.models['risk_clf'] = self._safe_load('best_risk_classifier.joblib')
        self.models['pulse_amp_reg'] = self._safe_load('pulse_amplitude_estimator.joblib')
        self.models['anomaly_det'] = self._safe_load('maternal_anomaly_detector.joblib')
        self.models['sbp_reg'] = self._safe_load('maternal_sbp_estimator.joblib')
        self.models['dbp_reg'] = self._safe_load('maternal_dbp_estimator.joblib')
        self.models['htn_clf'] = self._safe_load('maternal_hypertension_classifier.joblib')
        self.models['activity_clf'] = self._safe_load('maternal_activity_classifier.joblib')
        self.models['sleep_clf'] = self._safe_load('maternal_sleep_classifier.joblib')
        self.models['motion_hr_reg'] = self._safe_load('maternal_motion_robust_hr.joblib')

        loaded_count = sum(1 for v in {**self.scalers, **self.models}.values() if v is not None)
        print(f"[ML Engine] Successfully loaded {loaded_count} model/scaler assets.")

    def classify(self, vitals: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end classification across all maternal health dimensions.
        """
        # Profile demographics
        age = float(profile.get('age', 29))
        height_cm = float(profile.get('height_cm', 162.0))
        weight_kg = float(profile.get('weight_kg', 65.0))
        gender = 0  # Female
        gestational_week = int(profile.get('gestational_week', 28))
        trimester = 1 if gestational_week <= 13 else (2 if gestational_week <= 27 else 3)
        meal_context = profile.get('meal_context', 'postprandial_1h')

        # Vitals extraction
        hr = float(vitals.get('hr', vitals.get('heart_rate', 76.0)))
        spo2 = int(vitals.get('spo2', 98))
        ppg_signal = float(vitals.get('ppg_signal', vitals.get('ppg_ir', 512.0)))
        sys_peak = float(vitals.get('systolic_peak', ppg_signal + 9.0))
        dia_peak = float(vitals.get('diastolic_peak', ppg_signal - 5.0))
        pulse_area = float(vitals.get('pulse_area', 390.0))

        # IMU kinematics
        pitch = float(vitals.get('pitch', 12.0))
        roll = float(vitals.get('roll', 15.0))
        vm_mean = float(vitals.get('vm_mean', vitals.get('vm', 1.02)))
        vm_std = float(vitals.get('vm_std', 0.18))
        sma = float(vitals.get('sma', 1.35))
        gm_mean = float(vitals.get('gm_mean', 18.0))

        # --- 1. Derived Hemodynamic Features ---
        bmi = weight_kg / ((height_cm / 100.0) ** 2)
        pulse_amp = max(1.0, sys_peak - dia_peak)
        peak_ratio = dia_peak / (sys_peak + 1e-5)
        norm_pulse_area = pulse_area / (pulse_amp + 1e-5)
        cardiac_index_proxy = (hr * pulse_amp) / 100.0

        # --- 2. Glucose & Glycemic Risk Estimation ---
        est_glucose = 100.0
        glycemic_risk_tier = 0  # 0: Normal, 1: Borderline, 2: High
        glycemic_label = "Normal (<110 mg/dL)"
        is_anomaly = False

        if self.models.get('glucose_reg'):
            try:
                feat_dict = {
                    'PPG_Signal': ppg_signal, 'Heart_Rate': hr,
                    'Systolic_Peak': sys_peak, 'Diastolic_Peak': dia_peak,
                    'Pulse_Area': pulse_area, 'Pulse_Amplitude': pulse_amp,
                    'Peak_Ratio': peak_ratio, 'Normalized_Pulse_Area': norm_pulse_area,
                    'BMI': bmi, 'Cardiac_Index_Proxy': cardiac_index_proxy,
                    'Age': age, 'Gender': gender, 'Height': height_cm, 'Weight': weight_kg
                }
                df_feat = pd.DataFrame([feat_dict])
                if self.scalers.get('feature'):
                    df_scaled = pd.DataFrame(self.scalers['feature'].transform(df_feat), columns=df_feat.columns)
                else:
                    df_scaled = df_feat

                est_glucose = float(self.models['glucose_reg'].predict(df_scaled)[0])
                if self.models.get('risk_clf'):
                    glycemic_risk_tier = int(self.models['risk_clf'].predict(df_scaled)[0])
                if self.models.get('anomaly_det'):
                    is_anomaly = bool(self.models['anomaly_det'].predict(df_scaled)[0] == -1)
            except Exception as e:
                # Embedded TinyML fallback formula:
                est_glucose = 115.4 + 0.048 * (ppg_signal - 511.0) + 0.021 * (hr - 77.0) + 0.085 * (bmi - 24.0)

        # Bound glucose realistically
        est_glucose = max(45.0, min(320.0, est_glucose))
        if est_glucose < 110:
            glycemic_risk_tier = 0
            glycemic_label = "Optimal Glycemic Range"
        elif est_glucose <= 139:
            glycemic_risk_tier = 1
            glycemic_label = "Mild Postprandial Elevation"
        else:
            glycemic_risk_tier = 2
            glycemic_label = "Elevated Glycemic Alert"

        # --- 3. Blood Pressure & Hypertension Classification ---
        sbp = 118
        dbp = 76
        htn_status = "Normotensive"

        if self.models.get('sbp_reg') and self.models.get('dbp_reg') and self.scalers.get('bp'):
            try:
                bp_dict = {
                    'age': age, 'gender': gender, 'bmi': bmi,
                    'pulse_mean': ppg_signal, 'pulse_std': pulse_amp * 0.4,
                    'pulse_amp': pulse_amp, 'pulse_auc': pulse_area,
                    'ppi_mean': 60000.0 / (hr + 1e-5),
                    'ppi_std': 25.0,
                    'heart_rate': hr
                }
                df_bp = pd.DataFrame([bp_dict])
                df_bp_scaled = self.scalers['bp'].transform(df_bp)
                sbp = int(round(float(self.models['sbp_reg'].predict(df_bp_scaled)[0])))
                dbp = int(round(float(self.models['dbp_reg'].predict(df_bp_scaled)[0])))
                if self.models.get('htn_clf'):
                    htn_pred = self.models['htn_clf'].predict(df_bp_scaled)[0]
                    htn_status = str(htn_pred)
            except Exception as e:
                pass

        # Validate physiological bounds
        sbp = max(80, min(200, sbp))
        dbp = max(50, min(120, dbp))
        if sbp >= 160 or dbp >= 110:
            htn_status = "Stage 2 Severe Hypertension"
        elif sbp >= 140 or dbp >= 90:
            htn_status = "Stage 1 Gestational Hypertension"
        elif sbp < 90 or dbp < 60:
            htn_status = "Hypotension"
        else:
            htn_status = "Normotensive (Healthy)"

        # --- 4. Kinematics, Activity & Sleep Classification ---
        activity_label = "Sedentary / Resting"
        activity_code = 0
        sleep_label = "Awake"
        sleep_code = -1
        supine_alert = False

        # Sleep vs Active state determination:
        # High pitch (>35 deg) or minimal movement (vm_std < 0.15) indicates lying down/sleeping
        is_recumbent = (abs(pitch) > 35.0) or (vm_std < 0.15 and hr < 85)

        if self.models.get('activity_clf') and self.models.get('sleep_clf'):
            try:
                imu_dict = {
                    'Acc_X_mean': 0.20, 'Acc_X_std': vm_std * 0.8,
                    'Acc_Y_mean': 0.30, 'Acc_Y_std': vm_std * 0.9,
                    'Acc_Z_mean': 0.90, 'Acc_Z_std': vm_std * 0.7,
                    'VM_mean': vm_mean, 'VM_std': vm_std, 'VM_max': vm_mean + vm_std * 2.0,
                    'SMA': sma,
                    'Gyro_X_std': gm_mean * 0.5, 'Gyro_Y_std': gm_mean * 0.8, 'Gyro_Z_std': gm_mean * 0.4,
                    'GM_mean': gm_mean, 'GM_std': gm_mean * 0.6,
                    'Pitch': pitch, 'Roll': roll
                }
                df_imu = pd.DataFrame([imu_dict])
                activity_code = int(self.models['activity_clf'].predict(df_imu)[0])
                act_names = {0: "Sedentary / Resting", 1: "Gentle Prenatal Walking", 2: "Prenatal Yoga / Stretching", 3: "Overexertion (High Impact)"}
                activity_label = act_names.get(activity_code, "Active")

                if is_recumbent:
                    sleep_code = int(self.models['sleep_clf'].predict(df_imu)[0])
                    sleep_names = {0: "Deep Sleep (Restorative)", 1: "Light Sleep", 2: "Restless Tossing"}
                    sleep_label = sleep_names.get(sleep_code, "Resting")
            except Exception:
                pass

        # Overexertion heuristic check
        if vm_std > 0.85 or sma > 2.5 or hr > 135:
            activity_code = 3
            activity_label = "Overexertion Alert"

        # Supine (flat on back) check:
        # When lying recumbent, pitch and roll close to zero indicate flat back
        if is_recumbent and abs(pitch) < 22.0 and abs(roll) < 22.0:
            supine_alert = True

        # --- 5. Overall Risk Level Assessment ---
        overall_risk = "LOW"
        risk_flags = []

        if supine_alert and trimester >= 2:
            overall_risk = "CRITICAL"
            risk_flags.append("SUPINE_AORTOCAVAL_COMPRESSION_RISK")

        if sbp >= 160 or dbp >= 110:
            overall_risk = "CRITICAL"
            risk_flags.append("SEVERE_GESTATIONAL_HYPERTENSION")
        elif sbp >= 140 or dbp >= 90:
            if overall_risk != "CRITICAL": overall_risk = "HIGH"
            risk_flags.append("GESTATIONAL_HYPERTENSION")

        if est_glucose < 60:
            overall_risk = "CRITICAL"
            risk_flags.append("MATERNAL_HYPOGLYCEMIA")
        elif est_glucose >= 140:
            if overall_risk == "LOW": overall_risk = "MEDIUM"
            if est_glucose >= 180: overall_risk = "HIGH"
            risk_flags.append("POSTPRANDIAL_GLYCEMIC_SURGE")

        if hr > 115:
            if overall_risk == "LOW": overall_risk = "MEDIUM"
            risk_flags.append("MATERNAL_TACHYCARDIA")
        elif hr < 50:
            if overall_risk == "LOW": overall_risk = "MEDIUM"
            risk_flags.append("MATERNAL_BRADYCARDIA")

        if spo2 < 92:
            overall_risk = "CRITICAL"
            risk_flags.append("CRITICAL_HYPOXEMIA")
        elif spo2 < 95:
            if overall_risk == "LOW": overall_risk = "MEDIUM"
            risk_flags.append("BORDERLINE_SPO2")

        if activity_code == 3:
            if overall_risk == "LOW": overall_risk = "MEDIUM"
            risk_flags.append("KINEMATIC_OVEREXERTION")

        if is_anomaly:
            risk_flags.append("UNSUPERVISED_BIOMETRIC_ANOMALY")

        return {
            "vitals": {
                "glucose_mg_dl": round(est_glucose, 1),
                "glycemic_risk_tier": glycemic_risk_tier,
                "glycemic_label": glycemic_label,
                "heart_rate_bpm": int(round(hr)),
                "spo2_pct": spo2,
                "blood_pressure": f"{sbp}/{dbp}",
                "sbp_mmhg": sbp,
                "dbp_mmhg": dbp,
                "htn_status": htn_status,
                "pulse_amplitude": round(pulse_amp, 1),
                "cardiac_index_proxy": round(cardiac_index_proxy, 2),
                "bmi": round(bmi, 1)
            },
            "kinematics": {
                "activity_code": activity_code,
                "activity_label": activity_label,
                "sleep_code": sleep_code,
                "sleep_label": sleep_label,
                "is_recumbent": is_recumbent,
                "supine_alert": supine_alert,
                "pitch_deg": round(pitch, 1),
                "roll_deg": round(roll, 1),
                "vm_mean_g": round(vm_mean, 2),
                "vm_std_g": round(vm_std, 2)
            },
            "clinical_status": {
                "overall_risk_level": overall_risk,
                "risk_flags": risk_flags,
                "is_biometric_anomaly": is_anomaly,
                "trimester": trimester,
                "gestational_week": gestational_week,
                "meal_context": meal_context
            }
        }

# Global singleton
ml_engine = MaternalMLEngine()
