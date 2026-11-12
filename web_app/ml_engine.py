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
import json
import math
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# Set True only after the anomaly model is validated on real data (it flagged every test scenario)
ENABLE_ANOMALY_FLAG = False

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")


class MaternalMLEngine:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.schema = self._load_schema()
        self._load_models()

    def _load_schema(self) -> Dict[str, Any]:
        schema_path = os.path.join(self.models_dir, "feature_schema.json")
        if os.path.exists(schema_path):
            try:
                with open(schema_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Failed reading feature_schema.json: {e}")
        return {}

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
        gender = 1  # Female (matches train_on_fitband_data.py encoding)
        gestational_week = int(profile.get('gestational_week', 28))
        trimester = 1 if gestational_week <= 13 else (2 if gestational_week <= 27 else 3)
        meal_context = profile.get('meal_context', 'postprandial_1h')

        # Optical PPG features from windowing processor (or fallback defaults if raw stream starting)
        hr = float(vitals.get('hr', vitals.get('heart_rate', 76.0)))
        spo2 = int(vitals.get('spo2', 98))
        ppg_signal = float(vitals.get('ppg_signal', vitals.get('ppg_ir', 512.0)))
        sys_peak = float(vitals.get('systolic_peak', ppg_signal + 9.0))
        dia_peak = float(vitals.get('diastolic_peak', ppg_signal - 5.0))
        pulse_area = float(vitals.get('pulse_area', 390.0))
        pulse_amp = float(vitals.get('pulse_amp', max(1.0, sys_peak - dia_peak)))
        pulse_mean = float(vitals.get('pulse_mean', ppg_signal))
        pulse_std = float(vitals.get('pulse_std', max(1.0, pulse_amp * 0.4)))
        ppi_mean = float(vitals.get('ppi_mean', 60000.0 / (hr + 1e-5)))
        ppi_std = float(vitals.get('ppi_std', 25.0))

        # IMU kinematics from windowing processor
        ax_mean = float(vitals.get('Acc_X_mean', vitals.get('ax_g', 0.15)))
        ax_std  = float(vitals.get('Acc_X_std', 0.05))
        ay_mean = float(vitals.get('Acc_Y_mean', vitals.get('ay_g', 0.20)))
        ay_std  = float(vitals.get('Acc_Y_std', 0.05))
        az_mean = float(vitals.get('Acc_Z_mean', vitals.get('az_g', 0.95)))
        az_std  = float(vitals.get('Acc_Z_std', 0.05))

        vm_mean = float(vitals.get('VM_mean', vitals.get('vm_mean', vitals.get('vm', 1.02))))
        vm_std  = float(vitals.get('VM_std', vitals.get('vm_std', 0.08)))
        vm_max  = float(vitals.get('VM_max', vm_mean + vm_std * 2.0))
        sma     = float(vitals.get('SMA', vitals.get('sma', 1.25)))
        pitch   = float(vitals.get('Pitch', vitals.get('pitch', 15.0)))
        roll    = float(vitals.get('Roll', vitals.get('roll', 20.0)))
        gm_mean = float(vitals.get('GM_mean', vitals.get('gm_mean', 12.0)))
        gm_std  = float(vitals.get('GM_std', gm_mean * 0.6))
        gx_std  = float(vitals.get('Gyro_X_std', gm_mean * 0.4))
        gy_std  = float(vitals.get('Gyro_Y_std', gm_mean * 0.5))
        gz_std  = float(vitals.get('Gyro_Z_std', gm_mean * 0.3))

        # --- 1. Derived Hemodynamic Features ---
        bmi = weight_kg / ((height_cm / 100.0) ** 2)
        peak_ratio = dia_peak / (sys_peak + 1e-5)
        norm_pulse_area = pulse_area / (pulse_amp + 1e-5)
        cardiac_index_proxy = (hr * pulse_amp) / 100.0

        # --- 2. Glucose & Glycemic Risk Estimation ---
        est_glucose = 100.0
        glycemic_risk_tier = 0  # 0: Normal, 1: Borderline, 2: High
        glycemic_label = "Optimal Glycemic Range (<110 mg/dL)"
        is_anomaly = False

        if self.models.get('glucose_reg') and self.scalers.get('glucose'):
            try:
                gluc_cols = ['PPG_Signal', 'Heart_Rate', 'Systolic_Peak', 'Diastolic_Peak',
                             'Pulse_Area', 'Pulse_Amplitude', 'Cardiac_Index_Proxy']
                feat_dict = {
                    'PPG_Signal': ppg_signal, 'Heart_Rate': hr,
                    'Systolic_Peak': sys_peak, 'Diastolic_Peak': dia_peak,
                    'Pulse_Area': pulse_area, 'Pulse_Amplitude': pulse_amp,
                    'Cardiac_Index_Proxy': cardiac_index_proxy
                }
                df_gluc = pd.DataFrame([feat_dict])[gluc_cols]
                df_gluc_scaled = self.scalers['glucose'].transform(df_gluc)
                # Ridge was trained without feature names — pass numpy array
                est_glucose = float(self.models['glucose_reg'].predict(
                    df_gluc_scaled if not isinstance(df_gluc_scaled, pd.DataFrame) else df_gluc_scaled.values
                )[0])
            except Exception as e:
                # Fallback formula
                est_glucose = 115.4 + 0.048 * (ppg_signal - 511.0) + 0.021 * (hr - 77.0) + 0.085 * (bmi - 24.0)

        # Risk classifier & anomaly detector (14 features)
        if self.models.get('risk_clf') and self.scalers.get('feature'):
            try:
                feat14 = pd.DataFrame([{
                    'PPG_Signal': ppg_signal, 'Heart_Rate': hr,
                    'Systolic_Peak': sys_peak, 'Diastolic_Peak': dia_peak,
                    'Pulse_Area': pulse_area, 'Pulse_Amplitude': pulse_amp,
                    'Peak_Ratio': peak_ratio, 'Normalized_Pulse_Area': norm_pulse_area,
                    'BMI': bmi, 'Cardiac_Index_Proxy': cardiac_index_proxy,
                    'Age': age, 'Gender': gender, 'Height': height_cm, 'Weight': weight_kg
                }])
                df14 = pd.DataFrame(self.scalers['feature'].transform(feat14), columns=feat14.columns)
                glycemic_risk_tier = int(self.models['risk_clf'].predict(df14)[0])
                if self.models.get('anomaly_det'):
                    is_anomaly = bool(self.models['anomaly_det'].predict(df14)[0] == -1)
            except Exception as e:
                pass

        # Bound glucose realistically
        est_glucose = max(45.0, min(320.0, est_glucose))
        if est_glucose < 110:
            glycemic_risk_tier = 0
            glycemic_label = "Optimal Glycemic Target"
        elif est_glucose <= 139:
            glycemic_risk_tier = 1
            glycemic_label = "Mild Postprandial Elevation"
        else:
            glycemic_risk_tier = 2
            glycemic_label = "Elevated Glycemic Alert"

        # --- 3. Blood Pressure & Hypertension Classification ---
        sbp = None
        dbp = None
        htn_status = "Unavailable"

        if self.models.get('sbp_reg') and self.models.get('dbp_reg') and self.scalers.get('bp'):
            try:
                bp_dict = {
                    'age': age, 'gender': gender, 'bmi': bmi,
                    'pulse_mean': pulse_mean, 'pulse_std': pulse_std,
                    'pulse_amp': pulse_amp, 'pulse_auc': pulse_area,
                    'ppi_mean': ppi_mean,
                    'ppi_std': ppi_std,
                    'heart_rate': hr
                }
                df_bp = pd.DataFrame([bp_dict])
                df_bp_scaled = self.scalers['bp'].transform(df_bp)
                # BP models were trained without feature names — use numpy array
                if isinstance(df_bp_scaled, pd.DataFrame):
                    df_bp_scaled = df_bp_scaled.values
                sbp = int(round(float(self.models['sbp_reg'].predict(df_bp_scaled)[0])))
                dbp = int(round(float(self.models['dbp_reg'].predict(df_bp_scaled)[0])))
                if self.models.get('htn_clf'):
                    htn_pred = self.models['htn_clf'].predict(df_bp_scaled)[0]
                    htn_status = str(htn_pred)
            except Exception:
                pass

        bp_available = sbp is not None and dbp is not None
        if not bp_available:
            sbp, dbp = 118, 76  # Safe placeholder
            htn_status = "Calibrating (Estimating)"
        else:
            sbp = max(75, min(210, sbp))
            dbp = max(45, min(130, dbp))
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

        # Recumbent posture indicator:
        # Tilt angle > 30 deg or minimal movement at resting HR
        is_recumbent = (abs(pitch) > 30.0) or (abs(roll) > 30.0) or (vm_std < 0.08 and hr < 85)

        if self.models.get('activity_clf') and self.models.get('sleep_clf'):
            try:
                imu_dict = {
                    'Acc_X_mean': ax_mean, 'Acc_X_std': ax_std,
                    'Acc_Y_mean': ay_mean, 'Acc_Y_std': ay_std,
                    'Acc_Z_mean': az_mean, 'Acc_Z_std': az_std,
                    'VM_mean': vm_mean, 'VM_std': vm_std, 'VM_max': vm_max,
                    'SMA': sma,
                    'Gyro_X_std': gx_std, 'Gyro_Y_std': gy_std, 'Gyro_Z_std': gz_std,
                    'GM_mean': gm_mean, 'GM_std': gm_std,
                    'Pitch': pitch, 'Roll': roll
                }
                df_imu = pd.DataFrame([imu_dict])
                activity_code = int(self.models['activity_clf'].predict(df_imu)[0])
                act_names = {
                    0: "Sedentary / Resting",
                    1: "Gentle Prenatal Walking",
                    2: "Prenatal Yoga / Stretching",
                    3: "Overexertion (High Impact)"
                }
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
        # Recumbent and both pitch and roll are near horizontal (< 20 deg)
        if is_recumbent and abs(pitch) < 20.0 and abs(roll) < 20.0:
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
        elif est_glucose >= 180:
            if overall_risk != "CRITICAL": overall_risk = "HIGH"
            risk_flags.append("HIGH_GLYCEMIC_SURGE")
        elif est_glucose >= 140:
            if overall_risk == "LOW": overall_risk = "MEDIUM"
            risk_flags.append("POSTPRANDIAL_GLYCEMIC_SURGE")

        if hr > 120:
            if overall_risk != "CRITICAL": overall_risk = "HIGH"
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

        if is_anomaly and ENABLE_ANOMALY_FLAG:
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
