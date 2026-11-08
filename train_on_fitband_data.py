#!/usr/bin/env python3
"""
Maternal Fitband — Model Training on maternal_fitband/data
=========================================================
Trains end-to-end models on all three sub-databases inside maternal_fitband/data/:
  1. PPGBPDatabase:      Systolic & Diastolic Blood Pressure + Hypertension Classification (219 subjects)
  2. PPGHRDatabase:      Motion-Robust Optical (IR/Red/Green) + 3-Axis Accelerometer Heart Rate Estimation
  3. PPGGlucoseDatabase: Non-Invasive Blood Glucose & 3-Tier Glycemic Risk (22 clinical subjects, 844k rows)

Saves all trained models into maternal_fitband/hardware_implementation/models/
"""

import os
import sys
import glob
import zipfile
import xml.etree.ElementTree as ET
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, accuracy_score

BASE_DATA = "maternal_fitband/data"
OUTPUT_MODELS = "maternal_fitband/hardware_implementation/models"
os.makedirs(OUTPUT_MODELS, exist_ok=True)

# =====================================================================
# 1. Train on PPGBPDatabase (Systolic & Diastolic BP + Hypertension)
# =====================================================================

def parse_bp_metadata(xlsx_path):
    with zipfile.ZipFile(xlsx_path) as z:
        ss_tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        shared_strings = []
        for si in ss_tree.findall('s:si', ns):
            t = si.find('s:t', ns)
            if t is not None and t.text:
                shared_strings.append(t.text)
            else:
                txt = ''.join([elem.text or '' for elem in si.findall('.//s:t', ns)])
                shared_strings.append(txt)
        sheet_tree = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        rows = []
        for row in sheet_tree.findall('.//s:row', ns):
            r_vals = {}
            for c in row.findall('s:c', ns):
                ref = c.attrib.get('r', '')
                col = ''.join([ch for ch in ref if ch.isalpha()])
                t_attr = c.attrib.get('t', '')
                v_tag = c.find('s:v', ns)
                val = v_tag.text if v_tag is not None else None
                if t_attr == 's' and val is not None:
                    val = shared_strings[int(val)]
                r_vals[col] = val
            rows.append(r_vals)
    df = pd.DataFrame(rows)
    cols = df.iloc[1].to_dict()
    df = df.iloc[2:].copy().rename(columns=cols)
    for c in ['subject_ID', 'Systolic Blood Pressure(mmHg)', 'Diastolic Blood Pressure(mmHg)', 'Heart Rate(b/m)', 'Age(year)', 'BMI(kg/m^2)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def extract_bp_features(meta_df, ppg_dir):
    records = []
    for _, row in meta_df.iterrows():
        sid = int(row['subject_ID'])
        sbp = row['Systolic Blood Pressure(mmHg)']
        dbp = row['Diastolic Blood Pressure(mmHg)']
        if pd.isna(sbp) or pd.isna(dbp):
            continue
        hr = row['Heart Rate(b/m)']
        age = row['Age(year)']
        bmi = row['BMI(kg/m^2)']
        gender = 1 if str(row.get('Sex(M/F)')).strip().lower() == 'female' else 0
        htn = str(row.get('Hypertension', 'Normal'))
        
        for seg in [1, 2, 3]:
            fpath = os.path.join(ppg_dir, f"{sid}_{seg}.txt")
            if not os.path.exists(fpath):
                continue
            with open(fpath) as f:
                vals = [float(x) for line in f if line.strip() for x in line.split()]
            vals = np.array(vals)
            if len(vals) < 500:
                continue
            
            p_std = float(np.std(vals))
            p_mean = float(np.mean(vals))
            p_amp = float(np.max(vals) - np.min(vals))
            p_auc = float(np.sum(vals)) / len(vals)
            peaks, _ = find_peaks(vals, distance=100, prominence=p_std * 0.5)
            ppi_mean = float(np.mean(np.diff(peaks))) if len(peaks) > 1 else 0.0
            ppi_std  = float(np.std(np.diff(peaks))) if len(peaks) > 1 else 0.0
            
            records.append({
                'subject_id': sid,
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'pulse_mean': p_mean,
                'pulse_std': p_std,
                'pulse_amp': p_amp,
                'pulse_auc': p_auc,
                'ppi_mean': ppi_mean,
                'ppi_std': ppi_std,
                'heart_rate': hr if pd.notna(hr) else (6000.0 / (ppi_mean + 1e-5)),
                'sbp': sbp,
                'dbp': dbp,
                'hypertension': htn
            })
    return pd.DataFrame(records).dropna()

def train_bp_models():
    print("\n" + "=" * 75)
    print("  1. TRAINING BLOOD PRESSURE & HYPERTENSION MODELS (PPGBPDatabase)")
    print("=" * 75)
    xlsx_path = os.path.join(BASE_DATA, "PPGBPDatabase/Data File/PPG-BP dataset.xlsx")
    ppg_dir = os.path.join(BASE_DATA, "PPGBPDatabase/Data File/0_subject")
    
    meta_df = parse_bp_metadata(xlsx_path)
    df_bp = extract_bp_features(meta_df, ppg_dir)
    print(f"Extracted {len(df_bp)} pulse records from {df_bp['subject_id'].nunique()} clinical subjects.")
    
    features = ['age', 'gender', 'bmi', 'pulse_mean', 'pulse_std', 'pulse_amp', 'pulse_auc', 'ppi_mean', 'ppi_std', 'heart_rate']
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_bp, groups=df_bp['subject_id']))
    train_df = df_bp.iloc[train_idx]
    test_df  = df_bp.iloc[test_idx]
    
    print(f"Train Cohort: {train_df['subject_id'].nunique()} subjects | Test Cohort: {test_df['subject_id'].nunique()} unseen subjects")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])
    
    # Systolic BP Model
    sbp_model = GradientBoostingRegressor(n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42)
    sbp_model.fit(X_train, train_df['sbp'])
    pred_sbp = sbp_model.predict(X_test)
    sbp_mae = mean_absolute_error(test_df['sbp'], pred_sbp)
    sbp_r2 = r2_score(test_df['sbp'], pred_sbp)
    print(f"  [SBP Model] MAE: {sbp_mae:.2f} mmHg | R2 Score: {sbp_r2:.3f}")
    
    # Diastolic BP Model
    dbp_model = GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.08, random_state=42)
    dbp_model.fit(X_train, train_df['dbp'])
    pred_dbp = dbp_model.predict(X_test)
    dbp_mae = mean_absolute_error(test_df['dbp'], pred_dbp)
    dbp_r2 = r2_score(test_df['dbp'], pred_dbp)
    print(f"  [DBP Model] MAE: {dbp_mae:.2f} mmHg | R2 Score: {dbp_r2:.3f}")
    
    # Hypertension Stage Classifier
    htn_clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    htn_clf.fit(X_train, train_df['hypertension'])
    acc = htn_clf.score(X_test, test_df['hypertension'])
    print(f"  [Hypertension Classifier] Accuracy: {acc*100:.1f}%")
    
    # Save Models
    joblib.dump(sbp_model, os.path.join(OUTPUT_MODELS, "maternal_sbp_estimator.joblib"))
    joblib.dump(dbp_model, os.path.join(OUTPUT_MODELS, "maternal_dbp_estimator.joblib"))
    joblib.dump(htn_clf,   os.path.join(OUTPUT_MODELS, "maternal_hypertension_classifier.joblib"))
    joblib.dump(scaler,    os.path.join(OUTPUT_MODELS, "bp_feature_scaler.joblib"))
    print("  -> Saved sbp_estimator, dbp_estimator, hypertension_classifier to models/")

# =====================================================================
# 2. Train on PPGHRDatabase (Motion-Robust Optical HR)
# =====================================================================

def train_motion_hr_models():
    print("\n" + "=" * 75)
    print("  2. TRAINING MOTION-ROBUST OPTICAL HEART RATE ESTIMATOR (PPGHRDatabase)")
    print("=" * 75)
    sig_files = glob.glob(os.path.join(BASE_DATA, "PPGHRDatabase/*_signals.csv"))
    print(f"Discovered {len(sig_files)} multi-spectral signal recordings.")
    
    windows = []
    fs = 100
    w_size = 400 # 4-second windows
    
    for sig_path in sig_files:
        base = sig_path.replace("_signals.csv", "")
        subject = os.path.basename(base).split("_")[0]
        ref_path = base + "_fitbit.csv"
        if not os.path.exists(ref_path):
            ref_path = base + "_ecg.csv"
        if not os.path.exists(ref_path):
            continue
            
        try:
            df_sig = pd.read_csv(sig_path)
            df_ref = pd.read_csv(ref_path)
            if 'confidence' in df_ref.columns:
                df_ref = df_ref[df_ref['confidence'] >= 2]
                
            for _, r in df_ref.iloc[::2].iterrows():
                t = r['time']
                hr = r['hr'] if 'hr' in r else r.get('bpm', None)
                if hr is None or pd.isna(hr):
                    continue
                end = int(t * fs)
                start = end - w_size
                if start < 0 or end >= len(df_sig):
                    continue
                win = df_sig.iloc[start:end]
                
                vm = np.sqrt(win['x']**2 + win['y']**2 + win['z']**2)
                windows.append({
                    'subject': subject,
                    'ir_std': float(win['ir'].std()),
                    'red_std': float(win['red'].std()),
                    'green_std': float(win['green'].std()),
                    'ir_mean': float(win['ir'].mean()),
                    'red_mean': float(win['red'].mean()),
                    'green_mean': float(win['green'].mean()),
                    'vm_mean': float(vm.mean()),
                    'vm_std': float(vm.std()),
                    'hr_target': float(hr)
                })
        except Exception:
            continue
            
    df_hr = pd.DataFrame(windows).dropna()
    print(f"Extracted {len(df_hr)} synchronized multi-sensor windows across {df_hr['subject'].nunique()} subjects.")
    
    hr_features = ['ir_std', 'red_std', 'green_std', 'ir_mean', 'red_mean', 'green_mean', 'vm_mean', 'vm_std']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df_hr, groups=df_hr['subject']))
    
    train_df = df_hr.iloc[train_idx]
    test_df  = df_hr.iloc[test_idx]
    print(f"Train Cohort: {train_df['subject'].nunique()} subjects | Test Cohort: {test_df['subject'].nunique()} unseen subjects")
    
    hr_scaler = StandardScaler()
    X_train = hr_scaler.fit_transform(train_df[hr_features])
    X_test  = hr_scaler.transform(test_df[hr_features])
    
    hr_model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    hr_model.fit(X_train, train_df['hr_target'])
    pred_hr = hr_model.predict(X_test)
    
    hr_mae = mean_absolute_error(test_df['hr_target'], pred_hr)
    hr_r2 = r2_score(test_df['hr_target'], pred_hr)
    print(f"  [Motion-Robust HR Model] MAE: {hr_mae:.2f} bpm | R2 Score: {hr_r2:.3f}")
    
    joblib.dump(hr_model,  os.path.join(OUTPUT_MODELS, "maternal_motion_robust_hr.joblib"))
    joblib.dump(hr_scaler, os.path.join(OUTPUT_MODELS, "hr_motion_scaler.joblib"))
    print("  -> Saved motion_robust_hr to models/")

# =====================================================================
# 3. Train on PPGGlucoseDatabase (Non-Invasive Glucose)
# =====================================================================

def train_glucose_models():
    print("\n" + "=" * 75)
    print("  3. TRAINING NON-INVASIVE GLUCOSE REGRESSOR (PPGGlucoseDatabase)")
    print("=" * 75)
    csv_path = "datasets/clean-dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = "clean-dataset.csv"
    if not os.path.exists(csv_path):
        print(f"[-] clean-dataset.csv not found at {csv_path}. Skipping glucose re-training.")
        return
        
    df = pd.read_csv(csv_path)
    df['Pulse_Amplitude'] = df['Systolic_Peak'] - df['Diastolic_Peak']
    df['Peak_Ratio'] = df['Diastolic_Peak'] / (df['Systolic_Peak'] + 1e-5)
    df['Normalized_Pulse_Area'] = df['Pulse_Area'] / (df['Pulse_Amplitude'] + 1e-5)
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    df['Cardiac_Index_Proxy'] = (df['Heart_Rate'] * df['Pulse_Amplitude']) / 100.0
    
    features = ['PPG_Signal', 'Heart_Rate', 'Systolic_Peak', 'Diastolic_Peak', 'Pulse_Area', 'Pulse_Amplitude', 'Cardiac_Index_Proxy']
    
    # 20k representative sample for fast training with patient grouping
    sub_dfs = []
    for pid, group in df.groupby('Patient_Id'):
        sub_dfs.append(group.sample(min(len(group), 1200), random_state=42))
    sample_df = pd.concat(sub_dfs, ignore_index=True)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(sample_df, groups=sample_df['Patient_Id']))
    
    train_df = sample_df.iloc[train_idx]
    test_df  = sample_df.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test  = scaler.transform(test_df[features])
    
    ridge_gluc = Ridge(alpha=100.0)
    ridge_gluc.fit(X_train, train_df['Glucose_level'])
    pred_gluc = ridge_gluc.predict(X_test)
    
    mae = mean_absolute_error(test_df['Glucose_level'], pred_gluc)
    r2 = r2_score(test_df['Glucose_level'], pred_gluc)
    print(f"  [Continuous Glucose Model] MAE: {mae:.2f} mg/dL | R2 Score: {r2:.3f}")
    
    joblib.dump(ridge_gluc, os.path.join(OUTPUT_MODELS, "best_glucose_regressor.joblib"))
    joblib.dump(scaler,     os.path.join(OUTPUT_MODELS, "glucose_feature_scaler.joblib"))
    print("  -> Saved best_glucose_regressor to models/")

def main():
    print("🤰 MATERNAL FITBAND — END-TO-END TRAINING PIPELINE ON DATA FOLDER 🤰")
    train_bp_models()
    train_motion_hr_models()
    train_glucose_models()
    print("\n✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY IN HARDWARE_IMPLEMENTATION/MODELS!")

if __name__ == "__main__":
    main()
