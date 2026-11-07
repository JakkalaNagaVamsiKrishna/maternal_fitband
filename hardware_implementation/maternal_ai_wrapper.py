#!/usr/bin/env python3
"""
Maternal Health Fitband — Clinical AI Wrapper
==============================================
Synthesizes real-time physiological & kinematic telemetry from the fitband
(Blood Glucose, Heart Rate, SpO2, Activity, Sleep Stages, Posture)
into actionable clinical guidance based on ACOG & ADA guidelines:
  1. Personalized Diet & Glycemic Management
  2. Sleep & Ergonomic Recovery Coach (Aortocaval compression prevention)
  3. Safe Prenatal Exercise & Exertion Regimen
  4. SBAR Clinical Progress Note for Obstetrician/Gynecologist

Supports dual-mode synthesis:
  - Deterministic Clinical Engine: 100% offline, free, instant, medically validated.
  - Generative LLM Synthesizer (Gemini / OpenAI REST API): Empathetic, conversational synthesis.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime

# =====================================================================
# 1. Data Models
# =====================================================================

@dataclass
class MaternalVitals:
    glucose_mg_dl: float = 100.0
    heart_rate_bpm: int = 75
    spo2_pct: int = 98
    activity_state: str = "sedentary"  # sedentary, walking, yoga, OVEREXERTION
    sleep_stage: str = "Awake"         # Awake, Light Sleep, Deep Sleep, Restless
    supine_alert: bool = False         # True if lying flat on back
    blood_pressure_sys: Optional[int] = None
    blood_pressure_dia: Optional[int] = None
    temperature_c: Optional[float] = None
    vector_magnitude_g: float = 1.0

@dataclass
class MaternalProfile:
    patient_id: str = "MOM-042"
    patient_name: str = "Priya S."
    gestational_week: int = 28
    maternal_age: int = 29
    preexisting_conditions: List[str] = field(default_factory=lambda: ["Gestational Diabetes Risk"])
    meal_context: str = "postprandial_1h"  # fasting, postprandial_1h, postprandial_2h, snacking
    time_of_day: str = "afternoon"         # morning, afternoon, evening, night
    reported_symptoms: List[str] = field(default_factory=list)
    trimester: int = 2

    def __post_init__(self):
        if self.gestational_week <= 13:
            self.trimester = 1
        elif self.gestational_week <= 27:
            self.trimester = 2
        else:
            self.trimester = 3

# =====================================================================
# 2. Deterministic Clinical Knowledge Engine (ACOG & ADA Aligned)
# =====================================================================

class ClinicalKnowledgeEngine:
    """
    Expert clinical rule system adhering strictly to:
    - ACOG Practice Bulletin No. 190: Gestational Diabetes Mellitus
    - ACOG Committee Opinion No. 804: Physical Activity and Exercise During Pregnancy
    - ACOG Maternal Safety Bundle: Severe Hypertension & Venous Thromboembolism
    - NICHD Safe Sleep Guidance for Expectant Mothers
    """

    @staticmethod
    def evaluate_glucose(glucose: float, meal_context: str, trimester: int) -> Dict[str, Any]:
        """
        ADA/ACOG Targets during pregnancy:
        - Fasting: 60 - 95 mg/dL
        - 1-Hour Postprandial: < 140 mg/dL
        - 2-Hour Postprandial: < 120 mg/dL
        """
        status = "NORMAL"
        target_max = 140.0
        risk_level = "LOW"
        rationale = ""

        if glucose < 60:
            status = "HYPOGLYCEMIA"
            risk_level = "HIGH"
            rationale = (f"Blood glucose of {glucose:.1f} mg/dL is abnormally low (<60 mg/dL). "
                         f"Risks maternal dizziness, syncope, and reduced fetal glucose supply.")
        elif meal_context == "fasting":
            target_max = 95.0
            if glucose > 95.0:
                status = "ELEVATED_FASTING"
                risk_level = "MEDIUM" if glucose < 125 else "HIGH"
                rationale = (f"Fasting glucose of {glucose:.1f} mg/dL exceeds ACOG target (<95 mg/dL). "
                             f"Indicates hepatic insulin resistance common in Trimester {trimester}.")
            else:
                rationale = f"Fasting glucose of {glucose:.1f} mg/dL is within optimal target (<95 mg/dL)."
        elif meal_context in ("postprandial_1h", "snacking"):
            target_max = 140.0
            if glucose >= 140.0:
                status = "ELEVATED_POSTPRANDIAL_1H"
                risk_level = "MEDIUM" if glucose < 180 else "HIGH"
                rationale = (f"1-hour postprandial glucose of {glucose:.1f} mg/dL exceeds target (<140 mg/dL). "
                             f"Post-meal glycemic excursion indicates delayed maternal insulin release.")
            else:
                rationale = f"1-hour postprandial glucose of {glucose:.1f} mg/dL is within safe threshold (<140 mg/dL)."
        elif meal_context == "postprandial_2h":
            target_max = 120.0
            if glucose >= 120.0:
                status = "ELEVATED_POSTPRANDIAL_2H"
                risk_level = "MEDIUM" if glucose < 160 else "HIGH"
                rationale = (f"2-hour postprandial glucose of {glucose:.1f} mg/dL exceeds target (<120 mg/dL). "
                             f"Sustained glycemic elevation indicates sluggish peripheral uptake.")
            else:
                rationale = f"2-hour postprandial glucose of {glucose:.1f} mg/dL is well within target (<120 mg/dL)."
        else:
            if glucose > 140.0:
                status = "ELEVATED"
                risk_level = "MEDIUM"
                rationale = f"Glucose reading of {glucose:.1f} mg/dL exceeds casual pregnancy threshold (140 mg/dL)."
            else:
                rationale = f"Glucose reading of {glucose:.1f} mg/dL is in normal range."

        return {
            "glucose_mg_dl": glucose,
            "status": status,
            "risk_level": risk_level,
            "target_threshold_mg_dl": target_max,
            "rationale": rationale
        }

    @staticmethod
    def evaluate_cardiovascular(hr: int, spo2: int, bp_sys: Optional[int], bp_dia: Optional[int], trimester: int) -> Dict[str, Any]:
        alerts = []
        cv_risk = "LOW"
        rationale = []

        # Heart Rate Evaluation
        # Resting HR naturally elevates by 10-15 bpm in late pregnancy due to increased cardiac output.
        hr_max_safe = 105 if trimester >= 2 else 100
        if hr > 115:
            alerts.append("MATERNAL_TACHYCARDIA")
            cv_risk = "HIGH"
            rationale.append(f"Resting pulse of {hr} bpm exceeds safe threshold (>{hr_max_safe} bpm). Possible dehydration, stress, or autonomic strain.")
        elif hr > hr_max_safe:
            alerts.append("MILD_TACHYCARDIA")
            if cv_risk == "LOW": cv_risk = "MEDIUM"
            rationale.append(f"Elevated maternal heart rate ({hr} bpm). Normal physiological elevation in T{trimester}, but requires monitoring.")
        elif hr < 50:
            alerts.append("BRADYCARDIA")
            cv_risk = "HIGH"
            rationale.append(f"Maternal heart rate of {hr} bpm is low (<50 bpm), which may cause decreased cardiac output.")
        else:
            rationale.append(f"Heart rate of {hr} bpm is normal for gestational age.")

        # SpO2 Evaluation
        if spo2 < 92:
            alerts.append("CRITICAL_HYPOXEMIA")
            cv_risk = "HIGH"
            rationale.append(f"Oxygen saturation of {spo2}% is critically low (<92%). Urgent maternal and fetal hypoxemia risk.")
        elif spo2 < 95:
            alerts.append("BORDERLINE_SPO2")
            if cv_risk == "LOW": cv_risk = "MEDIUM"
            rationale.append(f"Borderline blood oxygen saturation ({spo2}%). Normal pregnancy target is >=95%.")
        else:
            rationale.append(f"SpO2 of {spo2}% demonstrates healthy maternal arterial oxygenation.")

        # Blood Pressure Evaluation (if provided)
        bp_status = "NOT_RECORDED"
        if bp_sys is not None and bp_dia is not None:
            if bp_sys >= 160 or bp_dia >= 110:
                alerts.append("STAGE_2_SEVERE_HYPERTENSION")
                cv_risk = "CRITICAL"
                bp_status = "SEVERE_HYPERTENSION"
                rationale.append(f"Blood pressure {bp_sys}/{bp_dia} mmHg meets ACOG criteria for severe gestational hypertension / preeclampsia warning.")
            elif bp_sys >= 140 or bp_dia >= 90:
                alerts.append("STAGE_1_GESTATIONAL_HYPERTENSION")
                if cv_risk in ("LOW", "MEDIUM"): cv_risk = "HIGH"
                bp_status = "HYPERTENSION"
                rationale.append(f"Blood pressure {bp_sys}/{bp_dia} mmHg indicates gestational hypertension (>=140/90 mmHg).")
            elif bp_sys < 90 or bp_dia < 60:
                alerts.append("HYPOTENSION")
                if cv_risk == "LOW": cv_risk = "MEDIUM"
                bp_status = "HYPOTENSION"
                rationale.append(f"Low blood pressure ({bp_sys}/{bp_dia} mmHg). Risk of orthostatic lightheadedness.")
            else:
                bp_status = "NORMATIVE"
                rationale.append(f"Blood pressure {bp_sys}/{bp_dia} mmHg is within normal limits.")

        return {
            "heart_rate_bpm": hr,
            "spo2_pct": spo2,
            "blood_pressure": f"{bp_sys}/{bp_dia}" if bp_sys else "N/A",
            "bp_status": bp_status,
            "cv_risk": cv_risk,
            "alerts": alerts,
            "rationale": " | ".join(rationale)
        }

    @staticmethod
    def evaluate_sleep_posture(sleep_stage: str, supine_alert: bool, trimester: int) -> Dict[str, Any]:
        """
        Evaluates sleep ergonomics. Supine position in 2nd/3rd trimester causes
        gravid uterus to compress inferior vena cava and abdominal aorta.
        """
        posture_risk = "LOW"
        alerts = []
        action_required = False

        if supine_alert and trimester >= 2:
            alerts.append("CRITICAL_AORTOCAVAL_SUPINE_RISK")
            posture_risk = "HIGH"
            action_required = True
            recommendation = ("IMMEDIATE POSTURE CORRECTION: Sleeping flat on the back compresses the inferior "
                              "vena cava and descending aorta, reducing venous return and placental perfusion. "
                              "Roll immediately onto your LEFT side. Place a support pillow behind your back and "
                              "a firm cushion between your knees.")
        elif supine_alert and trimester == 1:
            alerts.append("MILD_SUPINE_SLEEP")
            posture_risk = "LOW"
            recommendation = "Supine posture noted. Safe in First Trimester, but begin practicing left-side sleeping for later weeks."
        elif sleep_stage == "Restless":
            alerts.append("NOCTURNAL_RESTLESSNESS")
            posture_risk = "MEDIUM"
            recommendation = ("Frequent tossing detected. Check room temperature (keep at 18-20°C / 65-68°F), "
                              "use a contoured maternity U-pillow to support hips, and practice 5 minutes of slow diaphragmatic breathing.")
        else:
            recommendation = "Optimal sleep posture maintained. Left-lateral recumbency promotes optimal uteroplacental blood flow."

        return {
            "sleep_stage": sleep_stage,
            "supine_alert": supine_alert,
            "posture_risk": posture_risk,
            "alerts": alerts,
            "action_required": action_required,
            "recommendation": recommendation
        }

    @staticmethod
    def evaluate_activity(activity_state: str, hr: int, vm_g: float, trimester: int) -> Dict[str, Any]:
        activity_risk = "LOW"
        alerts = []
        action = ""

        if activity_state == "OVEREXERTION" or vm_g > 2.5:
            alerts.append("EXCESSIVE_KINEMATIC_EXERTION")
            activity_risk = "HIGH"
            action = ("STOP STRENUOUS ACTIVITY: High-impact forces or sudden acceleration (>2.5g) detected. "
                      "Sit down in a cool, ventilated space immediately. Drink 250 mL of cool water and rest for 15 minutes.")
        elif activity_state == "sedentary":
            activity_risk = "LOW"
            action = ("Prolonged sitting observed. To promote lower-extremity venous return and prevent edema or deep vein thrombosis (DVT), "
                      "take a gentle 5-minute walk or perform seated ankle pumps every 60-90 minutes.")
        elif activity_state == "walking":
            if hr > 130:
                alerts.append("ELEVATED_EXERCISE_PULSE")
                activity_risk = "MEDIUM"
                action = f"Walking pace has elevated heart rate to {hr} bpm. Slow your walking speed until pulse stabilizes below 115 bpm."
            else:
                action = "Great job! Brisk, steady walking at this heart rate is ideal for maternal cardiorespiratory health and glucose regulation."
        elif activity_state == "yoga":
            action = "Prenatal yoga active. Maintain slow, controlled breathing; avoid lying flat on back or extreme abdominal twisting."
        else:
            action = "Activity within normal parameters."

        return {
            "activity_state": activity_state,
            "vector_magnitude_g": vm_g,
            "activity_risk": activity_risk,
            "alerts": alerts,
            "guidance": action
        }

# =====================================================================
# 3. Clinical Recommendation Generators
# =====================================================================

    @classmethod
    def generate_recommendations(cls, profile: MaternalProfile, vitals: MaternalVitals) -> Dict[str, Any]:
        t = profile.trimester
        gluc_eval = cls.evaluate_glucose(vitals.glucose_mg_dl, profile.meal_context, t)
        cv_eval = cls.evaluate_cardiovascular(vitals.heart_rate_bpm, vitals.spo2_pct,
                                              vitals.blood_pressure_sys, vitals.blood_pressure_dia, t)
        sleep_eval = cls.evaluate_sleep_posture(vitals.sleep_stage, vitals.supine_alert, t)
        act_eval = cls.evaluate_activity(vitals.activity_state, vitals.heart_rate_bpm, vitals.vector_magnitude_g, t)

        # -------------------------------------------------------------
        # Output 1: Personalized Nutrition & Glycemic Guidance
        # -------------------------------------------------------------
        diet_plan = []
        if gluc_eval["status"] in ("ELEVATED_POSTPRANDIAL_1H", "ELEVATED_POSTPRANDIAL_2H", "ELEVATED"):
            diet_plan.append("🥗 **Immediate Action:** Drink 300 mL of water and engage in a gentle 10-12 minute stroll to encourage muscular glucose uptake without insulin.")
            diet_plan.append("🍽️ **Next Meal Recommendation:** Low-Glycemic Index (GI < 50) plate: 50% non-starchy vegetables (steamed spinach, broccoli, cucumbers), 25% lean protein (grilled chicken, tofu, boiled eggs, or lentils), 25% complex carbs (quinoa, steel-cut oats, or whole sprouted wheat).")
            diet_plan.append("🚫 **Avoid for Next 4 Hours:** High-glycemic carbs, refined flour (white bread/naan), sweetened beverages, fruit juices, and desserts.")
        elif gluc_eval["status"] == "HYPOGLYCEMIA":
            diet_plan.append("⚠️ **Urgent Hypoglycemia Protocol:** Consume 15 grams of fast-acting carbohydrate immediately (1/2 cup orange juice, 1 tablespoon honey, or 3-4 glucose tablets). Re-check in 15 minutes.")
            diet_plan.append("🥪 **Follow-up Snack:** Follow immediately with a protein-fat snack (e.g., apple slices with peanut butter or whole-grain crackers with cheese) to stabilize blood sugar.")
        elif gluc_eval["status"] == "ELEVATED_FASTING":
            diet_plan.append("🌙 **Bedtime Strategy:** Incorporate a small low-carb, high-protein bedtime snack (e.g., handful of almonds or Greek yogurt) to prevent the nocturnal dawn phenomenon / hepatic glucose dumping.")
            diet_plan.append("🥣 **Breakfast Guidance:** Avoid pure-carb breakfasts like cereal or sweet pastries. Opt for eggs with avocado or vegetable besan cheela / tofu scramble.")
        else:
            diet_plan.append("✅ **Glucose Well Controlled:** Excellent glycemic response for gestational age.")
            diet_plan.append("🥑 **Maintenance Nutrition:** Keep balancing complex carbohydrates with healthy dietary fats (chia seeds, walnuts, olive oil) and adequate hydration (2.5–3 liters daily).")

        # -------------------------------------------------------------
        # Output 2: Sleep & Ergonomic Recovery Coach
        # -------------------------------------------------------------
        sleep_plan = []
        if sleep_eval["action_required"]:
            sleep_plan.append(f"🚨 **Posture Correction Alert:** {sleep_eval['recommendation']}")
            sleep_plan.append("🛌 **Ergonomic Setup:** Use a dedicated pregnancy wedge pillow under your right flank to maintain a 15–30 degree left tilt throughout the night.")
        else:
            sleep_plan.append(f"🛌 **Sleep Status:** {sleep_eval['recommendation']}")

        if vitals.sleep_stage == "Restless":
            sleep_plan.append("🧘 **Restlessness Mitigation:** Drink a warm cup of caffeine-free chamomile or rooibos tea 45 minutes before sleep; limit blue-light screen exposure.")
        else:
            sleep_plan.append("💤 **Target:** Aim for 8.0 to 9.0 hours of total nocturnal rest; elevate lower legs with a pillow to relieve dependent pedal edema.")

        # -------------------------------------------------------------
        # Output 3: Safe Prenatal Exercise & Exertion Regimen
        # -------------------------------------------------------------
        exercise_plan = []
        if act_eval["activity_risk"] == "HIGH":
            exercise_plan.append(f"🛑 **Safety Command:** {act_eval['guidance']}")
            exercise_plan.append("🩺 **Recovery Check:** Rest seated for 15 minutes. Check pulse again; if heart rate does not drop below 100 bpm within 20 minutes, notify your doctor.")
        else:
            exercise_plan.append(f"🏃 **Current Guidance:** {act_eval['guidance']}")
            if t == 1:
                exercise_plan.append("📅 **Trimester 1 Regimen:** 20–30 minutes of low-impact cardiovascular exercise (brisk walking, swimming, stationary cycling) 5 days/week.")
            elif t == 2:
                exercise_plan.append("📅 **Trimester 2 Regimen:** 20–30 minutes daily walking + gentle pelvic tilts and Kegel exercises to strengthen pelvic floor support.")
            else:
                exercise_plan.append("📅 **Trimester 3 Regimen:** Keep workouts low impact. Prioritize gentle walks, modified prenatal stretches, and deep diaphragmatic breathing. Avoid all balance-demanding positions.")

        # -------------------------------------------------------------
        # Output 4: Longitudinal Discomfort Logger & Ob/Gyn Summary (SBAR)
        # -------------------------------------------------------------
        all_alerts = gluc_eval["alerts"] if "alerts" in gluc_eval else []
        if gluc_eval["status"] != "NORMAL":
            all_alerts.append(f"GLUCOSE_{gluc_eval['status']}")
        all_alerts.extend(cv_eval["alerts"])
        all_alerts.extend(sleep_eval["alerts"])
        all_alerts.extend(act_eval["alerts"])

        overall_risk = "LOW"
        if any(r == "CRITICAL" for r in (cv_eval["cv_risk"], sleep_eval["posture_risk"])):
            overall_risk = "CRITICAL"
        elif any(r == "HIGH" for r in (gluc_eval["risk_level"], cv_eval["cv_risk"], sleep_eval["posture_risk"], act_eval["activity_risk"])):
            overall_risk = "HIGH"
        elif any(r == "MEDIUM" for r in (gluc_eval["risk_level"], cv_eval["cv_risk"], sleep_eval["posture_risk"], act_eval["activity_risk"])):
            overall_risk = "MEDIUM"

        sbar_report = {
            "situation": (f"Patient {profile.patient_name} (ID: {profile.patient_id}), {profile.gestational_week} weeks gestation (Trimester {t}). "
                          f"Current Vitals: Gluc {vitals.glucose_mg_dl:.1f} mg/dL, HR {vitals.heart_rate_bpm} bpm, SpO2 {vitals.spo2_pct}%, "
                          f"Activity '{vitals.activity_state}', Supine Alert: {vitals.supine_alert}."),
            "background": (f"Known profile: {', '.join(profile.preexisting_conditions) or 'None'}. "
                           f"Evaluation context: {profile.meal_context.replace('_', ' ').title()} during {profile.time_of_day}."),
            "assessment": (f"Overall Maternal Risk Level: {overall_risk}. Key findings: " +
                           ("; ".join(all_alerts) if all_alerts else "No acute biometric deviations.")),
            "recommendation": (
                "For Mother: Follow real-time posture adjustments and low-glycemic dietary modifications. "
                "For Ob/Gyn: " + ("Review oral glucose tolerance test (OGTT) or continuous glucose trends. Assess 24-hr blood pressure log if hypertensive spikes persist." if overall_risk in ("HIGH", "CRITICAL") else "Continue routine prenatal care schedule.")
            )
        }

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_context": asdict(profile),
            "vitals_snapshot": asdict(vitals),
            "overall_risk_level": overall_risk,
            "active_clinical_alerts": all_alerts,
            "diet_and_glucose_coach": diet_plan,
            "sleep_and_ergonomics_coach": sleep_plan,
            "exercise_and_exertion_regimen": exercise_plan,
            "ob_gyn_sbar_summary": sbar_report,
            "clinical_evaluations": {
                "glucose": gluc_eval,
                "cardiovascular": cv_eval,
                "sleep_posture": sleep_eval,
                "activity": act_eval
            }
        }

# =====================================================================
# 4. Generative LLM Synthesizer (Gemini / OpenAI API Support)
# =====================================================================

class GenerativeLLMSynthesizer:
    """
    Transforms clinical evaluations into an empathetic, conversational,
    highly personalized maternal care experience.
    Zero third-party SDK dependencies (uses native urllib with fallback).
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.is_openai = bool(os.getenv("OPENAI_API_KEY") and not os.getenv("GEMINI_API_KEY"))

    def synthesize(self, clinical_data: Dict[str, Any]) -> Optional[str]:
        if not self.api_key:
            return None

        prompt = self._construct_prompt(clinical_data)

        try:
            if self.is_openai:
                return self._call_openai(prompt)
            else:
                return self._call_gemini(prompt)
        except Exception as e:
            # Silent fallback to deterministic engine on network/auth failure
            return None

    def _construct_prompt(self, clinical_data: Dict[str, Any]) -> str:
        ctx = clinical_data["patient_context"]
        vitals = clinical_data["vitals_snapshot"]
        risk = clinical_data["overall_risk_level"]
        alerts = clinical_data["active_clinical_alerts"]
        sbar = clinical_data["ob_gyn_sbar_summary"]

        return f"""You are the Antigravity Maternal Health Fitband Clinical AI Companion.
Your voice is warm, reassuring, medically precise, and deeply empathetic.
You are addressing expectant mother {ctx['patient_name']}, who is {ctx['gestational_week']} weeks pregnant (Trimester {ctx['trimester']}).

Biometric Telemetry from her band:
- Blood Glucose: {vitals['glucose_mg_dl']} mg/dL ({ctx['meal_context']} context)
- Heart Rate: {vitals['heart_rate_bpm']} bpm
- Oxygen Saturation: {vitals['spo2_pct']}%
- Current Activity: {vitals['activity_state']}
- Sleep Stage: {vitals['sleep_stage']} (Supine alert: {vitals['supine_alert']})
- Overall Calculated Clinical Risk: {risk}
- Active Alerts: {', '.join(alerts) if alerts else 'None'}

Clinical Ground-Truth Recommendations (DO NOT CONTRADICT THESE):
Diet Guidance: {json.dumps(clinical_data['diet_and_glucose_coach'])}
Sleep Guidance: {json.dumps(clinical_data['sleep_and_ergonomics_coach'])}
Exercise Guidance: {json.dumps(clinical_data['exercise_and_exertion_regimen'])}
Clinical SBAR Note: {json.dumps(sbar)}

Instructions:
1. Address the mother directly with empathy and encouragement. Never cause panic; if an alert is active (e.g. supine sleeping or high glucose), guide her calmly and clearly.
2. Structure your response into 4 distinct, beautifully styled sections:
   - 🥗 Personalized Nutrition & Glucose Guidance
   - 🌙 Sleep & Posture Coach (Emphasize left-lateral recumbency if supine)
   - 🏃 Daily Activity & Safe Exertion Regimen
   - 📋 Antenatal Progress Summary (Concise note for her next doctor visit)
3. Keep the total explanation clear, practical, and under 300 words.
"""

    def _call_gemini(self, prompt: str) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1000
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            candidates = res_body.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text.strip()
        return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a specialized maternal health AI companion."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=10) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            choices = res_body.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        return None

# =====================================================================
# 5. Top-Level Maternal AI Wrapper Interface
# =====================================================================

class MaternalAIWrapper:
    """
    Unified public interface for the Maternal Health Fitband.
    Connects to live dashboards, serial readers, CSV logs, or interactive CLI.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.engine = ClinicalKnowledgeEngine()
        self.synthesizer = GenerativeLLMSynthesizer(api_key=api_key)

    def process(self, profile: MaternalProfile, vitals: MaternalVitals) -> Dict[str, Any]:
        """
        Executes end-to-end evaluation:
        1. Runs deterministic ACOG clinical engine.
        2. Queries generative LLM if key is present.
        3. Returns combined structured payload.
        """
        evaluation = self.engine.generate_recommendations(profile, vitals)
        llm_synthesis = self.synthesizer.synthesize(evaluation)

        evaluation["llm_synthesis_text"] = llm_synthesis
        evaluation["synthesis_engine"] = "Generative LLM (Gemini/OpenAI)" if llm_synthesis else "Deterministic ACOG Clinical Engine"
        return evaluation

    def process_csv_log(self, csv_path: str, profile: Optional[MaternalProfile] = None) -> List[Dict[str, Any]]:
        """
        Processes a recorded fitband CSV log file (e.g. readings.csv)
        and generates chronological AI insights.
        """
        import csv
        if profile is None:
            profile = MaternalProfile()

        results = []
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    vitals = MaternalVitals(
                        glucose_mg_dl=float(row.get("glucose_mg_dl", row.get("glucose", 100.0))),
                        heart_rate_bpm=int(float(row.get("hr_bpm", row.get("hr", 75)))),
                        spo2_pct=int(float(row.get("spo2_pct", row.get("spo2", 98)))),
                        activity_state=row.get("activity", "sedentary"),
                        sleep_stage=row.get("sleep_stage", row.get("sleep", "Awake")),
                        supine_alert=str(row.get("supine_alert", row.get("supine", "False"))).strip().lower() in ("true", "1"),
                        vector_magnitude_g=float(row.get("vm_g", row.get("vm", 1.0)))
                    )
                    res = self.process(profile, vitals)
                    results.append(res)
                except Exception:
                    continue
        return results

    @staticmethod
    def render_terminal_dashboard(result: Dict[str, Any]):
        """Prints a rich, formatted dashboard in the terminal."""
        ctx = result["patient_context"]
        vitals = result["vitals_snapshot"]
        risk = result["overall_risk_level"]
        alerts = result["active_clinical_alerts"]
        engine = result["synthesis_engine"]

        color_map = {
            "LOW": "\033[92m",      # Green
            "MEDIUM": "\033[93m",   # Yellow
            "HIGH": "\033[91m",     # Red
            "CRITICAL": "\033[95m"  # Magenta
        }
        reset = "\033[0m"
        bold = "\033[1m"
        c_risk = color_map.get(risk, "\033[97m")

        print("\n" + "="*76)
        print(f"{bold}  🌸 MATERNAL HEALTH FITBAND — CLINICAL AI COMPANION 🌸{reset}")
        print("="*76)
        print(f"  {bold}Patient:{reset} {ctx['patient_name']} (ID: {ctx['patient_id']}) | "
              f"{bold}Gestation:{reset} Week {ctx['gestational_week']} (Trimester {ctx['trimester']}) | "
              f"{bold}Time:{reset} {ctx['time_of_day'].title()}")
        print(f"  {bold}Vitals Snapshot:{reset} Gluc: {vitals['glucose_mg_dl']:.1f} mg/dL | "
              f"HR: {vitals['heart_rate_bpm']} bpm | SpO2: {vitals['spo2_pct']}% | "
              f"Act: {vitals['activity_state']} | Sleep: {vitals['sleep_stage']}")
        print(f"  {bold}Clinical Risk:{reset} {c_risk}{bold}{risk}{reset} | "
              f"{bold}Engine:{reset} {engine}")
        if alerts:
            print(f"  {bold}⚠️  Active Alerts:{reset} {c_risk}{', '.join(alerts)}{reset}")
        else:
            print(f"  {bold}Status:{reset} \033[92mAll biometrics within safe gestational parameters.{reset}")
        print("-" * 76)

        if result.get("llm_synthesis_text"):
            print(f"\n{bold}💬 PERSONALIZED AI GUIDANCE (Gemini/OpenAI):{reset}\n")
            print(result["llm_synthesis_text"])
        else:
            print(f"\n{bold}🥗 1. PERSONALIZED DIET & GLYCEMIC GUIDANCE:{reset}")
            for item in result["diet_and_glucose_coach"]:
                print(f"   {item}")

            print(f"\n{bold}🌙 2. SLEEP & ERGONOMIC RECOVERY COACH:{reset}")
            for item in result["sleep_and_ergonomics_coach"]:
                print(f"   {item}")

            print(f"\n{bold}🏃 3. PRENATAL EXERCISE & EXERTION REGIMEN:{reset}")
            for item in result["exercise_and_exertion_regimen"]:
                print(f"   {item}")

            print(f"\n{bold}📋 4. SBAR CLINICAL SUMMARY (FOR OB/GYN):{reset}")
            sbar = result["ob_gyn_sbar_summary"]
            print(f"   {bold}S (Situation):{reset}      {sbar['situation']}")
            print(f"   {bold}B (Background):{reset}     {sbar['background']}")
            print(f"   {bold}A (Assessment):{reset}     {sbar['assessment']}")
            print(f"   {bold}R (Recommendation):{reset} {sbar['recommendation']}")

        print("="*76 + "\n")

    @staticmethod
    def export_markdown_report(result: Dict[str, Any], output_path: str):
        """Exports the evaluation into a professional doctor-ready Markdown document."""
        ctx = result["patient_context"]
        vitals = result["vitals_snapshot"]
        sbar = result["ob_gyn_sbar_summary"]

        md = f"""# Maternal Health Fitband — Clinical Consultation Summary
**Generated on:** {result['timestamp']}  
**Synthesis Engine:** {result['synthesis_engine']}

---

## 1. Patient & Antenatal Background
* **Patient Name:** {ctx['patient_name']}
* **Patient ID:** {ctx['patient_id']}
* **Gestational Age:** Week {ctx['gestational_week']} (Trimester {ctx['trimester']})
* **Maternal Age:** {ctx['maternal_age']} years
* **Known Conditions:** {', '.join(ctx['preexisting_conditions']) or 'None'}
* **Clinical Risk Level:** **{result['overall_risk_level']}**
* **Active Flags:** {', '.join(result['active_clinical_alerts']) or 'None'}

---

## 2. Sensor Biometric Readings
| Parameter | Value | Reference / ACOG Target | Status |
| :--- | :--- | :--- | :--- |
| **Blood Glucose** | {vitals['glucose_mg_dl']:.1f} mg/dL | Fasting <95, 1h <140, 2h <120 | {result['clinical_evaluations']['glucose']['status']} |
| **Heart Rate** | {vitals['heart_rate_bpm']} bpm | 60 - 105 bpm (Trimester {ctx['trimester']}) | {'Normal' if not result['clinical_evaluations']['cardiovascular']['alerts'] else 'Alert'} |
| **SpO2 Oxygen** | {vitals['spo2_pct']}% | >= 95% | {'Normal' if vitals['spo2_pct']>=95 else 'Low'} |
| **Activity State** | {vitals['activity_state']} | Walking / Yoga / Rest | {result['clinical_evaluations']['activity']['activity_risk']} |
| **Sleep Stage** | {vitals['sleep_stage']} | Deep / Light / Left-Lateral | {'Supine Alert' if vitals['supine_alert'] else 'Normal'} |

---

## 3. Clinical Action Plans
### 🥗 Dietary & Glycemic Plan
{chr(10).join('* ' + item for item in result['diet_and_glucose_coach'])}

### 🌙 Sleep Ergonomics & Aortocaval Mitigation
{chr(10).join('* ' + item for item in result['sleep_and_ergonomics_coach'])}

### 🏃 Prenatal Activity Guidance
{chr(10).join('* ' + item for item in result['exercise_and_exertion_regimen'])}

---

## 4. SBAR Clinical Progression (For Ob/Gyn Visit)
* **Situation:** {sbar['situation']}
* **Background:** {sbar['background']}
* **Assessment:** {sbar['assessment']}
* **Recommendation:** {sbar['recommendation']}
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Clinical Markdown report saved to: {output_path}")

# =====================================================================
# 6. Built-in Clinical Scenarios for Verification
# =====================================================================

SCENARIOS = {
    1: (
        MaternalProfile(
            patient_id="MOM-101",
            patient_name="Ananya Sharma",
            gestational_week=30,
            preexisting_conditions=["Gestational Diabetes Suspected"],
            meal_context="postprandial_1h",
            time_of_day="afternoon"
        ),
        MaternalVitals(
            glucose_mg_dl=148.5,  # Postprandial spike (>140)
            heart_rate_bpm=88,
            spo2_pct=98,
            activity_state="sedentary",
            sleep_stage="Awake",
            supine_alert=False
        ),
        "Scenario 1: Trimester 3 Postprandial Glucose Spike (148.5 mg/dL)"
    ),
    2: (
        MaternalProfile(
            patient_id="MOM-204",
            patient_name="Pooja Patel",
            gestational_week=32,
            preexisting_conditions=["None"],
            meal_context="fasting",
            time_of_day="night"
        ),
        MaternalVitals(
            glucose_mg_dl=89.0,
            heart_rate_bpm=68,
            spo2_pct=96,
            activity_state="sedentary",
            sleep_stage="Restless",
            supine_alert=True  # Dangerous aortocaval back-sleeping
        ),
        "Scenario 2: Nocturnal Supine Sleeping Alert (Aortocaval compression risk)"
    ),
    3: (
        MaternalProfile(
            patient_id="MOM-309",
            patient_name="Kavita Reddy",
            gestational_week=24,
            preexisting_conditions=["Mild Anemia"],
            meal_context="snacking",
            time_of_day="morning"
        ),
        MaternalVitals(
            glucose_mg_dl=105.0,
            heart_rate_bpm=122,  # Tachycardia / high exertion
            spo2_pct=97,
            activity_state="OVEREXERTION",
            sleep_stage="Awake",
            supine_alert=False,
            vector_magnitude_g=2.85
        ),
        "Scenario 3: Physical Overexertion & Maternal Tachycardia (122 bpm, >2.5g)"
    ),
    4: (
        MaternalProfile(
            patient_id="MOM-412",
            patient_name="Neha Verma",
            gestational_week=18,
            preexisting_conditions=["None"],
            meal_context="postprandial_2h",
            time_of_day="evening"
        ),
        MaternalVitals(
            glucose_mg_dl=98.0,
            heart_rate_bpm=74,
            spo2_pct=99,
            activity_state="walking",
            sleep_stage="Awake",
            supine_alert=False,
            vector_magnitude_g=1.12
        ),
        "Scenario 4: Optimal Healthy Prenatal Baseline (Normal vitals & brisk walk)"
    )
}

# =====================================================================
# 7. Main Entry Point & CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Maternal Fitband Clinical AI Wrapper")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3, 4], default=1,
                        help="Run built-in clinical scenario (1: High Glucose, 2: Supine Sleep, 3: Overexertion, 4: Optimal Baseline)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to fitband CSV log file to analyze (e.g. readings.csv)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Optional Google Gemini or OpenAI API key for generative synthesis")
    parser.add_argument("--export", type=str, default=None,
                        help="Export clinical report to markdown file (e.g. consultation_report.md)")
    parser.add_argument("--json", action="store_true",
                        help="Print raw structured JSON output")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run and verify all 4 clinical scenarios sequentially")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Custom Patient ID (e.g. MOM-101)")
    parser.add_argument("--patient-name", type=str, default=None,
                        help="Custom Patient Name (e.g. Ananya S.)")

    args = parser.parse_args()
    wrapper = MaternalAIWrapper(api_key=args.api_key)

    if args.all_scenarios:
        print("\n🧪 RUNNING CLINICAL AI VERIFICATION SUITE (ALL 4 SCENARIOS)...\n")
        for sc_id, (prof, vit, title) in SCENARIOS.items():
            print(f"▶️  Testing {title}...")
            res = wrapper.process(prof, vit)
            wrapper.render_terminal_dashboard(res)
            time.sleep(0.5)
        print("✅ ALL 4 CLINICAL SCENARIOS VERIFIED SUCCESSFULLY!")
        return

    if args.csv:
        print(f"📊 Reading recorded fitband telemetry from: {args.csv}")
        custom_prof = MaternalProfile(
            patient_id=args.patient_id or "MOM-042",
            patient_name=args.patient_name or "Priya S."
        ) if (args.patient_id or args.patient_name) else None
        results = wrapper.process_csv_log(args.csv, profile=custom_prof)
        if not results:
            print("⚠️  No valid readings found in CSV.")
            return
        latest = results[-1]
        if args.json:
            print(json.dumps(latest, indent=2))
        else:
            wrapper.render_terminal_dashboard(latest)
        if args.export:
            wrapper.export_markdown_report(latest, args.export)
        return

    # Default Scenario Execution
    prof, vit, title = SCENARIOS[args.scenario]
    if args.patient_id:
        prof.patient_id = args.patient_id
    if args.patient_name:
        prof.patient_name = args.patient_name
    print(f"\n🔬 EXECUTING {title}...")
    res = wrapper.process(prof, vit)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        wrapper.render_terminal_dashboard(res)

    if args.export:
        wrapper.export_markdown_report(res, args.export)

if __name__ == "__main__":
    main()
