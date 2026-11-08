"""
Maternal Fitband — Gemini Clinical AI & Maternal Companion Engine
=================================================================
Transforms raw biometric measurements and ML classification results into
an empathetic, intuitive, and medically safe report for expectant mothers.

Dual Synthesis:
  1. Google Gemini API (gemini-2.5-flash / gemini-1.5-flash)
  2. Deterministic ACOG & ADA Clinical Knowledge Engine (100% offline fallback)
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class MaternalGeminiEngine:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    def set_api_key(self, api_key: str):
        if api_key and api_key.strip():
            self.api_key = api_key.strip()
            print("[Gemini Engine] API Key updated.")

    def generate_maternal_report(self, classified_data: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the maternal health report.
        Tries Gemini API first if key is present; falls back seamlessly to deterministic ACOG engine.
        """
        vitals = classified_data.get("vitals", {})
        kinematics = classified_data.get("kinematics", {})
        clinical = classified_data.get("clinical_status", {})

        # Build prompt and try Gemini
        gemini_text = None
        source = "ACOG Clinical Knowledge Engine (Offline Mode)"

        if self.api_key:
            try:
                gemini_text = self._call_gemini(classified_data, profile)
                if gemini_text:
                    source = f"Google Gemini AI ({self.model})"
            except Exception as e:
                print(f"[WARN] Gemini API call failed: {e}. Falling back to deterministic engine.")

        # If Gemini succeeded, parse or format its sections
        if gemini_text:
            report_sections = self._parse_gemini_output(gemini_text, classified_data, profile)
        else:
            report_sections = self._generate_deterministic_report(classified_data, profile)

        report_sections["engine_source"] = source
        report_sections["raw_vitals"] = vitals
        report_sections["kinematics"] = kinematics
        report_sections["clinical_status"] = clinical
        report_sections["patient_profile"] = profile

        return report_sections

    def _call_gemini(self, classified_data: Dict[str, Any], profile: Dict[str, Any]) -> Optional[str]:
        vitals = classified_data["vitals"]
        kin = classified_data["kinematics"]
        clin = classified_data["clinical_status"]

        prompt = f"""You are the warm, supportive Antigravity Maternal Health Fitband Care Companion.
Your voice is reassuring, medically accurate, practical, and deeply encouraging.
You are talking directly to pregnant mother {profile.get('name', 'Mom')}, who is {profile.get('gestational_week', 28)} weeks pregnant (Trimester {clin.get('trimester', 3)}).

Biometric readings from her wearable band:
- Non-Invasive Blood Glucose: {vitals.get('glucose_mg_dl')} mg/dL ({vitals.get('glycemic_label')})
- Heart Rate: {vitals.get('heart_rate_bpm')} bpm
- Oxygen Saturation (SpO2): {vitals.get('spo2_pct')}%
- Blood Pressure: {vitals.get('blood_pressure')} mmHg ({vitals.get('htn_status')})
- Activity: {kin.get('activity_label')}
- Sleep / Rest: {kin.get('sleep_label')} (Supine/Flat-on-back Alert: {kin.get('supine_alert')})
- Overall Maternal Risk Tier: {clin.get('overall_risk_level')}
- Clinical Flags: {', '.join(clin.get('risk_flags', [])) or 'None (All normal)'}

Guidelines:
1. Explain in simple, intuitive, non-technical language what is happening with her body and baby right now.
2. If supine back-sleeping is True, urgently but gently guide her to immediately turn to her LEFT side with a pillow between knees to optimize blood flow to the placenta.
3. If glucose is elevated (>140 mg/dL), provide simple meal and walking tips without causing panic.
4. Structure your response into these 4 clear sections using emojis:
   ### 💖 How You & Baby Are Doing Right Now
   ### 🥗 Nourishment & Blood Sugar Tip
   ### 🌙 Rest, Sleep & Posture Coaching
   ### 🏃 Movement & Energy Level
   ### 🩺 Doctor Note (SBAR Summary)
5. Keep explanations warm, concise, and under 350 words total.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1024
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            candidates = res_json.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        return None

    def _parse_gemini_output(self, text: str, data: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structures the raw Gemini markdown into accessible card components.
        """
        return {
            "full_text": text,
            "condition_summary": self._extract_section(text, "How You & Baby Are Doing Right Now", "Nourishment"),
            "nutrition_guidance": self._extract_section(text, "Nourishment & Blood Sugar Tip", "Rest, Sleep"),
            "sleep_guidance": self._extract_section(text, "Rest, Sleep & Posture Coaching", "Movement"),
            "activity_guidance": self._extract_section(text, "Movement & Energy Level", "Doctor Note"),
            "doctor_sbar": self._extract_section(text, "Doctor Note", None) or self._build_sbar(data, profile)
        }

    def _extract_section(self, text: str, start_header: str, end_header: Optional[str]) -> str:
        lower_text = text.lower()
        start_idx = lower_text.find(start_header.lower())
        if start_idx == -1:
            return ""
        
        # Advance past the header line
        newline_idx = text.find("\n", start_idx)
        content_start = newline_idx + 1 if newline_idx != -1 else start_idx + len(start_header)

        if end_header:
            end_idx = lower_text.find(end_header.lower(), content_start)
            if end_idx != -1:
                return text[content_start:end_idx].strip("# \n\r")

        return text[content_start:].strip("# \n\r")

    def _generate_deterministic_report(self, data: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        ACOG-aligned deterministic clinical report generator.
        """
        vitals = data["vitals"]
        kin = data["kinematics"]
        clin = data["clinical_status"]
        t = clin.get("trimester", 3)
        week = clin.get("gestational_week", 28)
        gluc = vitals.get("glucose_mg_dl", 100.0)
        hr = vitals.get("heart_rate_bpm", 76)
        sbp = vitals.get("sbp_mmhg", 118)
        dbp = vitals.get("dbp_mmhg", 78)
        supine = kin.get("supine_alert", False)

        # 1. Condition Summary
        summary_points = []
        if clin["overall_risk_level"] == "CRITICAL" and supine:
            summary_points.append("🚨 **Immediate Posture Adjustment Needed:** You are currently lying flat on your back. Please roll over to your left side right now to keep blood flowing easily to your baby.")
        elif clin["overall_risk_level"] in ("HIGH", "CRITICAL"):
            summary_points.append(f"⚠️ **Attention Needed:** We noticed elevated physiological readings (Glucose {gluc} mg/dL, BP {sbp}/{dbp} mmHg). Rest calmly and follow the guidance below.")
        else:
            summary_points.append(f"💖 **You and Baby Are Doing Wonderful!** Your vital signs are currently stable and in healthy harmony for Week {week} (Trimester {t}).")

        # 2. Nutrition & Blood Sugar
        diet_points = []
        if gluc >= 140:
            diet_points.append("🥗 **Post-Meal Glucose Surge:** Your blood sugar is currently elevated. Drink a tall glass of cool water (300 mL) and take a gentle 10-minute slow indoor stroll to help your muscles naturally absorb glucose.")
            diet_points.append("🚫 **For your next snack:** Skip sugary snacks and juices. Choose high-protein, fiber-rich options like a handful of almonds, cucumbers with hummus, or Greek yogurt.")
        elif gluc < 60:
            diet_points.append("⚠️ **Low Blood Sugar Alert:** Have 15g of fast-acting carbohydrate right away (half a cup of fresh fruit juice or 1 spoon of honey), followed by whole grain crackers.")
        else:
            diet_points.append("✅ **Blood Sugar in Ideal Target:** Your glycemic response is steady and nourishing your baby effectively.")
            diet_points.append("🥑 **Tip:** Continue pairing complex carbohydrates with healthy proteins and fats (eggs, nuts, lentils, avocado) and stay well-hydrated.")

        # 3. Sleep & Posture
        sleep_points = []
        if supine and t >= 2:
            sleep_points.append("🚨 **Roll Onto Your Left Side:** Sleeping flat on your back in the second or third trimester presses the growing womb against your main blood vessels (inferior vena cava). Rolling to your left restores optimal oxygen flow to your little one.")
            sleep_points.append("🛌 **Pillow Trick:** Tuck a firm pillow behind your lower back and place a cushion between your bent knees for support.")
        elif kin.get("sleep_code") == 2:
            sleep_points.append("🧘 **Restless Night:** We noticed some tossing and turning. Try a 5-minute deep belly breathing routine in a cool room (65-68°F / 19°C) and use a maternity U-pillow.")
        else:
            sleep_points.append("🛌 **Safe Posture Detected:** You are resting in a safe lateral posture that supports optimal blood circulation.")

        # 4. Movement & Energy
        activity_points = []
        if kin.get("activity_code") == 3:
            activity_points.append("🛑 **Time to Rest:** High kinematic impact or sudden exertion detected! Sit down comfortably in a cool spot, take slow deep breaths, and drink water.")
        elif kin.get("activity_code") == 1:
            activity_points.append("🏃 **Great Walking Rhythm:** Gentle walking is fantastic for prenatal circulation and natural glucose control. Aim for 20-30 minutes.")
        else:
            activity_points.append("🧘 **Gentle Movement:** Remember to stand up, stretch your ankles, and take short strolls if you've been sitting for more than an hour.")

        sbar = self._build_sbar(data, profile)

        return {
            "full_text": "\n\n".join(summary_points + diet_points + sleep_points + activity_points),
            "condition_summary": "\n".join(summary_points),
            "nutrition_guidance": "\n".join(diet_points),
            "sleep_guidance": "\n".join(sleep_points),
            "activity_guidance": "\n".join(activity_points),
            "doctor_sbar": sbar
        }

    def _build_sbar(self, data: Dict[str, Any], profile: Dict[str, Any]) -> str:
        vitals = data["vitals"]
        kin = data["kinematics"]
        clin = data["clinical_status"]
        return (
            f"**SITUATION:** Patient {profile.get('name', 'Mom')} (Gestational Week {clin.get('gestational_week', 28)}, Trimester {clin.get('trimester', 3)}). "
            f"Current Vitals: Glucose {vitals.get('glucose_mg_dl')} mg/dL, HR {vitals.get('heart_rate_bpm')} bpm, SpO2 {vitals.get('spo2_pct')}%, "
            f"BP {vitals.get('blood_pressure')} mmHg, Activity: '{kin.get('activity_label')}', Supine Alert: {kin.get('supine_alert')}.\n\n"
            f"**BACKGROUND:** Context: {profile.get('meal_context', 'postprandial').replace('_', ' ').title()}. "
            f"Pre-existing risk factors: {', '.join(profile.get('conditions', ['None']))}.\n\n"
            f"**ASSESSMENT:** Overall Risk Level: {clin.get('overall_risk_level')}. "
            f"Active Flags: {', '.join(clin.get('risk_flags', [])) or 'No acute anomalies'}.\n\n"
            f"**RECOMMENDATION:** Maternal: Maintain left-lateral sleep posture and low-GI nutritional intake. "
            f"Clinical: {'Review continuous glycemic log and evaluate ambulatory BP monitoring.' if clin.get('overall_risk_level') in ('HIGH', 'CRITICAL') else 'Continue routine antenatal care.'}"
        )

# Global singleton
gemini_engine = MaternalGeminiEngine()
