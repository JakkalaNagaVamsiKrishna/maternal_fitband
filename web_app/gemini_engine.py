"""
Maternal FitBand — Gemini Clinical AI & Maternal Companion Engine
=================================================================
Transforms deterministic guideline evaluations and multi-sensor biometrics
into an empathetic, intuitive, and reassuring report for expectant mothers.

Architecture Rule (per Clinical Audit):
  The guideline layer in Python determines ALL alert levels and medical actions.
  Gemini ONLY rephrases predetermined decisions in a reassuring tone.
  Gemini is strictly prohibited from altering risk tiers or issuing diagnoses.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List

from .clinical_guidelines import evaluate_guidelines, PROTOTYPE_DISCLAIMER


class MaternalGeminiEngine:
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

    def set_api_key(self, api_key: str):
        if api_key and api_key.strip():
            self.api_key = api_key.strip()
            print("[Gemini Engine] API Key updated.")

    def generate_maternal_report(
        self,
        classified_data: Dict[str, Any],
        profile: Dict[str, Any],
        active_symptoms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates the maternal health report.
        1. Evaluates deterministic ACOG/NICE clinical rules.
        2. If Gemini API is available, prompts Gemini to rephrase predetermined guidance in a reassuring tone.
        3. Falls back seamlessly to offline rule synthesis if API is unavailable.
        """
        vitals = classified_data.get("vitals", {})
        kinematics = classified_data.get("kinematics", {})
        active_symptoms = active_symptoms or []

        # 1. Deterministic Guideline Evaluation (Authoritative Ground Truth)
        guidelines = evaluate_guidelines(vitals, kinematics, profile, active_symptoms)

        # 2. Rephrase via Gemini if API key present
        gemini_text = None
        source = "ACOG / NICE Rule Engine (Offline Deterministic Mode)"

        if self.api_key:
            try:
                gemini_text = self._call_gemini(classified_data, profile, guidelines)
                if gemini_text:
                    source = f"Google Gemini AI ({self.model}) [Rephrasing Guideline Layer]"
            except Exception as e:
                print(f"[WARN] Gemini API call failed: {e}. Using deterministic guideline guidance.")

        if gemini_text:
            report_sections = self._parse_gemini_output(gemini_text, classified_data, profile, guidelines)
        else:
            report_sections = self._generate_deterministic_report(classified_data, profile, guidelines)

        report_sections["engine_source"] = source
        report_sections["raw_vitals"] = vitals
        report_sections["kinematics"] = kinematics
        report_sections["clinical_guidelines"] = guidelines
        report_sections["patient_profile"] = profile
        report_sections["prototype_disclaimer"] = PROTOTYPE_DISCLAIMER

        return report_sections

    def _call_gemini(
        self,
        classified_data: Dict[str, Any],
        profile: Dict[str, Any],
        guidelines: Dict[str, Any]
    ) -> Optional[str]:
        vitals = classified_data.get("vitals", {})
        kin = classified_data.get("kinematics", {})
        cards = guidelines.get("cards", {})
        bp = cards.get("blood_pressure", {})
        gluc = cards.get("glucose", {})
        kin_card = cards.get("kinematics", {})
        red_flags = guidelines.get("red_flag_details", [])

        red_flags_str = ", ".join([rf["title"] for rf in red_flags]) if red_flags else "None reported"

        prompt = f"""You are the supportive, warm Maternal Health FitBand Companion for pregnant mother {profile.get('name', 'Mom')}.
She is {profile.get('gestational_week', 28)} weeks pregnant (Trimester {profile.get('trimester', 3)}).

*** STRICT CLINICAL BOUNDARIES (NON-NEGOTIABLE) ***
1. Alert levels and medical decisions have ALREADY been computed by the deterministic clinical guideline engine.
2. DO NOT change the risk tier ({guidelines.get('overall_tier')}).
3. DO NOT diagnose any condition (e.g., you cannot diagnose gestational diabetes or preeclampsia; diagnosis requires lab tests and a physician).
4. If red-flag symptoms or critical alerts are present, your tone must be calm, direct, and firmly insist on contacting a physician or visiting labor & delivery immediately.
5. Your sole responsibility is to rephrase the clinical decisions below in an encouraging, clear, and reassuring tone.

PREDETERMINED CLINICAL FINDINGS:
- Overall Status: {guidelines.get('overall_badge')} ({guidelines.get('overall_tier')})
- Headline: {guidelines.get('headline')}
- Primary Required Action: {guidelines.get('primary_action')}
- Reported Emergency Symptoms: {red_flags_str}
- Blood Pressure: {vitals.get('blood_pressure')} mmHg -> Status: {bp.get('status')}. Action: {bp.get('action')}
- Blood Glucose Estimate: {vitals.get('glucose_mg_dl')} mg/dL -> Status: {gluc.get('status')}. Action: {gluc.get('action')}
- Heart Rate: {vitals.get('heart_rate_bpm')} BPM
- Posture & Movement: {kin_card.get('status')}. Action: {kin_card.get('action')}

Format your response strictly into these 4 concise sections:
### 💖 How You & Baby Are Doing Right Now
### 🥗 Nourishment & Blood Sugar Tip
### 🌙 Rest, Sleep & Posture Coaching
### 🩺 Doctor Summary (SBAR Format)

Keep total length under 300 words. Always include reassurance where appropriate without downplaying necessary medical contact.
"""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
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

    def _parse_gemini_output(
        self,
        text: str,
        data: Dict[str, Any],
        profile: Dict[str, Any],
        guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "full_text": text,
            "condition_summary": self._extract_section(text, "How You & Baby Are Doing Right Now", "Nourishment"),
            "nutrition_guidance": self._extract_section(text, "Nourishment & Blood Sugar Tip", "Rest, Sleep"),
            "sleep_guidance": self._extract_section(text, "Rest, Sleep & Posture Coaching", "Doctor Summary"),
            "doctor_sbar": self._extract_section(text, "Doctor Summary", None) or self._build_sbar(data, profile, guidelines)
        }

    def _extract_section(self, text: str, start_header: str, end_header: Optional[str]) -> str:
        lower_text = text.lower()
        start_idx = lower_text.find(start_header.lower())
        if start_idx == -1:
            return ""

        newline_idx = text.find("\n", start_idx)
        content_start = newline_idx + 1 if newline_idx != -1 else start_idx + len(start_header)

        if end_header:
            end_idx = lower_text.find(end_header.lower(), content_start)
            if end_idx != -1:
                return text[content_start:end_idx].strip("# \n\r")

        return text[content_start:].strip("# \n\r")

    def _generate_deterministic_report(
        self,
        data: Dict[str, Any],
        profile: Dict[str, Any],
        guidelines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pure rule-based clinical synthesis with 100% adherence to medical thresholds.
        """
        cards = guidelines.get("cards", {})
        bp = cards.get("blood_pressure", {})
        gluc = cards.get("glucose", {})
        kin_card = cards.get("kinematics", {})
        red_flags = guidelines.get("red_flag_details", [])

        # 1. Condition Summary
        summary_points = []
        if red_flags:
            titles = ", ".join([rf["title"] for rf in red_flags])
            summary_points.append(
                f"🚨 **Warning Signs Active ({titles}):** {guidelines.get('primary_action')}"
            )
        elif guidelines.get("overall_tier") == "CRITICAL":
            summary_points.append(f"⚠️ **Attention Required:** {guidelines.get('headline')}. {guidelines.get('primary_action')}")
        else:
            summary_points.append(f"💖 **You and Baby Are Doing Well!** {guidelines.get('headline')}. {guidelines.get('primary_action')}")

        # 2. Nutrition & Blood Sugar
        diet_points = [
            f"**Blood Sugar Status:** {gluc.get('summary', '')}",
            f"👉 **Action:** {gluc.get('action', '')}"
        ]

        # 3. Sleep & Posture
        sleep_points = [
            f"**Ergonomics & Movement:** {kin_card.get('summary', '')}",
            f"👉 **Recommendation:** {kin_card.get('action', '')}"
        ]

        sbar = self._build_sbar(data, profile, guidelines)

        return {
            "full_text": "\n\n".join(summary_points + diet_points + sleep_points),
            "condition_summary": "\n".join(summary_points),
            "nutrition_guidance": "\n".join(diet_points),
            "sleep_guidance": "\n".join(sleep_points),
            "doctor_sbar": sbar
        }

    def _build_sbar(self, data: Dict[str, Any], profile: Dict[str, Any], guidelines: Dict[str, Any]) -> str:
        vitals = data.get("vitals", {})
        kin = data.get("kinematics", {})
        cards = guidelines.get("cards", {})
        bp = cards.get("blood_pressure", {})
        gluc = cards.get("glucose", {})
        red_flags = guidelines.get("red_flag_details", [])

        sbar = (
            f"**SITUATION:** Patient {profile.get('name', 'Mom')} (Gestational Week {profile.get('gestational_week', 28)}, "
            f"Age {profile.get('age', 29)}). Estimated Vitals: BP {vitals.get('blood_pressure')} mmHg ({bp.get('status')}), "
            f"Glucose {vitals.get('glucose_mg_dl')} mg/dL ({gluc.get('status')}), HR {vitals.get('heart_rate_bpm')} BPM, "
            f"SpO2 {vitals.get('spo2_pct')}%. Reported Symptoms: {', '.join([rf['title'] for rf in red_flags]) or 'None'}.\n\n"
            f"**BACKGROUND:** Context: {profile.get('meal_context', 'postprandial').replace('_', ' ').title()}. "
            f"Pre-existing risk factors: {', '.join(profile.get('conditions', ['None']))}.\n\n"
            f"**ASSESSMENT:** Overall Guideline Tier: {guidelines.get('overall_badge')} ({guidelines.get('overall_tier')}). "
            f"Clinical Summary: {guidelines.get('headline')}.\n\n"
            f"**RECOMMENDATION:** {guidelines.get('primary_action')} "
            f"(Note: Prototype device estimates require formal clinical verification)."
        )
        return sbar


# Global singleton
gemini_engine = MaternalGeminiEngine()
