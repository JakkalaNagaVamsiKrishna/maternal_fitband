"""
Maternal FitBand — Deterministic Clinical Guideline Layer
=========================================================
Strictly rule-based clinical decision layer grounded in:
  - ACOG Practice Bulletin No. 222 (Gestational Hypertension & Preeclampsia)
  - NICE NG133 (Hypertension in Pregnancy)
  - WHO Guidelines on Maternal and Perinatal Health
  - ADA / ACOG Guidelines for Glycemic Targets in Pregnancy

LLMs (Gemini) are strictly prohibited from determining alert levels,
diagnoses, or clinical severity. They only rephrase rule determinations
into a reassuring, empathetic tone.
"""

from typing import Dict, Any, List, Optional

PROTOTYPE_DISCLAIMER = (
    "Research Prototype: Vital estimations (Blood Pressure, Glucose) require clinical "
    "validation and do not replace professional medical advice, laboratory testing, "
    "or standard prenatal care."
)

RED_FLAG_SYMPTOM_DEFINITIONS = {
    "severe_headache": {
        "title": "Severe Persistent Headache",
        "detail": "Headache that does not go away with rest or approved pain relief"
    },
    "visual_disturbances": {
        "title": "Visual Disturbances",
        "detail": "Blurriness, flashes of light, floaters, or blind spots"
    },
    "epigastric_pain": {
        "title": "Upper-Right Abdominal Pain",
        "detail": "Severe pain below the ribs or upper right stomach"
    },
    "shortness_of_breath": {
        "title": "Shortness of Breath",
        "detail": "Sudden difficulty breathing or shortness of breath at rest"
    },
    "sudden_swelling": {
        "title": "Sudden Severe Swelling",
        "detail": "Rapid onset swelling in hands, face, or around eyes"
    },
    "decreased_fetal_movement": {
        "title": "Decreased Baby Movement",
        "detail": "Noticeable drop in kick counts or movement after week 28"
    },
    "vaginal_bleeding": {
        "title": "Bleeding / Fluid Leakage",
        "detail": "Any vaginal bleeding or fluid leakage"
    }
}


def evaluate_blood_pressure(sbp: Optional[int], dbp: Optional[int]) -> Dict[str, Any]:
    """
    Evaluates blood pressure according to ACOG and NICE guidelines.
    ACOG Thresholds:
      - SBP >= 160 or DBP >= 110: Severe Hypertension (Urgent Care)
      - SBP >= 140 or DBP >= 90:  Gestational Hypertension
      - SBP < 90 or DBP < 60:     Hypotension
      - Otherwise:                Normotensive
    """
    if sbp is None or dbp is None:
        return {
            "status": "Unavailable",
            "tier": "INFO",
            "color": "gray",
            "summary": "Blood pressure estimation is currently calibrating.",
            "action": "Ensure fitband is snug on your wrist and stay still for 30 seconds."
        }

    # Bounded sanity
    sbp_val = max(60, min(230, int(sbp)))
    dbp_val = max(35, min(140, int(dbp)))

    if sbp_val >= 160 or dbp_val >= 110:
        return {
            "status": "Stage 2 Severe Hypertension",
            "tier": "CRITICAL",
            "color": "red",
            "summary": f"Estimated blood pressure ({sbp_val}/{dbp_val} mmHg) is severely elevated.",
            "action": (
                "Seek urgent in-person medical care immediately. ACOG guidelines consider "
                "BP >= 160/110 mmHg in pregnancy an urgent clinical emergency requiring immediate evaluation."
            ),
            "urgent": True
        }
    elif sbp_val >= 140 or dbp_val >= 90:
        return {
            "status": "Stage 1 Gestational Hypertension",
            "tier": "HIGH",
            "color": "amber",
            "summary": f"Estimated blood pressure ({sbp_val}/{dbp_val} mmHg) is elevated above normal prenatal target.",
            "action": (
                "Sit comfortably, rest quietly for 10 minutes, and re-test. Contact your "
                "obstetrician or prenatal clinic today to inform them of this elevated reading."
            ),
            "urgent": False
        }
    elif sbp_val < 90 or dbp_val < 60:
        return {
            "status": "Maternal Hypotension",
            "tier": "MEDIUM",
            "color": "blue",
            "summary": f"Estimated blood pressure ({sbp_val}/{dbp_val} mmHg) is lower than usual.",
            "action": (
                "Rest on your left side, drink a tall glass of cool water or electrolyte fluid, "
                "and stand up slowly to prevent lightheadedness."
            ),
            "urgent": False
        }
    else:
        return {
            "status": "Normotensive (Healthy)",
            "tier": "LOW",
            "color": "green",
            "summary": f"Estimated blood pressure ({sbp_val}/{dbp_val} mmHg) is in a healthy range.",
            "action": "Continue your normal daily activities and maintain regular hydration.",
            "urgent": False
        }


def evaluate_glucose(glucose_mg_dl: Optional[float], meal_context: str = "postprandial_1h") -> Dict[str, Any]:
    """
    Evaluates estimated blood sugar.
    Strict Medical Rule: Wearables CANNOT diagnose Gestational Diabetes.
    Diagnosis requires laboratory oral glucose tolerance tests (OGTT).
    """
    if glucose_mg_dl is None or glucose_mg_dl <= 0:
        return {
            "status": "Calibrating",
            "tier": "INFO",
            "color": "gray",
            "summary": "Non-invasive glucose reading is stabilizing.",
            "action": "Keep wrist resting quietly during optical reading."
        }

    val = round(float(glucose_mg_dl), 1)

    if val < 60.0:
        return {
            "status": "Hypoglycemia Alert",
            "tier": "CRITICAL",
            "color": "red",
            "summary": f"Estimated glucose ({val} mg/dL) is lower than safe prenatal target (<60 mg/dL).",
            "action": (
                "Consume 15 grams of quick-acting carbohydrates right away (half a cup of orange juice, "
                "or 1 tablespoon of honey). Rest and re-check. Contact provider if feeling dizzy."
            ),
            "urgent": True
        }
    elif val >= 180.0:
        return {
            "status": "Marked Postprandial Elevation",
            "tier": "HIGH",
            "color": "red",
            "summary": f"Estimated glucose ({val} mg/dL) is substantially above normal pregnancy targets.",
            "action": (
                "Drink 300 mL of water, avoid carbohydrates/juices, and mention this reading to your "
                "healthcare team. Reminder: this device cannot diagnose gestational diabetes; formal diagnosis requires a lab test."
            ),
            "urgent": False
        }
    elif val >= 140.0:
        return {
            "status": "Elevated Post-Meal Glucose",
            "tier": "MEDIUM",
            "color": "amber",
            "summary": f"Estimated glucose ({val} mg/dL) is slightly above the 1-hour target (140 mg/dL).",
            "action": (
                "Take a light, relaxing 10-minute stroll and drink water to assist natural glucose uptake. "
                "Keep a log of post-meal foods to review with your doctor."
            ),
            "urgent": False
        }
    elif val < 65.0:
        return {
            "status": "Borderline Low Glucose",
            "tier": "MEDIUM",
            "color": "amber",
            "summary": f"Estimated glucose ({val} mg/dL) is on the lower edge.",
            "action": "Have a healthy snack pairing protein with complex carbs (e.g., apple slices with peanut butter).",
            "urgent": False
        }
    else:
        return {
            "status": "Optimal Glycemic Target",
            "tier": "LOW",
            "color": "green",
            "summary": f"Estimated glucose ({val} mg/dL) is within target range for pregnancy.",
            "action": "Great job. Continue eating balanced meals rich in fiber, lean protein, and leafy vegetables.",
            "urgent": False
        }


def evaluate_heart_rate(hr_bpm: int) -> Dict[str, Any]:
    """Evaluates maternal heart rate with physiological adaptations of pregnancy accounted for."""
    hr = int(round(hr_bpm))
    if hr >= 125:
        return {
            "status": "Maternal Tachycardia",
            "tier": "HIGH",
            "color": "red",
            "summary": f"Heart rate ({hr} BPM) is elevated.",
            "action": "Sit down immediately, rest in a cool room, take slow breaths, and drink water. If accompanied by chest tightness or flutter, contact your clinic."
        }
    elif hr >= 105:
        return {
            "status": "Mild Elevation",
            "tier": "MEDIUM",
            "color": "amber",
            "summary": f"Heart rate ({hr} BPM) is slightly fast.",
            "action": "Rest quietly and ensure adequate hydration."
        }
    elif hr < 50:
        return {
            "status": "Maternal Bradycardia",
            "tier": "MEDIUM",
            "color": "blue",
            "summary": f"Heart rate ({hr} BPM) is lower than usual.",
            "action": "Lie comfortably on your left side. Contact your care provider if you experience lightheadedness."
        }
    else:
        return {
            "status": "Normal Pregnancy Rhythm",
            "tier": "LOW",
            "color": "green",
            "summary": f"Heart rate ({hr} BPM) is in healthy harmony.",
            "action": "Your cardiovascular rhythm is stable."
        }


def evaluate_kinematics(is_recumbent: bool, supine_alert: bool, trimester: int,
                        activity_code: int, vm_std: float) -> Dict[str, Any]:
    """
    Evaluates sleep posture and activity levels.
    Supine (flat on back) posture in Trimester 2/3 causes inferior vena cava compression.
    """
    if supine_alert and trimester >= 2:
        return {
            "status": "Supine Posture Alert (Roll to Left Side)",
            "tier": "CRITICAL",
            "color": "red",
            "summary": "You are lying flat on your back (supine position).",
            "action": (
                "Roll immediately onto your LEFT side. In the 2nd and 3rd trimesters, sleeping on your "
                "back compresses the vena cava, reducing maternal blood return and placental blood flow."
            ),
            "supine": True
        }

    if activity_code == 3 or vm_std > 0.85:
        return {
            "status": "Overexertion Warning",
            "tier": "HIGH",
            "color": "amber",
            "summary": "High movement impact or intense exertion detected.",
            "action": "Take a break, sit down in a cool shaded area, drink water, and allow your body to cool down.",
            "supine": False
        }

    if is_recumbent:
        return {
            "status": "Safe Lateral Rest",
            "tier": "LOW",
            "color": "green",
            "summary": "Resting safely in an ergonomic lateral posture.",
            "action": "Comfortable lateral positioning supports maternal and fetal blood flow.",
            "supine": False
        }

    return {
        "status": "Normal Activity",
        "tier": "LOW",
        "color": "green",
        "summary": "Gentle prenatal activity detected.",
        "action": "Stay active with regular walking, but avoid abrupt twisting or high-impact jumping.",
        "supine": False
    }


def evaluate_guidelines(
    vitals: Dict[str, Any],
    kinematics: Dict[str, Any],
    profile: Dict[str, Any],
    active_symptoms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Synthesizes the overall deterministic clinical report.
    Non-negotiable rule: Any active red-flag symptom immediately elevates
    overall status to EMERGENCY, regardless of what biometric sensors read.
    """
    active_symptoms = active_symptoms or []
    gestational_week = int(profile.get("gestational_week", 28))
    trimester = 1 if gestational_week <= 13 else (2 if gestational_week <= 27 else 3)
    meal_context = profile.get("meal_context", "postprandial_1h")

    # Evaluate individual dimensions
    bp_eval = evaluate_blood_pressure(vitals.get("sbp_mmhg"), vitals.get("dbp_mmhg"))
    gluc_eval = evaluate_glucose(vitals.get("glucose_mg_dl"), meal_context)
    hr_eval = evaluate_heart_rate(vitals.get("heart_rate_bpm", vitals.get("hr", 75)))
    kin_eval = evaluate_kinematics(
        is_recumbent=kinematics.get("is_recumbent", False),
        supine_alert=kinematics.get("supine_alert", False),
        trimester=trimester,
        activity_code=kinematics.get("activity_code", 0),
        vm_std=kinematics.get("vm_std_g", 0.1)
    )

    # Emergency symptoms check (Absolute Override)
    red_flags_present = [s for s in active_symptoms if s in RED_FLAG_SYMPTOM_DEFINITIONS]
    red_flag_details = [RED_FLAG_SYMPTOM_DEFINITIONS[s] for s in red_flags_present]

    if red_flags_present:
        overall_tier = "CRITICAL"
        overall_badge = "URGENT MEDICAL ATTENTION"
        overall_color = "red"
        headline = "Emergency Prenatal Signs Detected — Contact Doctor Immediately"
        primary_action = (
            "You reported warning symptoms ("
            + ", ".join([d["title"] for d in red_flag_details])
            + "). Clinical guidelines require you to contact your obstetrician, midwife, or visit "
            "labor & delivery emergency care immediately, regardless of current sensor numbers."
        )
    elif bp_eval["tier"] == "CRITICAL" or gluc_eval["tier"] == "CRITICAL" or kin_eval["tier"] == "CRITICAL":
        overall_tier = "CRITICAL"
        overall_badge = "CRITICAL ALERT"
        overall_color = "red"
        headline = "Critical Vital Parameter Exceeded"
        primary_action = bp_eval.get("action") if bp_eval["tier"] == "CRITICAL" else kin_eval["action"]
    elif bp_eval["tier"] == "HIGH" or gluc_eval["tier"] == "HIGH" or hr_eval["tier"] == "HIGH" or kin_eval["tier"] == "HIGH":
        overall_tier = "HIGH"
        overall_badge = "ATTENTION REQUIRED"
        overall_color = "amber"
        headline = "Vital Parameter Above Prenatal Target"
        primary_action = bp_eval["action"] if bp_eval["tier"] == "HIGH" else gluc_eval["action"]
    elif bp_eval["tier"] == "MEDIUM" or gluc_eval["tier"] == "MEDIUM" or hr_eval["tier"] == "MEDIUM":
        overall_tier = "MEDIUM"
        overall_badge = "MONITOR"
        overall_color = "blue"
        headline = "Mild Variance Detected"
        primary_action = gluc_eval["action"] if gluc_eval["tier"] == "MEDIUM" else hr_eval["action"]
    else:
        overall_tier = "LOW"
        overall_badge = "ALL NORMAL"
        overall_color = "green"
        headline = "All Vitals in Healthy Harmony"
        primary_action = "Your vital sign estimates are stable and within standard prenatal targets."

    return {
        "overall_tier": overall_tier,
        "overall_badge": overall_badge,
        "overall_color": overall_color,
        "headline": headline,
        "primary_action": primary_action,
        "cards": {
            "blood_pressure": bp_eval,
            "glucose": gluc_eval,
            "heart_rate": hr_eval,
            "kinematics": kin_eval
        },
        "red_flag_symptoms_active": red_flags_present,
        "red_flag_details": red_flag_details,
        "disclaimer": PROTOTYPE_DISCLAIMER
    }
