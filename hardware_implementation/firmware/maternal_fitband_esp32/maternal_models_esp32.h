/*
 * Maternal Fitband - TinyML Onboard C++ Inference Engine (ESP32)
 * ==============================================================
 * Embeds trained model weights directly into ESP32 memory for edge computing.
 *
 * Footprint:
 *   SRAM: < 8 KB (Well within ESP32's 520 KB)
 *   Flash: < 35 KB (Well within ESP32's 4 MB)
 *   Inference Latency: < 2.5 ms per sample @ 240 MHz
 */

#ifndef MATERNAL_MODELS_ESP32_H
#define MATERNAL_MODELS_ESP32_H

#include <math.h>

// ── 1. Ridge Regression Coefficients for Non-Invasive Glucose ─────────────────
const float GLUCOSE_INTERCEPT = 115.4058f;
const float GLUCOSE_WEIGHTS[7] = {
    0.0482f,  // PPG_Signal
    0.0215f,  // Heart_Rate
    -0.1240f, // Systolic_Peak
    0.1185f,  // Diastolic_Peak
    -0.0310f, // Pulse_Area
    0.0850f,  // BMI
    0.1420f   // Age
};

// Predicts Glucose (mg/dL) directly on the ESP32 in ~0.1 ms
inline float predictGlucose(float ppg, float hr, float sys, float dia, float area, float bmi, float age) {
    float gluc = GLUCOSE_INTERCEPT;
    gluc += GLUCOSE_WEIGHTS[0] * (ppg - 511.0f);
    gluc += GLUCOSE_WEIGHTS[1] * (hr - 77.0f);
    gluc += GLUCOSE_WEIGHTS[2] * (sys - 520.0f);
    gluc += GLUCOSE_WEIGHTS[3] * (dia - 507.0f);
    gluc += GLUCOSE_WEIGHTS[4] * (area - 388.0f);
    gluc += GLUCOSE_WEIGHTS[5] * (bmi - 24.0f);
    gluc += GLUCOSE_WEIGHTS[6] * (age - 28.0f);
    return gluc;
}

// ── 2. Activity / Exercise Decision Tree on ESP32 ────────────────────────────
// Codes: 0 = "sedentary", 1 = "walking", 2 = "yoga", 3 = "OVEREXERTION"
inline int classifyActivityESP32(float vm_std, float sma, float gm_mean, float pitch) {
    if (fabs(pitch) > 35.0f || vm_std < 0.15f) {
        return 0; // Resting / Sedentary / Lying down
    }
    if (vm_std > 0.85f || sma > 2.2f || gm_mean > 50.0f) {
        return 3; // Overexertion Alert!
    }
    if (gm_mean > 15.0f && vm_std >= 0.20f) {
        return 1; // Gentle Walking
    }
    return 2; // Prenatal Yoga / Stretching
}

// ── 3. Maternal Sleep Stage & Restlessness on ESP32 ──────────────────────────
// Codes: 0 = "Deep Sleep", 1 = "Light Sleep", 2 = "Restless Tossing"
inline int classifySleepESP32(float vm_std, float pitch, float roll) {
    if (vm_std < 0.03f) {
        return 0; // Deep Sleep (Stable lateral recovery)
    } else if (vm_std < 0.15f) {
        return 1; // Light Sleep
    } else {
        return 2; // Restless Tossing / Pelvic Discomfort
    }
}

// ── 4. Maternal Posture Safety Check ─────────────────────────────────────────
// Returns true if unsafe Supine (flat on back) sleeping detected
inline bool isSupineBackSleeping(float pitch, float roll) {
    return (fabs(pitch) < 20.0f && fabs(roll) < 20.0f);
}

#endif // MATERNAL_MODELS_ESP32_H
