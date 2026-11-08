"""
Maternal Fitband — Universal Sensor Stream & Telemetry Bridge
=============================================================
Provides:
  1. Real-time Telemetry Ingestion from physical ESP32 (Serial / BLE / Wi-Fi HTTP)
  2. Built-in Multi-Scenario Physiological & Kinematic Sensor Simulator
  3. Real-time Listener Dispatch (SSE / WebSockets)
"""

import time
import math
import random
import threading
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

class SensorStreamManager:
    def __init__(self, ml_engine, gemini_engine):
        self.ml_engine = ml_engine
        self.gemini_engine = gemini_engine

        # Active state
        self.latest_raw: Dict[str, Any] = {}
        self.latest_classified: Dict[str, Any] = {}
        self.telemetry_history: List[Dict[str, Any]] = []
        self.max_history = 60

        # Profile state
        self.maternal_profile = {
            "name": "Priya S.",
            "patient_id": "MOM-042",
            "gestational_week": 28,
            "age": 29,
            "height_cm": 162.0,
            "weight_kg": 65.0,
            "meal_context": "postprandial_1h",
            "conditions": ["Gestational Diabetes Risk"]
        }

        # Subscribers
        self.subscribers: List[Callable[[Dict[str, Any]], None]] = []

        # Simulator State
        self.simulator_active = True
        self.current_scenario = "normal_resting"
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_step = 0

        # Hardware Serial State
        self.serial_port_name: Optional[str] = None
        self.serial_conn = None
        self._serial_thread: Optional[threading.Thread] = None

        # Start initial baseline
        self._init_baseline()
        self.start_simulator()

    def _init_baseline(self):
        raw = {
            "hr": 74, "spo2": 98, "ppg_signal": 512.0,
            "systolic_peak": 521.0, "diastolic_peak": 507.0, "pulse_area": 390.0,
            "pitch": 45.0, "roll": 25.0, "vm_mean": 1.01, "vm_std": 0.08,
            "sma": 1.25, "gm_mean": 12.0, "timestamp": datetime.now().isoformat()
        }
        self.ingest_raw(raw)

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def update_profile(self, new_profile: Dict[str, Any]):
        self.maternal_profile.update(new_profile)
        # Re-classify immediately
        if self.latest_raw:
            self.ingest_raw(self.latest_raw)

    def ingest_raw(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives raw sensor packets from ESP32 or simulator,
        executes ML classification, stores in history, and notifies subscribers.
        """
        raw_data["timestamp"] = datetime.now().strftime("%H:%M:%S")
        self.latest_raw = raw_data

        # Classify through ML models
        classified = self.ml_engine.classify(raw_data, self.maternal_profile)
        classified["timestamp"] = raw_data["timestamp"]
        classified["raw_inputs"] = raw_data
        classified["profile"] = self.maternal_profile
        classified["data_source"] = "Physical ESP32" if self.serial_conn else ("Simulator (" + self.current_scenario + ")")

        self.latest_classified = classified

        # Buffer history for charts
        chart_point = {
            "time": raw_data["timestamp"],
            "glucose": classified["vitals"]["glucose_mg_dl"],
            "hr": classified["vitals"]["heart_rate_bpm"],
            "spo2": classified["vitals"]["spo2_pct"],
            "sbp": classified["vitals"]["sbp_mmhg"],
            "dbp": classified["vitals"]["dbp_mmhg"],
            "vm": classified["kinematics"]["vm_mean_g"]
        }
        self.telemetry_history.append(chart_point)
        if len(self.telemetry_history) > self.max_history:
            self.telemetry_history.pop(0)

        # Notify active streaming listeners
        for cb in list(self.subscribers):
            try:
                cb(classified)
            except Exception:
                pass

        return classified

    # =========================================================================
    # Simulator Controls
    # =========================================================================

    def set_scenario(self, scenario: str):
        valid = ["normal_resting", "postprandial_hyperglycemia", "supine_sleep_alert", "overexertion_high_hr", "gestational_hypertension"]
        if scenario in valid:
            self.current_scenario = scenario
            print(f"[Simulator] Switched scenario to: {scenario}")
            return True
        return False

    def start_simulator(self):
        if not self.simulator_active:
            self.simulator_active = True
        if self._sim_thread is None or not self._sim_thread.is_alive():
            self._sim_thread = threading.Thread(target=self._run_simulator_loop, daemon=True)
            self._sim_thread.start()

    def stop_simulator(self):
        self.simulator_active = False

    def _run_simulator_loop(self):
        while True:
            if self.simulator_active and (self.serial_conn is None):
                self._sim_step += 1
                t = self._sim_step * 0.5
                raw = self._generate_simulated_packet(t, self.current_scenario)
                self.ingest_raw(raw)
            time.sleep(1.0)  # Stream at 1 Hz for stable web dashboards

    def _generate_simulated_packet(self, t: float, scenario: str) -> Dict[str, Any]:
        rnd = lambda: random.uniform(-0.5, 0.5)

        if scenario == "postprandial_hyperglycemia":
            # Mother after heavy meal: elevated glucose, moderate heart rate, sedentary
            hr = 84 + 2 * math.sin(t / 4) + rnd()
            ppg = 525.0 + 3 * math.sin(t / 2) + rnd()
            sys_peak = ppg + 14.0
            dia_peak = ppg - 6.0
            pulse_area = 430.0 + rnd() * 5
            pitch = 14.0 + rnd()
            roll = 18.0 + rnd()
            vm_mean = 1.02 + rnd() * 0.05
            vm_std = 0.12
            sma = 1.30

        elif scenario == "supine_sleep_alert":
            # Mother lying flat on back (pitch and roll < 20 deg), vena cava compression risk!
            hr = 68 + 2 * math.sin(t / 6) + rnd()
            ppg = 510.0 + rnd()
            sys_peak = ppg + 9.0
            dia_peak = ppg - 4.0
            pulse_area = 380.0
            pitch = 8.0 + rnd() * 0.5    # Flat back!
            roll = 10.0 + rnd() * 0.5   # Flat back!
            vm_mean = 0.98 + rnd() * 0.02
            vm_std = 0.02                # Deep sleep minimal movement
            sma = 0.95

        elif scenario == "overexertion_high_hr":
            # Mother walking briskly / jerky movement triggering high heart rate and overexertion
            hr = 138 + 4 * math.sin(t / 3) + rnd() * 2
            ppg = 535.0 + rnd() * 4
            sys_peak = ppg + 18.0
            dia_peak = ppg - 8.0
            pulse_area = 460.0
            pitch = 18.0 + rnd() * 3
            roll = 22.0 + rnd() * 3
            vm_mean = 2.45 + rnd() * 0.4
            vm_std = 1.15                # Strenuous movement!
            sma = 3.10                   # Impact forces!

        elif scenario == "gestational_hypertension":
            # Hypertensive surge: tachycardia and elevated vascular resistance
            hr = 104 + 3 * math.sin(t / 4) + rnd()
            ppg = 522.0 + rnd() * 2
            sys_peak = ppg + 22.0        # High pulse amplitude
            dia_peak = ppg - 11.0
            pulse_area = 440.0
            pitch = 12.0 + rnd()
            roll = 16.0 + rnd()
            vm_mean = 1.04 + rnd() * 0.04
            vm_std = 0.18
            sma = 1.40

        else: # "normal_resting"
            # Healthy, calm, left-lateral sleep posture or seated rest
            hr = 74 + 2 * math.sin(t / 5) + rnd()
            ppg = 512.0 + rnd()
            sys_peak = ppg + 10.0
            dia_peak = ppg - 4.5
            pulse_area = 390.0
            pitch = 48.0 + rnd()         # Safe lateral tilt
            roll = 26.0 + rnd()          # Safe lateral tilt
            vm_mean = 1.01 + rnd() * 0.02
            vm_std = 0.04
            sma = 1.10

        return {
            "hr": round(hr, 1),
            "spo2": 98 if scenario != "overexertion_high_hr" else 97,
            "ppg_signal": round(ppg, 1),
            "systolic_peak": round(sys_peak, 1),
            "diastolic_peak": round(dia_peak, 1),
            "pulse_area": round(pulse_area, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "vm_mean": round(vm_mean, 2),
            "vm_std": round(vm_std, 2),
            "sma": round(sma, 2),
            "gm_mean": 15.0 if scenario == "normal_resting" else 45.0
        }

    # =========================================================================
    # Physical Serial Port Integration
    # =========================================================================

    def detect_serial_ports(self) -> List[str]:
        if not HAS_SERIAL:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def connect_serial(self, port_name: str, baud: int = 115200) -> bool:
        if not HAS_SERIAL:
            return False
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.serial_conn = serial.Serial(port_name, baud, timeout=1.0)
            self.serial_port_name = port_name
            self.stop_simulator()  # Stop simulator in favor of real hardware
            self._serial_thread = threading.Thread(target=self._run_serial_reader, daemon=True)
            self._serial_thread.start()
            print(f"[Serial Bridge] Connected to ESP32 on {port_name} @ {baud}")
            return True
        except Exception as e:
            print(f"[Serial Bridge] Failed connecting to {port_name}: {e}")
            self.serial_conn = None
            return False

    def _run_serial_reader(self):
        import json
        while self.serial_conn and self.serial_conn.is_open:
            try:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('{') and line.endswith('}'):
                    data = json.loads(line)
                    self.ingest_raw(data)
            except Exception:
                time.sleep(0.1)
