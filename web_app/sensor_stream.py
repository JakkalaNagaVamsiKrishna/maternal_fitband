"""
Maternal FitBand — Universal Sensor Stream & Telemetry Bridge
=============================================================
Provides:
  1. Real-time 100 Hz USB Serial Telemetry Ingestion from ESP32
     with rolling PPG & IMU windowed feature extraction
  2. Guideline-backed multi-scenario physiological simulator
  3. Red-flag symptoms state management
  4. Real-time SSE dispatch for the mother-facing dashboard
"""

import time
import math
import random
import threading
import traceback
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

from .signal_processing import PPGSignalProcessor, MotionWindowProcessor
from .clinical_guidelines import evaluate_guidelines, RED_FLAG_SYMPTOM_DEFINITIONS, PROTOTYPE_DISCLAIMER


class SensorStreamManager:
    def __init__(self, ml_engine, gemini_engine):
        self.ml_engine = ml_engine
        self.gemini_engine = gemini_engine

        # Signal Processing Buffers
        self.ppg_processor = PPGSignalProcessor(sample_rate_hz=100.0, window_sec=4.0)
        self.motion_processor = MotionWindowProcessor(sample_rate_hz=50.0, window_sec=6.0)

        # Active state
        self.latest_raw: Dict[str, Any] = {}
        self.latest_classified: Dict[str, Any] = {}
        self.telemetry_history: List[Dict[str, Any]] = []
        self.max_history = 60
        self.active_symptoms: List[str] = []

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

        # Start initial baseline & simulator
        self._init_baseline()
        self.start_simulator()

    def _init_baseline(self):
        raw = {
            "hr": 74, "spo2": 98, "ppg_signal": 512.0,
            "systolic_peak": 521.0, "diastolic_peak": 507.0, "pulse_area": 390.0,
            "pulse_amp": 14.0, "pulse_mean": 512.0, "pulse_std": 5.0,
            "ppi_mean": 810.0, "ppi_std": 22.0,
            "Acc_X_mean": 0.12, "Acc_X_std": 0.04,
            "Acc_Y_mean": 0.18, "Acc_Y_std": 0.05,
            "Acc_Z_mean": 0.97, "Acc_Z_std": 0.03,
            "VM_mean": 1.01, "VM_std": 0.04, "VM_max": 1.08,
            "SMA": 1.15, "Pitch": 45.0, "Roll": 25.0,
            "raw_sensors": {
                "ir": 51200, "red": 48640,
                "ax": 0.12, "ay": 0.18, "az": 0.97,
                "t_ms": 0, "packet_count": 0
            },
            "timestamp": datetime.now().isoformat()
        }
        self.ingest_raw(raw)

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        if callback in self.subscribers:
            self.subscribers.remove(callback)

    def update_profile(self, new_profile: Dict[str, Any]):
        self.maternal_profile.update(new_profile)
        if self.latest_raw:
            self.ingest_raw(self.latest_raw)

    def update_symptoms(self, symptoms: List[str]):
        """Sets the active list of mother-reported red-flag symptoms."""
        self.active_symptoms = [s for s in symptoms if s in RED_FLAG_SYMPTOM_DEFINITIONS]
        if self.latest_raw:
            self.ingest_raw(self.latest_raw)

    def ingest_raw(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives raw or windowed sensor packets, executes ML classification,
        evaluates deterministic clinical guidelines, and notifies subscribers.
        """
        raw_data["timestamp"] = datetime.now().strftime("%H:%M:%S")
        self.latest_raw = raw_data

        # 1. ML classification
        classified = self.ml_engine.classify(raw_data, self.maternal_profile)
        classified["timestamp"] = raw_data["timestamp"]
        classified["raw_inputs"] = raw_data
        classified["raw_sensors"] = raw_data.get("raw_sensors", {})
        classified["profile"] = self.maternal_profile
        classified["data_source"] = "Physical ESP32 (USB)" if (self.serial_conn and self.serial_conn.is_open) else ("Simulator (" + self.current_scenario + ")")

        # 2. Deterministic Clinical Guidelines Evaluation
        guidelines = evaluate_guidelines(
            vitals=classified["vitals"],
            kinematics=classified["kinematics"],
            profile=self.maternal_profile,
            active_symptoms=self.active_symptoms
        )
        classified["guidelines"] = guidelines
        classified["active_symptoms"] = self.active_symptoms

        self.latest_classified = classified

        # 3. Buffer history for continuous charts
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

        # 4. Notify active listeners (SSE)
        for cb in list(self.subscribers):
            try:
                cb(classified)
            except Exception:
                pass

        return classified

    # =========================================================================
    # Simulator Controls (Backup for Rehearsal & Testing)
    # =========================================================================

    def set_scenario(self, scenario: str):
        valid = [
            "normal_resting",
            "postprandial_hyperglycemia",
            "supine_sleep_alert",
            "overexertion_high_hr",
            "gestational_hypertension",
            "severe_hypertension"
        ]
        if scenario in valid:
            self.current_scenario = scenario
            self._sim_step = 0
            # Force immediate packet
            packet = self._generate_simulated_packet(scenario, self._sim_step)
            self.ingest_raw(packet)

    def start_simulator(self):
        if self._sim_thread and self._sim_thread.is_alive():
            return
        self.simulator_active = True
        self._sim_thread = threading.Thread(target=self._run_simulator, daemon=True)
        self._sim_thread.start()

    def stop_simulator(self):
        self.simulator_active = False

    def _run_simulator(self):
        while self.simulator_active:
            # If physical USB serial is active and streaming, suspend simulator ticks
            if self.serial_conn and self.serial_conn.is_open:
                time.sleep(1.0)
                continue

            packet = self._generate_simulated_packet(self.current_scenario, self._sim_step)
            self.ingest_raw(packet)
            self._sim_step += 1
            time.sleep(1.0)

    def _generate_simulated_packet(self, scenario: str, step: int) -> Dict[str, Any]:
        """
        Generates physiologically faithful dynamic waveforms & windowed motion metrics.
        """
        t = step * 0.2
        rnd = lambda: random.uniform(-0.5, 0.5)

        if scenario == "postprandial_hyperglycemia":
            # Elevated glucose post-meal (~148 mg/dL)
            hr = 82 + 2 * math.sin(t / 4) + rnd()
            ppg = 528.0 + 3 * math.sin(t / 2) + rnd()
            sys_peak = ppg + 14.0
            dia_peak = ppg - 6.0
            pulse_area = 430.0 + rnd() * 5
            pitch = 14.0 + rnd()
            roll = 18.0 + rnd()
            vm_mean = 1.02 + rnd() * 0.03
            vm_std = 0.06
            sma = 1.25

        elif scenario == "supine_sleep_alert":
            # Mother lying flat on back (pitch and roll < 15 deg) in bed -> Vena cava compression!
            hr = 68 + 2 * math.sin(t / 6) + rnd()
            ppg = 510.0 + rnd()
            sys_peak = ppg + 9.0
            dia_peak = ppg - 4.0
            pulse_area = 380.0
            pitch = 6.0 + rnd() * 0.3    # Flat back!
            roll = 8.0 + rnd() * 0.3    # Flat back!
            vm_mean = 0.99 + rnd() * 0.01
            vm_std = 0.02                # Deep resting minimal movement
            sma = 0.95

        elif scenario == "overexertion_high_hr":
            # Brisk walking / jerky movement
            hr = 138 + 4 * math.sin(t / 3) + rnd() * 2
            ppg = 535.0 + rnd() * 4
            sys_peak = ppg + 18.0
            dia_peak = ppg - 8.0
            pulse_area = 460.0
            pitch = 18.0 + rnd() * 3
            roll = 22.0 + rnd() * 3
            vm_mean = 2.45 + rnd() * 0.4
            vm_std = 1.15                # Strenuous movement
            sma = 3.10

        elif scenario == "gestational_hypertension":
            # SBP 148 / DBP 94 spike (ACOG Stage 1)
            hr = 98 + 3 * math.sin(t / 4) + rnd()
            ppg = 524.0 + rnd() * 2
            sys_peak = ppg + 22.0
            dia_peak = ppg - 11.0
            pulse_area = 450.0
            pitch = 14.0 + rnd()
            roll = 16.0 + rnd()
            vm_mean = 1.03 + rnd() * 0.04
            vm_std = 0.09
            sma = 1.30

        elif scenario == "severe_hypertension":
            # SBP 164 / DBP 112 spike (ACOG Severe Urgent Care)
            hr = 108 + 3 * math.sin(t / 4) + rnd()
            ppg = 532.0 + rnd() * 2
            sys_peak = ppg + 28.0
            dia_peak = ppg - 14.0
            pulse_area = 480.0
            pitch = 12.0 + rnd()
            roll = 14.0 + rnd()
            vm_mean = 1.04 + rnd() * 0.03
            vm_std = 0.08
            sma = 1.35

        else:  # "normal_resting"
            # Healthy, left-lateral safe posture (Pitch 45, Roll 25)
            hr = 74 + 2 * math.sin(t / 5) + rnd()
            ppg = 512.0 + rnd()
            sys_peak = ppg + 10.0
            dia_peak = ppg - 4.5
            pulse_area = 390.0
            pitch = 46.0 + rnd()
            roll = 24.0 + rnd()
            vm_mean = 1.01 + rnd() * 0.02
            vm_std = 0.04
            sma = 1.10

        pulse_amp = max(2.0, sys_peak - dia_peak)

        return {
            "hr": round(hr, 1),
            "spo2": 98 if scenario != "overexertion_high_hr" else 96,
            "ppg_signal": round(ppg, 1),
            "systolic_peak": round(sys_peak, 1),
            "diastolic_peak": round(dia_peak, 1),
            "pulse_amp": round(pulse_amp, 1),
            "pulse_area": round(pulse_area, 1),
            "pulse_mean": round(ppg, 1),
            "pulse_std": round(pulse_amp * 0.35, 2),
            "pulse_auc": round(pulse_area, 1),
            "ppi_mean": round(60000.0 / hr, 1),
            "ppi_std": 22.0,
            "Acc_X_mean": round(math.sin(math.radians(pitch)) * 0.3, 3),
            "Acc_X_std": round(vm_std * 0.6, 3),
            "Acc_Y_mean": round(math.sin(math.radians(roll)) * 0.3, 3),
            "Acc_Y_std": round(vm_std * 0.7, 3),
            "Acc_Z_mean": round(math.cos(math.radians(max(pitch, roll))), 3),
            "Acc_Z_std": round(vm_std * 0.5, 3),
            "VM_mean": round(vm_mean, 2),
            "VM_std": round(vm_std, 2),
            "VM_max": round(vm_mean + vm_std * 2.0, 2),
            "SMA": round(sma, 2),
            "Pitch": round(pitch, 1),
            "Roll": round(roll, 1),
            "gm_mean": 15.0 if scenario == "normal_resting" else 45.0,
            "raw_sensors": {
                "ir": int(ppg * 100),
                "red": int(ppg * 95),
                "ax": round(math.sin(math.radians(pitch)) * 0.3, 3),
                "ay": round(math.sin(math.radians(roll)) * 0.3, 3),
                "az": round(math.cos(math.radians(max(pitch, roll))), 3),
                "t_ms": step * 1000,
                "packet_count": step + 1
            }
        }

    # =========================================================================
    # Physical Serial Port Integration (100 Hz USB JSON Streamer)
    # =========================================================================

    def detect_serial_ports(self) -> List[str]:
        if not HAS_SERIAL:
            return []
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]

    def connect_serial(self, port_name: str, baud: int = 115200) -> bool:
        if not HAS_SERIAL:
            print("[Serial Bridge] pyserial not installed — cannot connect.")
            return False
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                time.sleep(0.3)
            self.serial_conn = serial.Serial(port_name, baud, timeout=1.0)
            self.serial_port_name = port_name
            # Flush any stale data in the serial buffer
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            # Stop the simulator so only real data flows
            self.stop_simulator()
            # Reset signal processing buffers for fresh calibration
            self.ppg_processor = PPGSignalProcessor(sample_rate_hz=100.0, window_sec=4.0)
            self.motion_processor = MotionWindowProcessor(sample_rate_hz=50.0, window_sec=6.0)
            self._serial_thread = threading.Thread(target=self._run_serial_reader, daemon=True)
            self._serial_thread.start()
            print(f"[Serial Bridge] Connected to ESP32 on {port_name} @ {baud}")
            return True
        except Exception as e:
            print(f"[Serial Bridge] Failed connecting to {port_name}: {e}")
            import traceback
            traceback.print_exc()
            self.serial_conn = None
            return False

    def _run_serial_reader(self):
        import json
        last_emit_time = time.time()
        print("[Serial Reader] Thread started, waiting for data...")
        line_count = 0
        error_count = 0
        while self.serial_conn and self.serial_conn.is_open:
            try:
                raw_bytes = self.serial_conn.readline()
                if not raw_bytes:
                    continue
                line = raw_bytes.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                # Skip comment/debug lines from firmware (e.g. "# [OK] ...")
                if line.startswith('#'):
                    print(f"[Serial Reader] Firmware: {line}")
                    continue

                if not (line.startswith('{') and line.endswith('}')):
                    # Partial or corrupted line — skip silently
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    error_count += 1
                    if error_count <= 5:
                        print(f"[Serial Reader] Malformed JSON (#{error_count}): {line[:80]}")
                    continue

                line_count += 1
                if line_count == 1:
                    print(f"[Serial Reader] First valid packet received: {line[:120]}")

                # Check if incoming packet is high-speed 100 Hz raw sample:
                if "ir" in data and ("ax" in data or "ax_g" in data):
                    ir = float(data.get("ir", 512))
                    red = float(data.get("red", ir))
                    t_ms = float(data.get("t", 0))
                    ax = float(data.get("ax", data.get("ax_g", 0.0)))
                    ay = float(data.get("ay", data.get("ay_g", 0.0)))
                    az = float(data.get("az", data.get("az_g", 1.0)))

                    self.ppg_processor.add_sample(ir, red, t_ms)
                    self.motion_processor.add_sample(ax, ay, az)

                    now = time.time()
                    # Throttle feature extraction & SSE dispatch to ~2 Hz (every 0.5s)
                    if now - last_emit_time >= 0.5:
                        last_emit_time = now
                        ppg_feats = self.ppg_processor.extract_features()
                        motion_feats = self.motion_processor.extract_features()
                        merged = {
                            **ppg_feats,
                            **motion_feats,
                            "ts": data.get("t", int(now * 1000)),
                            "raw_sensors": {
                                "ir": int(ir),
                                "red": int(red),
                                "ax": round(ax, 3),
                                "ay": round(ay, 3),
                                "az": round(az, 3),
                                "t_ms": int(t_ms),
                                "packet_count": line_count
                            }
                        }
                        if line_count <= 10 or line_count % 200 == 0:
                            print(f"[Serial Reader] Emitting merged features (sample #{line_count}): HR={ppg_feats.get('heart_rate')}, PPG={ppg_feats.get('ppg_signal')}")
                        self.ingest_raw(merged)
                else:
                    # Direct pre-computed JSON
                    data["raw_sensors"] = {
                        "ir": int(data.get("ir", data.get("ppg_signal", 512))),
                        "red": int(data.get("red", 0)),
                        "ax": round(float(data.get("ax", data.get("Acc_X_mean", 0.0))), 3),
                        "ay": round(float(data.get("ay", data.get("Acc_Y_mean", 0.0))), 3),
                        "az": round(float(data.get("az", data.get("Acc_Z_mean", 1.0))), 3),
                        "t_ms": int(data.get("t", time.time() * 1000)),
                        "packet_count": line_count
                    }
                    self.ingest_raw(data)
            except serial.SerialException as e:
                print(f"[Serial Reader] Serial port error: {e}")
                break
            except Exception as e:
                if error_count < 10:
                    print(f"[Serial Reader] Unexpected error: {e}")
                    traceback.print_exc()
                error_count += 1
                time.sleep(0.01)
        print(f"[Serial Reader] Thread exiting. Processed {line_count} packets, {error_count} errors.")
