"""
Maternal Fitband — Live Hardware Serial Monitor & Analytics
===========================================================
Connects to the ESP32 via USB Serial, parses real-time telemetry,
logs data to readings.csv, and checks for clinical alerts.

Usage:
  python3 live_dashboard.py                  (auto-detects port)
  python3 live_dashboard.py --port /dev/tty...
  python3 live_dashboard.py --simulate       (test mode without hardware)
"""

import json
import csv
import time
import argparse
import os
import sys
import math
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    from maternal_ai_wrapper import MaternalAIWrapper, MaternalProfile, MaternalVitals
    HAS_AI_WRAPPER = True
except ImportError:
    HAS_AI_WRAPPER = False

def find_esp32_port():
    if not HAS_SERIAL:
        return None
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()
        if any(chip in desc or chip in dev for chip in ['cp210', 'ch340', 'wch', 'ftdi', 'usb serial', 'slab', 'esp32']):
            return p.device
    return ports[0].device

def process_payload(data, writer=None, csv_file=None):
    now = datetime.now().strftime('%H:%M:%S')
    hr = data.get('hr', 0)
    spo2 = data.get('spo2', 0)
    gluc = data.get('glucose', 0.0)
    act = data.get('activity', 'sedentary')
    sleep = data.get('sleep', 'Awake')
    supine = data.get('supine', False)
    vm = data.get('vm', 1.0)
    hr_v = data.get('hr_valid', True)
    sp_v = data.get('spo2_valid', True)

    if writer and csv_file:
        writer.writerow([
            now, data.get('ts', 0), hr, spo2, gluc,
            act, sleep, supine, vm,
            data.get('ax_g', 0), data.get('ay_g', 0), data.get('az_g', 0)
        ])
        csv_file.flush()

    alerts = []
    if hr_v and hr > 100: alerts.append("TACHYCARDIA (>100)")
    if hr_v and hr < 50:  alerts.append("BRADYCARDIA (<50)")
    if sp_v and spo2 < 94: alerts.append("LOW SPO2 (<94%)")
    if supine:            alerts.append("SUPINE (ROLL TO LEFT)")
    if act == "OVEREXERTION": alerts.append("OVEREXERTION ALERT")

    alert_str = " | ⚠️  " + ", ".join(alerts) if alerts else ""
    hr_disp = f"{hr:3d} bpm" if hr_v else "--- bpm"
    sp_disp = f"{spo2:2d} %" if sp_v else "-- %"

    print(
        f"[{now}] HR: {hr_disp} | SpO2: {sp_disp} | Gluc: {gluc:5.1f} mg/dL | "
        f"Act: {act:<12s} | Sleep: {sleep:<15s}{alert_str}"
    )

def main():
    parser = argparse.ArgumentParser(description='Maternal Fitband Live Serial Dashboard')
    parser.add_argument('--port', default=None, help='Serial port (e.g. /dev/ttyUSB0 or COM3)')
    parser.add_argument('--baud', default=115200, type=int, help='Baud rate')
    parser.add_argument('--csv',  default='readings.csv', help='CSV output file')
    parser.add_argument('--simulate', action='store_true', help='Run in live simulation mode without hardware')
    parser.add_argument('--ai', action='store_true', help='Enable Clinical AI Wrapper recommendations on session exit')
    parser.add_argument('--export-report', default='session_clinical_report.md', help='Output Markdown clinical report path')
    args = parser.parse_args()

    ai_wrapper = MaternalAIWrapper() if (args.ai and HAS_AI_WRAPPER) else None
    latest_data = {}

    print("=" * 75)
    print("  MATERNAL FITBAND: LIVE SERIAL MONITOR & CLINICAL TELEMETRY")
    print("=" * 75)

    if args.simulate:
        print("  MODE: Simulation Mode (Testing without physical hardware)")
        print(f"  Logging To: {args.csv}")
        print("=" * 75 + "\n")
    else:
        if not HAS_SERIAL:
            print("[!] 'pyserial' library is not installed.")
            print("    Run: pip install pyserial")
            print("\n    To test the dashboard immediately without hardware, run:")
            print("      python3 live_dashboard.py --simulate")
            sys.exit(0)

        port = args.port or find_esp32_port()
        if not port:
            print("[-] No serial port detected!")
            print("    Please connect your ESP32 via USB and check:")
            print("      • Mac: ls /dev/tty.usb* or /dev/tty.wch*")
            print("      • Windows: Device Manager -> Ports (COM & LPT)")
            print("\n    To test without hardware, run:")
            print("      python3 live_dashboard.py --simulate")
            sys.exit(0)
        print(f"  Target Port : {port}")
        print(f"  Baud Rate   : {args.baud}")
        print(f"  Logging To  : {args.csv}")
        print("=" * 75)

    # Open CSV for persistent logging
    csv_exists = os.path.exists(args.csv)
    csv_file = open(args.csv, 'a', newline='')
    writer = csv.writer(csv_file)

    if not csv_exists or os.path.getsize(args.csv) == 0:
        writer.writerow([
            'timestamp', 'device_ts_s', 'hr_bpm', 'spo2_pct', 'glucose_mg_dl',
            'activity', 'sleep_stage', 'supine_alert', 'vm_g', 'ax_g', 'ay_g', 'az_g'
        ])
        csv_file.flush()

    def trigger_ai_summary(last_reading):
        if not (ai_wrapper and last_reading):
            return
        print("\n" + "=" * 75)
        print("  GENERATING POST-SESSION CLINICAL AI EVALUATION...")
        print("=" * 75)
        vitals = MaternalVitals(
            glucose_mg_dl=float(last_reading.get("glucose", 100.0)),
            heart_rate_bpm=int(last_reading.get("hr", 75)),
            spo2_pct=int(last_reading.get("spo2", 98)),
            activity_state=last_reading.get("activity", "sedentary"),
            sleep_stage=last_reading.get("sleep", "Awake"),
            supine_alert=bool(last_reading.get("supine", False)),
            vector_magnitude_g=float(last_reading.get("vm", 1.0))
        )
        profile = MaternalProfile()
        res = ai_wrapper.process(profile, vitals)
        ai_wrapper.render_terminal_dashboard(res)
        if args.export_report:
            ai_wrapper.export_markdown_report(res, args.export_report)

    if args.simulate:
        try:
            t = 0
            while True:
                t += 1
                hr = int(74 + 6 * math.sin(t / 5))
                spo2 = int(98 - (1 if t % 20 == 0 else 0))
                gluc = 108.5 + 4.0 * math.cos(t / 8)
                act = "walking" if (t % 15 > 5) else "sedentary"
                sleep = "Deep Sleep" if t > 30 else "Light Sleep"
                supine = (t % 25 == 0)

                sim_data = {
                    "ts": t, "hr": hr, "spo2": spo2, "hr_valid": True, "spo2_valid": True,
                    "ax_g": 0.05, "ay_g": 0.02, "az_g": 0.98, "vm": 0.985,
                    "activity": act, "glucose": round(gluc, 1),
                    "sleep": sleep, "supine": supine
                }
                latest_data = sim_data
                process_payload(sim_data, writer, csv_file)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[+] Simulation stopped by user.")
            trigger_ai_summary(latest_data)
        finally:
            csv_file.close()
        sys.exit(0)

    # Hardware Serial connection loop
    try:
        ser = serial.Serial(port, args.baud, timeout=2)
        print(f"\n[+] Connected to ESP32 on {port}!\n")
    except Exception as e:
        print(f"\n[-] Could not open {port}: {e}")
        sys.exit(1)

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='ignore').strip()

            if line.startswith("#"):
                print(f"  [BOOT] {line}")
                continue

            if not line.startswith("{"):
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            latest_data = data
            process_payload(data, writer, csv_file)

    except KeyboardInterrupt:
        print("\n[+] Monitoring stopped by user. CSV saved.")
        trigger_ai_summary(latest_data)
    finally:
        ser.close()
        csv_file.close()

if __name__ == '__main__':
    main()
