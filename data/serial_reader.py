"""
Maternal Fitband — USB Serial Reader & Live Plotter
====================================================
Run this on your laptop while the ESP32 is plugged in via USB.

Usage:
  python data/serial_reader.py --port /dev/ttyUSB0 --baud 115200
  python data/serial_reader.py --port COM3 --baud 115200  (Windows)

Install dependencies first:
  pip install pyserial matplotlib pandas
"""

import serial
import json
import csv
import time
import argparse
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import os

# ── CLI Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='Maternal Fitband — USB Serial Reader & Live Plotter'
)
parser.add_argument('--port',  default='/dev/ttyUSB0',
                    help='Serial port (e.g. /dev/ttyUSB0 or COM3)')
parser.add_argument('--baud',  default=115200, type=int,
                    help='Baud rate (default: 115200)')
parser.add_argument('--csv',   default=os.path.join(os.path.dirname(__file__), 'readings.csv'),
                    help='CSV output file path')
args = parser.parse_args()

# ── Shared data buffers (thread-safe deques) ───────────────────────────────────
MAX_POINTS  = 60   # Show last 60 readings (~4 minutes at 4-second intervals)
timestamps  = deque(maxlen=MAX_POINTS)
hr_values   = deque(maxlen=MAX_POINTS)
spo2_values = deque(maxlen=MAX_POINTS)
vm_values   = deque(maxlen=MAX_POINTS)
latest      = {}   # Most recent reading for the status annotation

# ── CSV Logger ─────────────────────────────────────────────────────────────────
csv_file  = open(args.csv, 'a', newline='')
csvwriter = csv.writer(csv_file)
if os.path.getsize(args.csv) == 0:
    csvwriter.writerow([
        'wall_time', 'device_ts_s', 'hr_bpm', 'spo2_pct',
        'hr_valid', 'spo2_valid',
        'ax_g', 'ay_g', 'az_g', 'vm_g', 'activity'
    ])

print("=" * 65)
print("  Maternal Fitband -- Live Sensor Monitor")
print("=" * 65)
print(f"  Port   : {args.port}")
print(f"  Baud   : {args.baud}")
print(f"  CSV    : {args.csv}")
print("=" * 65)

# ── Serial Reader Thread ───────────────────────────────────────────────────────
def read_serial():
    """Runs in background thread. Reads JSON lines from ESP32 over USB Serial."""
    try:
        ser = serial.Serial(args.port, args.baud, timeout=2)
        print(f"\n[OK] Connected to {args.port} @ {args.baud} baud\n")
    except serial.SerialException as e:
        print(f"\n[ERR] Could not open port {args.port}: {e}")
        print("   -> Check that the ESP32 is plugged in and the port is correct.")
        print("   -> On Linux: sudo chmod 666 /dev/ttyUSB0\n")
        return

    while True:
        try:
            raw  = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='ignore').strip()

            if not line.startswith('{'):
                if line:
                    print(f"  [ESP32] {line}")
                continue

            data      = json.loads(line)
            wall_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            hr   = data.get('hr',   -1)
            spo2 = data.get('spo2', -1)
            vm   = data.get('vm',    0.0)
            act  = data.get('activity', 'unknown')
            hr_v = bool(data.get('hr_valid',   False))
            sp_v = bool(data.get('spo2_valid', False))
            ts   = data.get('ts', 0)
            ax_g = data.get('ax_g', 0.0)
            ay_g = data.get('ay_g', 0.0)
            az_g = data.get('az_g', 0.0)

            latest.update(data)
            latest['wall_time'] = wall_time

            # Terminal output
            status   = 'OK' if (hr_v and sp_v) else 'WARN'
            hr_str   = f"{hr:3d} bpm" if hr_v  else "--- bpm"
            spo2_str = f"{spo2:2d}%"  if sp_v  else "--%"
            alerts   = []
            if sp_v and spo2 < 94:  alerts.append("SPO2 CRITICAL")
            if hr_v and hr  > 100:  alerts.append("Tachycardia")
            if hr_v and hr  < 50:   alerts.append("Bradycardia")
            alert_str = "  !!! " + " | ".join(alerts) if alerts else ""
            print(
                f"  [{status}] [{wall_time}]  "
                f"HR: {hr_str}  |  SpO2: {spo2_str}  |  "
                f"VM: {vm:.3f}g  |  {act:<10s}{alert_str}"
            )

            # Update live chart buffers
            timestamps.append(ts)
            hr_values.append(hr     if hr_v else None)
            spo2_values.append(spo2 if sp_v else None)
            vm_values.append(vm)

            # Write CSV row
            csvwriter.writerow([
                wall_time, ts,
                hr   if hr_v else '',
                spo2 if sp_v else '',
                hr_v, sp_v, ax_g, ay_g, az_g, vm, act
            ])
            csv_file.flush()

        except json.JSONDecodeError:
            pass
        except UnicodeDecodeError:
            pass
        except Exception as e:
            print(f"  [reader error] {e}")


# ── Live Plot ──────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
fig.patch.set_facecolor('#0D1117')
fig.suptitle('Maternal Fitband -- Live Sensor Readings',
             fontsize=14, fontweight='bold', color='white', y=0.98)

PANEL_BG = '#161B22'
RED      = '#E63946'
BLUE     = '#457B9D'
TEAL     = '#2A9D8F'
ORANGE   = '#F4A261'
GRAY     = '#8B949E'

for ax in (ax1, ax2, ax3):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=GRAY)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(GRAY)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.yaxis.label.set_color(GRAY)
    ax.xaxis.label.set_color(GRAY)
    ax.title.set_color('white')


def animate(frame):
    t    = list(timestamps)
    hr   = list(hr_values)
    spo2 = list(spo2_values)
    vm   = list(vm_values)

    for ax in (ax1, ax2, ax3):
        ax.clear()
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, alpha=0.15, color=GRAY)

    # Heart Rate panel
    ax1.plot(t, hr, color=RED, linewidth=2, marker='o', markersize=4)
    ax1.axhline(y=100, color=ORANGE, linestyle='--', alpha=0.7, linewidth=1,
                label='Tachycardia (>100)')
    ax1.axhline(y=60,  color=GRAY,   linestyle=':',  alpha=0.5, linewidth=1,
                label='Normal lower (60)')
    ax1.set_ylabel('Heart Rate (bpm)', color=GRAY)
    ax1.set_ylim(40, 140)
    ax1.set_title('Heart Rate', color='white', fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               labelcolor=GRAY, framealpha=0.7)
    cur_hr = latest.get('hr', '--')
    hr_ok  = latest.get('hr_valid', False)
    ax1.text(0.01, 0.82, f"Now: {cur_hr} bpm" if hr_ok else "Now: no valid reading",
             transform=ax1.transAxes, color=RED, fontsize=10, fontweight='bold')

    # SpO2 panel
    ax2.plot(t, spo2, color=BLUE, linewidth=2, marker='s', markersize=4)
    ax2.axhline(y=94, color=RED,    linestyle='--', alpha=0.8, linewidth=1.5,
                label='Critical: SpO2 < 94%')
    ax2.axhline(y=95, color=ORANGE, linestyle=':',  alpha=0.6, linewidth=1,
                label='Caution: SpO2 < 95%')
    ax2.set_ylabel('SpO2 (%)', color=GRAY)
    ax2.set_ylim(85, 101)
    ax2.set_title('Blood Oxygen (SpO2)', color='white', fontsize=11)
    ax2.legend(loc='lower right', fontsize=8, facecolor=PANEL_BG,
               labelcolor=GRAY, framealpha=0.7)
    cur_sp = latest.get('spo2', '--')
    sp_ok  = latest.get('spo2_valid', False)
    ax2.text(0.01, 0.10, f"Now: {cur_sp}%" if sp_ok else "Now: no valid reading",
             transform=ax2.transAxes, color=BLUE, fontsize=10, fontweight='bold')

    # Activity / VM panel
    ax3.fill_between(t, vm, alpha=0.3, color=TEAL)
    ax3.plot(t, vm, color=TEAL, linewidth=1.5)
    ax3.axhline(y=0.05, color=GRAY,   linestyle=':', alpha=0.6, linewidth=1,
                label='Light (0.05g)')
    ax3.axhline(y=0.20, color=ORANGE, linestyle=':', alpha=0.6, linewidth=1,
                label='Moderate (0.20g)')
    ax3.axhline(y=0.60, color=RED,    linestyle=':', alpha=0.6, linewidth=1,
                label='Vigorous (0.60g)')
    ax3.set_ylabel('Vector Magnitude (g)', color=GRAY)
    ax3.set_xlabel('Device Time (seconds since boot)', color=GRAY)
    ax3.set_title('Accelerometer -- Activity Level', color='white', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8, facecolor=PANEL_BG,
               labelcolor=GRAY, framealpha=0.7)
    cur_act = latest.get('activity', '--')
    cur_vm  = latest.get('vm', 0.0)
    ax3.text(0.01, 0.82, f"Now: {cur_act}  ({cur_vm:.3f}g)",
             transform=ax3.transAxes, color=TEAL, fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    reader_thread = threading.Thread(target=read_serial, daemon=True)
    reader_thread.start()

    time.sleep(1.0)
    print("\n[INFO] Opening live chart window... (close the window to stop)\n")

    ani = animation.FuncAnimation(fig, animate, interval=1000, cache_frame_data=False)
    plt.show()

    csv_file.close()
    print(f"\n[DONE] Session complete. Data saved to: {args.csv}")
    print("       Run  python data/analyze.py  for a session summary.\n")
