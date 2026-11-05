"""
Maternal Fitband — Wireless Bluetooth Low Energy (BLE) Receiver
===============================================================
Receives live telemetry wirelessly from the ESP32 Fitband over BLE.

Prerequisites:
  pip install bleak

Usage:
  python3 wireless_ble_receiver.py
"""

import asyncio
import json
import sys

SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
DEVICE_NAME  = "MaternalFitband"

try:
    from bleak import BleakScanner, BleakClient
except ImportError:
    print("[!] 'bleak' library is not installed.")
    print("    To run wireless BLE reception on your computer, run:")
    print("      pip install bleak")
    print("\nAlternatively, you can connect directly using:")
    print("  • 'nRF Connect' app (iOS / Android)")
    print("  • 'LightBlue' app (iOS / Android)")
    sys.exit(0)

def notification_handler(sender, data):
    payload_str = data.decode('utf-8', errors='ignore')
    try:
        telemetry = json.loads(payload_str)
        print("\n" + "=" * 65)
        print("  WIRELESS FITBAND TELEMETRY RECEIVED")
        print("=" * 65)
        print(f"  ❤️  Heart Rate       : {telemetry.get('hr', '--')} bpm")
        print(f"  🫁  Blood Oxygen SpO2: {telemetry.get('spo2', '--')} %")
        gluc_val = telemetry.get('glucose', telemetry.get('gluc', '--'))
        print(f"  🩸  Est. Glucose     : {gluc_val} mg/dL")
        print(f"  🏃  Activity State   : {telemetry.get('act', '--')}")
        print(f"  🌙  Sleep Stage      : {telemetry.get('sleep', '--')}")
        if telemetry.get('supine'):
            print("  ⚠️  POSTURE ALERT     : Flat on back (Supine)! Turn to left side.")
        else:
            print("  ✅  POSTURE OK       : Safe lateral position detected.")
        print("=" * 65)
    except json.JSONDecodeError:
        print(f"Raw data: {payload_str}")

async def main():
    print(f"Scanning for '{DEVICE_NAME}' over Bluetooth Low Energy...")
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and DEVICE_NAME.lower() in d.name.lower(),
        timeout=10.0
    )

    if not device:
        print(f"[!] Could not find '{DEVICE_NAME}'.")
        print("    Ensure your ESP32 is powered and the BLE firmware is running.")
        return

    print(f"[+] Found {device.name} [{device.address}]. Connecting...")
    async with BleakClient(device) as client:
        print("[+] Connected wirelessly! Listening for live maternal telemetry...")
        await client.start_notify(TX_CHAR_UUID, notification_handler)

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping wireless reception...")
            await client.stop_notify(TX_CHAR_UUID)

if __name__ == '__main__':
    asyncio.run(main())
