#!/usr/bin/env python3
"""
Maternal Health Fitband — Cloud Web App & AI Companion Launcher
==============================================================
Launches the full-stack Tornado web server and Progressive Web App:
- Live Sensor Ingestion (ESP32 Serial, Wi-Fi HTTP Webhook, Simulator)
- Multi-Model Cloud ML Classification (Glucose, BP, HR, Sleep, Anomaly)
- Gemini AI Maternal Clinical Guidance & SBAR Progress Notes
- Mobile-friendly PWA interface for expectant mothers

Usage:
  python3 run_web_app.py
  PORT=8080 python3 run_web_app.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from web_app.server import run_server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    run_server(port=port)
