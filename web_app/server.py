"""
Maternal Fitband — Cloud Web Server & PWA Backend (Tornado)
===========================================================
High-performance asynchronous server providing:
  - Responsive Mobile-first PWA Web Application
  - Real-Time Server-Sent Events (SSE) telemetry stream
  - ML Condition Classification & Anomaly Alerting
  - Gemini AI Clinical Report Generation & Ob/Gyn SBAR Export
  - Universal Sensor Ingestion (ESP32 Serial, BLE, Wi-Fi Webhook, Simulator)
"""

import os
import sys
import json
import asyncio
import warnings

# Suppress unpickling version warnings in logs
warnings.filterwarnings('ignore', category=UserWarning)

import tornado.ioloop
import tornado.web
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_app.ml_engine import ml_engine
from web_app.gemini_engine import gemini_engine
from web_app.sensor_stream import SensorStreamManager

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
stream_manager = SensorStreamManager(ml_engine, gemini_engine)


class BaseApiHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def options(self, *args, **kwargs):
        self.set_status(204)
        self.finish()

    def get_json_body(self) -> Dict[str, Any]:
        try:
            return json.loads(self.request.body.decode('utf-8'))
        except Exception:
            return {}


class MainAppHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "text/html; charset=utf-8")
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            with open(index_file, "r", encoding="utf-8") as f:
                self.write(f.read())
        else:
            self.write("<h1>Maternal Fitband Web App is initializing...</h1>")

    def head(self):
        self.get()


class ManifestHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/manifest+json; charset=utf-8")
        manifest_file = os.path.join(STATIC_DIR, "manifest.json")
        if os.path.exists(manifest_file):
            with open(manifest_file, "r", encoding="utf-8") as f:
                self.write(f.read())
        else:
            self.set_status(404)

    def head(self):
        self.get()


class ServiceWorkerHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/javascript; charset=utf-8")
        sw_file = os.path.join(STATIC_DIR, "service-worker.js")
        if os.path.exists(sw_file):
            with open(sw_file, "r", encoding="utf-8") as f:
                self.write(f.read())
        else:
            self.set_status(404)

    def head(self):
        self.get()


class FaviconHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "image/x-icon")
        self.set_status(204)

    def head(self):
        self.get()


class StatusHandler(BaseApiHandler):
    def get(self):
        self.write({
            "status": "online",
            "simulator_active": stream_manager.simulator_active,
            "current_scenario": stream_manager.current_scenario,
            "serial_connected": stream_manager.serial_conn is not None,
            "serial_port": stream_manager.serial_port_name,
            "available_ports": stream_manager.detect_serial_ports(),
            "gemini_api_key_configured": bool(gemini_engine.api_key),
            "gemini_model": gemini_engine.model,
            "maternal_profile": stream_manager.maternal_profile,
            "active_symptoms": stream_manager.active_symptoms,
            "models_loaded": len(ml_engine.models)
        })


class TelemetryIngestHandler(BaseApiHandler):
    """
    Direct Webhook for ESP32 (Wi-Fi HTTP POST) or external bridges.
    """
    def post(self):
        body = self.get_json_body()
        if not body:
            self.set_status(400)
            self.write({"error": "Empty or invalid JSON payload"})
            return
        classified = stream_manager.ingest_raw(body)
        self.write({
            "success": True,
            "timestamp": classified.get("timestamp"),
            "vitals": classified.get("vitals"),
            "clinical_status": classified.get("clinical_status")
        })


class StreamSSEHandler(BaseApiHandler):
    """
    Server-Sent Events (SSE) streaming endpoint for live telemetry updates.
    """
    async def get(self):
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Cache-Control', 'no-cache')
        self.set_header('Connection', 'keep-alive')
        self.set_header('X-Accel-Buffering', 'no')

        queue = asyncio.Queue()

        def listener(data):
            try:
                queue.put_nowait(data)
            except Exception:
                pass

        stream_manager.subscribe(listener)

        # Send initial snapshot immediately
        if stream_manager.latest_classified:
            initial_msg = f"data: {json.dumps(stream_manager.latest_classified)}\n\n"
            self.write(initial_msg)
            await self.flush()

        try:
            while True:
                data = await queue.get()
                message = f"data: {json.dumps(data)}\n\n"
                self.write(message)
                await self.flush()
        except (tornado.iostream.StreamClosedError, asyncio.CancelledError):
            pass
        finally:
            stream_manager.unsubscribe(listener)


class HistoryHandler(BaseApiHandler):
    def get(self):
        self.write({
            "history": stream_manager.telemetry_history,
            "latest": stream_manager.latest_classified
        })


class SymptomsHandler(BaseApiHandler):
    def get(self):
        from web_app.clinical_guidelines import RED_FLAG_SYMPTOM_DEFINITIONS
        self.write({
            "active_symptoms": stream_manager.active_symptoms,
            "definitions": RED_FLAG_SYMPTOM_DEFINITIONS
        })

    def post(self):
        body = self.get_json_body()
        symptoms = body.get("symptoms", [])
        stream_manager.update_symptoms(symptoms)
        self.write({
            "success": True,
            "active_symptoms": stream_manager.active_symptoms,
            "latest": stream_manager.latest_classified
        })


class GenerateReportHandler(BaseApiHandler):
    def post(self):
        try:
            body = self.get_json_body()
            custom_key = body.get("gemini_api_key")
            if custom_key:
                gemini_engine.set_api_key(custom_key)

            classified = stream_manager.latest_classified
            profile = stream_manager.maternal_profile

            if not classified:
                self.set_status(400)
                self.write({"success": False, "error": "No sensor readings available to evaluate"})
                return

            report = gemini_engine.generate_maternal_report(
                classified,
                profile,
                active_symptoms=stream_manager.active_symptoms
            )
            self.write({
                "success": True,
                "report": report
            })
        except Exception as e:
            print(f"[ERR] Report generation failed: {e}")
            self.set_status(500)
            self.write({"success": False, "error": str(e)})


class SimulatorControlHandler(BaseApiHandler):
    def post(self):
        body = self.get_json_body()
        action = body.get("action")
        scenario = body.get("scenario")

        if action == "start":
            stream_manager.start_simulator()
        elif action == "stop":
            stream_manager.stop_simulator()

        if scenario:
            stream_manager.set_scenario(scenario)

        self.write({
            "success": True,
            "simulator_active": stream_manager.simulator_active,
            "current_scenario": stream_manager.current_scenario
        })


class ProfileUpdateHandler(BaseApiHandler):
    def post(self):
        body = self.get_json_body()
        if body:
            stream_manager.update_profile(body)
        self.write({
            "success": True,
            "profile": stream_manager.maternal_profile
        })


class ConfigHandler(BaseApiHandler):
    def post(self):
        body = self.get_json_body()
        key = body.get("gemini_api_key")
        model = body.get("gemini_model")
        if key is not None:
            gemini_engine.set_api_key(key)
        if model:
            gemini_engine.model = model
        self.write({
            "success": True,
            "gemini_api_key_configured": bool(gemini_engine.api_key),
            "gemini_model": gemini_engine.model
        })


class SerialConnectHandler(BaseApiHandler):
    def post(self):
        body = self.get_json_body()
        port = body.get("port")
        baud = int(body.get("baud", 115200))
        if not port:
            self.set_status(400)
            self.write({"error": "Port name is required"})
            return
        success = stream_manager.connect_serial(port, baud)
        self.write({
            "success": success,
            "connected": stream_manager.serial_conn is not None,
            "port": stream_manager.serial_port_name
        })


def make_app():
    return tornado.web.Application([
        (r"/", MainAppHandler),
        (r"/manifest\.json", ManifestHandler),
        (r"/service-worker\.js", ServiceWorkerHandler),
        (r"/favicon\.ico", FaviconHandler),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": STATIC_DIR}),
        (r"/api/status", StatusHandler),
        (r"/api/telemetry", TelemetryIngestHandler),
        (r"/api/stream", StreamSSEHandler),
        (r"/api/history", HistoryHandler),
        (r"/api/report/generate", GenerateReportHandler),
        (r"/api/symptoms", SymptomsHandler),
        (r"/api/simulator", SimulatorControlHandler),
        (r"/api/profile", ProfileUpdateHandler),
        (r"/api/config", ConfigHandler),
        (r"/api/serial/connect", SerialConnectHandler),
    ], debug=False)


def get_local_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def run_server(port: int = 8080, host: str = "0.0.0.0"):
    app = make_app()
    app.listen(port, address=host)
    local_ip = get_local_ip()
    print("=" * 75)
    print("  🤰 MATERNAL HEALTH FITBAND — CLOUD WEB APP & AI SERVER STARTED  ")
    print("=" * 75)
    print(f"  • Open on your Laptop:   http://localhost:{port}")
    print(f"  • Open on your Mobile:   http://{local_ip}:{port}")
    print(f"  • ESP32 Webhook URL:     http://{local_ip}:{port}/api/telemetry")
    print(f"  • Live SSE Data Stream:  http://{local_ip}:{port}/api/stream")
    print("=" * 75)
    print("  👉 NOTE: Ensure your mobile phone is connected to the same Wi-Fi!")
    print("=" * 75)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    run_server(port=port)
