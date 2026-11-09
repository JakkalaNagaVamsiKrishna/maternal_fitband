#!/usr/bin/env bash
# ==============================================================================
# Maternal Fitband — 1-Click Live Demonstration Launcher
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "  🌸 MATERNAL HEALTH FITBAND — LAUNCHING LIVE DEMONSTRATION 🌸"
echo "======================================================================"

# 1. Start Web App Backend
echo "[1/3] Starting Cloud Web App & AI Companion Engine..."
.venv/bin/python run_web_app.py > /tmp/maternal_web_app.log 2>&1 &
WEB_PID=$!

# Trap Ctrl+C to cleanly stop both services
cleanup() {
    trap - SIGINT SIGTERM EXIT
    echo ""
    echo "Stopping demo processes..."
    kill $WEB_PID 2>/dev/null || true
    kill $TUNNEL_PID 2>/dev/null || true
    echo "Demo stopped cleanly. Have a great presentation!"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Give server 2 seconds to bind port
sleep 2

# 2. Check cloudflared binary
CLOUDFLARED_BIN="/home/vamsi/.local/bin/cloudflared"
if [ ! -f "$CLOUDFLARED_BIN" ]; then
    CLOUDFLARED_BIN=$(which cloudflared || true)
fi

if [ -z "$CLOUDFLARED_BIN" ]; then
    echo "Warning: cloudflared not found in PATH. You can view locally at http://localhost:8080"
    wait $WEB_PID
fi

# 3. Start Cloudflare Tunnel for Instant HTTPS Domain
echo "[2/3] Launching Cloudflare HTTPS Domain Tunnel..."
rm -f /tmp/maternal_tunnel.log
"$CLOUDFLARED_BIN" tunnel --url http://localhost:8080 > /tmp/maternal_tunnel.log 2>&1 &
TUNNEL_PID=$!

echo "[3/3] Acquiring Public Demonstration URL..."
PUBLIC_URL=""
for i in {1..15}; do
    if grep -q "trycloudflare.com" /tmp/maternal_tunnel.log 2>/dev/null; then
        PUBLIC_URL=$(grep -o 'https://[-a-zA-Z0-9.]*\.trycloudflare\.com' /tmp/maternal_tunnel.log | head -n 1)
        break
    fi
    sleep 1
done

echo ""
echo "======================================================================"
echo "  ✅ LIVE DEMONSTRATION READY!"
echo "======================================================================"
if [ -n "$PUBLIC_URL" ]; then
    echo "  🌐 PUBLIC HTTPS URL  : $PUBLIC_URL"
    echo "     (Open this on your mobile phone or share with your audience)"
else
    echo "  ⚠️  Could not auto-detect trycloudflare URL yet. Check /tmp/maternal_tunnel.log"
fi
echo "  💻 LOCAL LAPTOP URL  : http://localhost:8080"
echo "======================================================================"
echo "  👉 Press Ctrl + C at any time to stop the demo."
echo "======================================================================"

# Keep alive until user hits Ctrl+C
wait $WEB_PID
