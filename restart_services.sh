#!/bin/bash

set -u
set -o pipefail

echo "=============================================="
echo "[INFO] Restart Precision Landing Services"
echo "=============================================="

SERVICES=(
  "microxrce-agent.service"
  "imx219_camera.service"
  "aruco-tracker.service"
  "kf.service"
  "ekf.service"
  "target_drop.service"
)

echo ""
echo "[INFO] Stopping services..."

# Stop theo thứ tự ngược:
# target_drop -> EKF/KF -> ArUco -> Camera -> XRCE Agent
for ((i=${#SERVICES[@]}-1; i>=0; i--)); do
    svc="${SERVICES[$i]}"

    echo "  → Stopping ${svc}"

    if sudo systemctl stop "${svc}"; then
        echo "    ✔ ${svc} stopped"
    else
        echo "    ⚠ Failed to stop ${svc}"
    fi
done

echo ""
echo "[INFO] Reloading systemd..."
sudo systemctl daemon-reload

echo ""
echo "[INFO] Starting services..."

FAILED_SERVICES=()

# Start lại theo đúng thứ tự
for svc in "${SERVICES[@]}"; do
    echo "  → Starting ${svc}"

    if sudo systemctl start "${svc}"; then
        sleep 1

        if systemctl is-active --quiet "${svc}"; then
            echo "    ✔ ${svc} is active"
        else
            echo "    ❌ ${svc} is NOT active"
            FAILED_SERVICES+=("${svc}")
        fi
    else
        echo "    ❌ Failed to start ${svc}"
        FAILED_SERVICES+=("${svc}")
    fi
done

echo ""
echo "=============================================="

if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    echo "✅ ALL SERVICES RESTARTED SUCCESSFULLY"
else
    echo "❌ SOME SERVICES FAILED:"
    for svc in "${FAILED_SERVICES[@]}"; do
        echo "   - ${svc}"
    done

    echo ""
    echo "Check failed service with:"
    echo "  journalctl -u <service-name> -n 100 --no-pager"
fi

echo "=============================================="

echo ""
echo "[STATUS] Current service status:"
echo ""

for svc in "${SERVICES[@]}"; do
    STATUS=$(systemctl is-active "${svc}" 2>/dev/null || true)
    printf "  %-30s %s\n" "${svc}" "${STATUS}"
done
