#!/bin/bash

set -e

echo "=============================================="
echo "[INFO] Setup Precision Landing Autostart Stack"
echo "=============================================="

USER_NAME="pihuy"
BASE_DIR="/home/${USER_NAME}/Precision-Landing"
SERVICE_DIR="${BASE_DIR}/services"
SYSTEMD_DIR="/etc/systemd/system"

# === Service cũ cần dọn sạch ===
LEGACY_SERVICES=(
  # "precision-land-stack.service"
  # "rtsp-relay-node-cam2.service"
  # "uav-autostart.service"
  "microxrce-agent.service"
  "rtsp-camera.service"
  "aruco-tracker.service"
  "target-pose-fusion.service"
  "gimbal-controller.service"
  "precision-land.service"
  "eth0-force-10mbps.service"
  # "taggpio_monitor.service"
)

SERVICES=(
  "microxrce-agent.service"
  "rtsp-camera.service"
  # "aruco-tracker.service"
  # "target-pose-fusion.service"
  "gimbal-controller.service"
  # "precision-land.service"
  "eth0-force-10mbps.service"
  # "taggpio_monitor.service"
)

echo "[INFO] Checking service scripts permissions..."
chmod +x ${SERVICE_DIR}/*.sh

echo "[INFO] Stopping & disabling legacy services if exist..."
for svc in "${LEGACY_SERVICES[@]}"; do
  if systemctl list-unit-files | grep -q "^${svc}"; then
    echo "  ⚠ Removing legacy service: ${svc}"
    sudo systemctl stop ${svc} 2>/dev/null || true
    sudo systemctl disable ${svc} 2>/dev/null || true
  fi
done

echo "[INFO] Removing existing services if any..."
for svc in "${SERVICES[@]}"; do
  if [ -f "${SYSTEMD_DIR}/${svc}" ] || [ -L "${SYSTEMD_DIR}/${svc}" ]; then
    echo "  ⚠ Removing existing ${svc}"
    sudo systemctl stop ${svc} 2>/dev/null || true
    sudo systemctl disable ${svc} 2>/dev/null || true
    sudo rm -f ${SYSTEMD_DIR}/${svc}
  fi
done

echo "[INFO] Copying systemd service files..."
for svc in "${SERVICES[@]}"; do
  if [ -f "${SERVICE_DIR}/${svc}" ]; then
    sudo cp ${SERVICE_DIR}/${svc} ${SYSTEMD_DIR}/
    echo "  ✔ Copied ${svc}"
  else
    echo "  ❌ Missing ${SERVICE_DIR}/${svc}"
    exit 1
  fi
done

echo "[INFO] Reloading systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo "[INFO] Enabling services..."
for svc in "${SERVICES[@]}"; do
  sudo systemctl enable ${svc}
  echo "  ✔ Enabled ${svc}"
done

echo "[INFO] Starting services in correct order..."

for svc in "${SERVICES[@]}"; do
  sudo systemctl start "${svc}"
  systemctl is-active --quiet "${svc}" \
    && echo "  ✔ ${svc} is active" \
    || echo "  ❌ ${svc} failed"
done


echo "=============================================="
echo "✅ Precision Landing stack AUTOSTART READY"
echo "Reboot machine to verify auto start"
echo "=============================================="
