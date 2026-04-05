set -e

SERVICES=(
  "rtsp-camera.service"
  "aruco-tracker.service"
  "target-pose-fusion.service"
  "gimbal-controller.service"
  "precision-land.service"
  "eth0-force-10mbps.service"
  "taggpio_monitor.service"
)

for s in "${SERVICES[@]}"; do
  sudo systemctl stop "$s"
  sudo systemctl disable "$s"
done
