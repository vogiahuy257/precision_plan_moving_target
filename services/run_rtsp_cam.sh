#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/pihuy/Precision-Landing/install/setup.bash

RTSP_URL="rtsp://192.168.144.25:8554/main.264"
CAM_INFO="file:///home/pihuy/.ros/camera_info/camera.yaml"

ros2 run rtsp_camera rtsp_camera_node \
  --ros-args \
  -p rtsp_url:=${RTSP_URL} \
  -p image_topic:=/camera/image_raw \
  -p camera_info_topic:=/camera/camera_info \
  -p camera_info_url:=${CAM_INFO}
