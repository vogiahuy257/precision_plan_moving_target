#!/bin/bash

# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash
source "$HOME/Precision-Landing/install/setup.bash"

PARAM_FILE="$(ros2 pkg prefix kalman_filter)/share/kalman_filter/cfg/params.yaml"

ros2 run kalman_filter kalman_filter_node --ros-args --params-file \
  --ros-args \
  --params-file "$PARAM_FILE"