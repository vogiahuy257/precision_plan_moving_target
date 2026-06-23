#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/pihuy/precision_plan_moving_target/install/setup.bash

ros2 run imx219_camera_cpp imx219_camera_node --ros-args \
  --params-file ~/precision_plan_moving_target/src/imx219_camera_cpp/config/imx219_camera.yaml