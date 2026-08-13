#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /home/pihuy/precision_plan_moving_target/install/setup.bash

exec ros2 run kalman_filter kalman_filter_node \
    --ros-args \
    --params-file /home/pihuy/precision_plan_moving_target/src/kalman_filter/cfg/params.yaml
