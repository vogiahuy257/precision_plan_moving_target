#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/pihuy/precision_plan_moving_target/install/setup.bash

ros2 run kalman_filter kalman_filter_node --ros-args --params-file ~/precision_plan_moving_target/src/kalman_filter/cfg/params.yaml