#!/bin/bash
# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash
source /home/pihuy/precision_plan_moving_target/install/setup.bash

ros2 run precision_land precision_land --ros-args \
 --params-file ~/precision_plan_moving_target/src/precision_land/cfg/params.yaml
