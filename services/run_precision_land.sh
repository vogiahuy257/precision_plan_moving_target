#!/bin/bash
# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash
source /home/pihuy/Precision-Landing/install/setup.bash
ros2 run precision_land precision_land --ros-args --params-file /home/pihuy/Precision-Landing/src/precision_land/params_land.yaml 
