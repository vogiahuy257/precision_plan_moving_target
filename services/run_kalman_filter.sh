#!/bin/bash


# ROS 2 Jazzy
source /opt/ros/jazzy/setup.bash
source /home/pihuy/Precision-Landing/install/setup.bash

ros2 run target_pose_fusion target_pose_fusion_node --ros-args --params-file /home/pihuy/Precision-Landing/src/target_pose_fusion/params_kalman.yaml