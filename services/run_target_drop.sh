#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /home/pihuy/precision_plan_moving_target/install/setup.bash

TARGET_ESTIMATOR="${TARGET_ESTIMATOR:-kf}"

BASE_PARAMS="/home/pihuy/precision_plan_moving_target/src/target_drop/cfg/params.yaml"

case "${TARGET_ESTIMATOR}" in
    kf)
        ESTIMATOR_PARAMS="/home/pihuy/precision_plan_moving_target/src/target_drop/cfg/estimator_kf.yaml"
        ;;
    ekf)
        ESTIMATOR_PARAMS="/home/pihuy/precision_plan_moving_target/src/target_drop/cfg/estimator_ekf.yaml"
        ;;
    *)
        echo "[target_drop] Invalid TARGET_ESTIMATOR='${TARGET_ESTIMATOR}'. Use 'kf' or 'ekf'." >&2
        exit 1
        ;;
esac

echo "[target_drop] Estimator: ${TARGET_ESTIMATOR}"

exec ros2 run target_drop target_drop \
    --ros-args \
    --params-file "${BASE_PARAMS}" \
    --params-file "${ESTIMATOR_PARAMS}"
