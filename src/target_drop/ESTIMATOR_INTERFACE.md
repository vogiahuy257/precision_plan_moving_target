# Estimator interface for target_drop

`target_drop` does not own KF/EKF tuning parameters. Process-noise values are published by the active estimator through a transient-local topic and are used only to propagate target covariance to the payload impact horizon.

## KF / CV

- `/KF/target_pose_est_NED` — `geometry_msgs/PoseStamped`
- `/KF/target_velocity_est_NED` — `geometry_msgs/PoseStamped`
- `/KF/target_covariance_NE` — `std_msgs/Float64MultiArray`, 16 row-major values for `[pN,pE,vN,vE]`
- `/KF/process_noise` — `std_msgs/Float64MultiArray`, `[q_acc_N, q_acc_E]`

## EKF / CTRA

- `/EKF/target_pose_est_NED` — `geometry_msgs/PoseStamped`; quaternion yaw contains `psi`
- `/EKF/target_velocity_est_NED` — `geometry_msgs/PoseStamped`; Cartesian `[vN,vE,0]`
- `/EKF/target_motion` — `std_msgs/Float64MultiArray`, `[a, omega]`
- `/EKF/target_covariance_NE` — `std_msgs/Float64MultiArray`, 36 row-major values for `[pN,pE,v,psi,a,omega]`
- `/EKF/process_noise` — `std_msgs/Float64MultiArray`, `[q_acc,q_turn_rate]`

Use `cfg/estimator_kf.yaml` or `cfg/estimator_ekf.yaml` as an override on top of `cfg/params.yaml`.
