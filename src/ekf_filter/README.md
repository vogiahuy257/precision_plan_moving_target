# ekf_filter

CTRA Extended Kalman Filter for the moving-target payload-drop project.

State order:

```text
[pN, pE, v, psi, a, omega]
```

Measurement:

```text
[pN, pE]
```

Outputs compatible with `target_drop` when `estimator.model: "ctra"`:

- `/EKF/target_pose_est_NED` (`PoseStamped`): filtered NED position; quaternion yaw = `psi`.
- `/EKF/target_velocity_est_NED` (`PoseStamped`): Cartesian NED velocity `(v cos psi, v sin psi, 0)`.
- `/EKF/target_motion` (`Float64MultiArray`): `[a, omega]`.
- `/EKF/target_covariance_NE` (`Float64MultiArray`): 36 row-major values for the 6x6 state covariance.
- `/EKF/process_noise` (`Float64MultiArray`, transient-local): `[q_acc, q_turn_rate]`; parameters remain owned by EKF.
- `/EKF/target_pose_NED` (`PoseStamped`): transformed raw target position.

The implementation uses the nonlinear CTRA transition, its EKF Jacobian, random-walk process noise on `a` and `omega`, position-only correction, Joseph covariance update, angle wrapping, and NIS measurement rejection.

The target is planar in the estimator. The output `z` value is the latest transformed target measurement, while output `vz` is zero.
