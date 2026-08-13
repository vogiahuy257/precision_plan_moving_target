# kalman_filter

[English](./README.en.md) | [Tiếng Việt](./README.vi.md)

A lightweight ROS 2 package for transforming target pose from the camera optical frame to the world frame, filtering it with a Kalman Filter, and publishing stable pose/velocity outputs for downstream control.

## Main idea

This node does 3 things:

- receives target pose from the vision node
- converts pose from optical frame to world frame
- estimates smooth position and velocity with a Kalman Filter

It also supports:

- hold mode when target is lost
- timeout handling
- CSV debug logging

---

## Design pattern used

This package follows a **Modular Architecture with Central Data Manager**.

### Why this design?

Instead of letting each class store random local data, all important runtime/config data is grouped inside a shared structure:

- `SystemData`

This makes the code easier to:

- read
- debug
- log
- extend later

### Main modules

- `KalmanFilterNode`  
  Main coordinator. Handles ROS topics, processing flow, Kalman update/predict, and publish logic.

- `FrameTransformer`  
  Handles coordinate transformation from camera optical frame to world frame.

- `DebugLogger`  
  Writes important runtime states into CSV for debugging and analysis.

- `DataStructs`  
  Central place for shared structs such as config, runtime state, vehicle state, target measurement, Kalman estimate, and debug row.

---

## Code architecture

```text
kalman_filter/
├── cfg/
│   └── params.yaml
├── docs/
│   └── images/
├── includes/
│   ├── DataStructs.hpp
│   ├── DebugLogger.hpp
│   ├── FrameTransformer.hpp
│   └── KalmanFilter.hpp
├── CMakeLists.txt
├── DebugLogger.cpp
├── FrameTransformer.cpp
├── KalmanFilter.cpp
├── package.xml
└── README.md
```

---

## Figures

### 1. Component Interaction Architecture

![Component Interaction Architecture](images/Component%20Interaction%20Architecture.png)

This figure shows how the main objects communicate:

- ROS topics send data into `KalmanFilterNode`
- `KalmanFilterNode` stores/updates `SystemData`
- `FrameTransformer` converts pose to world frame
- `OpenCV KalmanFilter` estimates filtered state
- `DebugLogger` writes CSV logs
- output topics publish filtered pose and velocity

---

### 2. Runtime Processing Flow

![Runtime Processing Flow](images/Runtime%20Processing%20Flow.png)

This figure explains the runtime logic:

- initialize parameters and modules
- wait for enough input data
- check timeout / hold conditions
- predict and correct Kalman state
- publish output
- write debug log

If target is lost or timeout happens, the node switches to hold mode.

---

### 3. Runtime Sequence of Interactions

![Runtime Sequence of Interactions](images/Runtime%20Sequence%20of%20Interactions.png)

This figure shows the real execution order:

- PX4 odometry and local position update UAV state
- target pose arrives from the vision node
- `FrameTransformer` converts pose to world frame
- Kalman is initialized or corrected
- timer loop runs predict + publish
- logger stores runtime data into CSV

---

## Topics

### Input

- `/Aruco/target_pose_FRD`
- `/Aruco/target_state`
- `/target_valid`
- `/fmu/out/vehicle_odometry`
- `/fmu/out/vehicle_local_position`

### Output

- `/KF/target_pose_NED`
- `/KF/target_pose_est_NED`
- `/KF/target_velocity_est_NED`
- `/KF/target_covariance_NE`
- `/KF/process_noise`

---

## Build

```bash
colcon build --packages-select kalman_filter --symlink-install
source install/setup.bash
```

---

## Run

```bash
ros2 run kalman_filter kalman_filter_node --ros-args --params-file src/kalman_filter/cfg/params.yaml
```

Or run from launch file in your full system.

---

## Debug log

Enable in `params.yaml`:

```yaml
debug: true
debug_csv_path: "kalman_logs/"
```

When enabled, the package automatically creates a CSV log file inside:

```text
kalman_logs/
```

---

## Notes

- If required PX4 or vision topics are missing, the node pauses processing safely.
- In hold mode, the node publishes the current world position to help the UAV hover instead of returning to the world origin.
- The log system is designed to avoid terminal spam by printing only when the processing state changes.