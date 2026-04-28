# kalman_filter

<div align="center">

ROS 2 package for transforming target pose from the camera optical frame to the world frame, filtering it with a Kalman Filter, and publishing stable pose and velocity outputs.

[English Documentation](./docs/README.en.md) • [Tài liệu Tiếng Việt](./docs/README.vi.md)

</div>

---

## Documentation

This README is only a quick entry page.  
For full details, please read:

- [English documentation](./docs/README.en.md)
- [Tài liệu tiếng Việt](./docs/README.vi.md)

---

## Quick Start

### Build

```bash
colcon build --packages-select kalman_filter --symlink-install
source install/setup.bash
```

### Run

```bash
ros2 run kalman_filter kalman_filter_node --ros-args --params-file src/kalman_filter/cfg/params.yaml
```

---

## Project Structure

```text
kalman_filter/
├── cfg/
├── docs/
│   ├── README.en.md
│   ├── README.vi.md
│   └── images/
├── includes/
├── CMakeLists.txt
├── DebugLogger.cpp
├── FrameTransformer.cpp
├── KalmanFilter.cpp
└── package.xml
```

---

## Author

Prepared by **Vo Gia Huy**