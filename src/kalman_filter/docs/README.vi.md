# kalman_filter

[English](./README.en.md) | [Tiếng Việt](./README.vi.md)

Một gói ROS 2 gọn nhẹ dùng để biến đổi pose mục tiêu từ hệ tọa độ optical của camera sang hệ tọa độ world, lọc bằng Kalman Filter, và publish pose/vận tốc ổn định cho bộ điều khiển phía sau.

## Ý tưởng chính

Node này thực hiện 3 việc:

- nhận pose mục tiêu từ node thị giác
- chuyển pose từ hệ optical sang hệ world
- ước lượng vị trí và vận tốc mượt hơn bằng Kalman Filter

Ngoài ra, node còn hỗ trợ:

- chế độ giữ vị trí khi mất mục tiêu
- xử lý timeout
- ghi log CSV để debug

---

## Design pattern sử dụng

Package này áp dụng **Modular Architecture with Central Data Manager**.

### Vì sao dùng thiết kế này?

Thay vì để mỗi class tự giữ dữ liệu rời rạc, toàn bộ dữ liệu runtime/config quan trọng được gom vào một cấu trúc dùng chung:

- `SystemData`

Cách này giúp code:

- dễ đọc hơn
- dễ debug hơn
- dễ ghi log hơn
- dễ mở rộng hơn về sau

### Các module chính

- `KalmanFilterNode`  
  Thành phần điều phối trung tâm. Xử lý topic ROS, luồng chạy chính, Kalman update/predict và logic publish.

- `FrameTransformer`  
  Xử lý biến đổi tọa độ từ camera optical frame sang world frame.

- `DebugLogger`  
  Ghi các trạng thái runtime quan trọng ra file CSV để debug và phân tích.

- `DataStructs.hpp`  
  Nơi chứa các struct dùng chung như config, runtime state, vehicle state, target measurement, Kalman estimate và debug row.

---

## Kiến trúc mã nguồn

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

## Hình minh họa

### 1. Kiến trúc tương tác giữa các thành phần

![Component Interaction Architecture](images/Component%20Interaction%20Architecture.png)

Hình này mô tả cách các đối tượng chính giao tiếp với nhau:

- các topic ROS gửi dữ liệu vào `KalmanFilterNode`
- `KalmanFilterNode` lưu và cập nhật `SystemData`
- `FrameTransformer` chuyển pose sang hệ world
- `OpenCV KalmanFilter` ước lượng trạng thái đã lọc
- `DebugLogger` ghi log CSV
- các topic output publish pose và vận tốc đã xử lý

---

### 2. Luồng xử lý runtime

![Runtime Processing Flow](images/Runtime%20Processing%20Flow.png)

Hình này mô tả logic runtime:

- khởi tạo parameter và các module
- chờ đủ dữ liệu đầu vào
- kiểm tra timeout / hold conditions
- predict và correct Kalman state
- publish output
- ghi log debug

Khi mất mục tiêu hoặc xảy ra timeout, node sẽ chuyển sang chế độ hold.

---

### 3. Trình tự tương tác khi chạy

![Runtime Sequence of Interactions](images/Runtime%20Sequence%20of%20Interactions.png)

Hình này mô tả thứ tự thực thi thực tế:

- PX4 odometry và local position cập nhật trạng thái UAV
- pose mục tiêu được gửi từ node thị giác
- `FrameTransformer` chuyển pose sang hệ world
- Kalman được khởi tạo hoặc cập nhật
- timer loop chạy predict + publish
- logger ghi dữ liệu runtime ra file CSV

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

## Chạy

```bash
ros2 run kalman_filter kalman_filter_node --ros-args --params-file src/kalman_filter/cfg/params.yaml
```

Hoặc chạy thông qua launch file trong toàn bộ hệ thống.

---

## Debug log

Bật trong `params.yaml`:

```yaml
debug: true
debug_csv_path: "kalman_logs/"
```

Khi bật, package sẽ tự động tạo file log CSV trong thư mục:

```text
kalman_logs/
```

---

## Ghi chú

- Nếu thiếu topic đầu vào từ PX4 hoặc vision, node sẽ tự tạm dừng xử lý một cách an toàn.
- Ở chế độ hold, node sẽ publish vị trí world hiện tại để giúp UAV hover tại chỗ thay vì quay về gốc tọa độ world.
- Hệ thống log được thiết kế để tránh spam terminal, chỉ in khi trạng thái xử lý thay đổi.