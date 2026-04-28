#!/usr/bin/env python3
"""
HelipadTrackerNode không dùng cv_bridge.

Lý do:
    Môi trường hiện tại bị lỗi tương thích NumPy 2.x với cv_bridge.
    File này tự chuyển sensor_msgs/Image <-> OpenCV image bằng numpy buffer.

Luồng:
    Image + CameraInfo
        -> YOLO detect helipad
        -> solvePnP từ 4 góc bbox
        -> publish raw pose camera/FRD ra /Helipad/target_pose_FRD
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from HelipadDetector import HelipadDetector
from Helipad_3D import HelipadPoseEstimator


class HelipadTrackerNode(Node):
    def __init__(self) -> None:
        super().__init__("helipad_tracker")

        self.declare_parameter("debug", False)
        self.declare_parameter("model_path", "model/detect.pt")
        self.declare_parameter("conf_threshold", 0.6)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("marker_size_w", 0.5)
        self.declare_parameter("marker_size_h", 0.5)

        self.declare_parameter(
            "image_topic",
            "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/camera_info",
        )

        self.declare_parameter("output_pose_topic", "/Helipad/target_pose_FRD")
        self.declare_parameter("output_state_topic", "/Helipad/target_state")
        self.declare_parameter("output_image_topic", "/Helipad/image_proc")
        self.declare_parameter("debug_dt_topic", "/debug_dt/helipad")
        self.declare_parameter("marker_timeout_s", 5.0)
        self.declare_parameter("frame_id", "camera_frd")
        self.declare_parameter("show_image", False)

        self.debug = bool(self.get_parameter("debug").value)
        self.model_path = self._resolve_model_path(str(self.get_parameter("model_path").value))
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.marker_size_w = float(self.get_parameter("marker_size_w").value)
        self.marker_size_h = float(self.get_parameter("marker_size_h").value)

        image_topic = str(self.get_parameter("image_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        output_pose_topic = str(self.get_parameter("output_pose_topic").value)
        output_state_topic = str(self.get_parameter("output_state_topic").value)
        output_image_topic = str(self.get_parameter("output_image_topic").value)
        debug_dt_topic = str(self.get_parameter("debug_dt_topic").value)

        self.marker_timeout_s = float(self.get_parameter("marker_timeout_s").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.show_image = bool(self.get_parameter("show_image").value)

        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.detector: Optional[HelipadDetector] = None
        self.pose_estimator: Optional[HelipadPoseEstimator] = None

        self.last_detect_time = None
        self.reset_published = False
        self.frame_count = 0
        self.det_count = 0

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        output_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        debug_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.info_callback,
            sensor_qos,
        )
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            sensor_qos,
        )

        self.pose_pub = self.create_publisher(PoseStamped, output_pose_topic, output_qos)
        self.state_pub = self.create_publisher(String, output_state_topic, output_qos)
        self.image_pub = self.create_publisher(Image, output_image_topic, output_qos)
        self.debug_pub = self.create_publisher(String, debug_dt_topic, debug_qos)

        self.get_logger().info("HelipadTrackerNode started")
        self.get_logger().info(f"model_path: {self.model_path}")
        self.get_logger().info(f"pose topic: {output_pose_topic}")
        self.get_logger().info(f"state topic: {output_state_topic}")
        self.get_logger().info("Waiting for camera_info...")

    def _resolve_model_path(self, model_path: str) -> str:
        path = Path(model_path)

        if path.is_absolute():
            return str(path)

        try:
            package_share = Path(get_package_share_directory("helipad_tracker"))
            candidate = package_share / path
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass

        source_candidate = Path(__file__).resolve().parents[1] / path
        if source_candidate.exists():
            return str(source_candidate)

        return model_path

    def info_callback(self, msg: CameraInfo) -> None:
        camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if camera_matrix[0, 0] == 0.0 or camera_matrix[1, 1] == 0.0:
            return

        self.camera_matrix = camera_matrix
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

        self.detector = HelipadDetector(
            weights_path=self.model_path,
            conf_threshold=self.conf_threshold,
            imgsz=self.imgsz,
        )

        self.pose_estimator = HelipadPoseEstimator(
            obj_w_m=self.marker_size_w,
            obj_h_m=self.marker_size_h,
            camera_matrix=self.camera_matrix,
        )

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        self.get_logger().info(
            f"Camera ready: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}"
        )

        self.destroy_subscription(self.info_sub)

    def image_callback(self, msg: Image) -> None:
        if self.camera_matrix is None or self.detector is None or self.pose_estimator is None:
            return

        start_time = self.get_clock().now()
        self.frame_count += 1

        try:
            frame = self.ros_image_to_bgr(msg)
        except Exception as exc:
            self.get_logger().warn(f"ros_image_to_bgr failed: {exc}")
            return

        display = frame.copy()
        found = False
        confidence = 0.0

        detection = self.detector.detect(frame)

        if detection is not None:
            confidence = float(detection.conf)

            corners_2d = np.array(
                [
                    [[detection.x1, detection.y1]],
                    [[detection.x2, detection.y1]],
                    [[detection.x2, detection.y2]],
                    [[detection.x1, detection.y2]],
                ],
                dtype=np.float64,
            )

            rvec_deg, tvec_m = self.pose_estimator.solve_pnp(corners_2d)

            if rvec_deg is not None and tvec_m is not None:
                found = True
                self.det_count += 1
                self.last_detect_time = self.get_clock().now()
                self.reset_published = False

                self.publish_pose(msg, tvec_m)
                self.draw_detection(display, detection, rvec_deg, tvec_m)

        if not found:
            self.draw_lost(display)
            self.handle_target_loss()

        self.publish_debug_image(msg, display)
        self.publish_timing(start_time, msg.header.stamp, found, confidence)

        if self.show_image:
            cv2.imshow("Helipad Tracker", display)
            cv2.waitKey(1)

    def ros_image_to_bgr(self, msg: Image) -> np.ndarray:
        height = int(msg.height)
        width = int(msg.width)
        encoding = msg.encoding.lower()

        if height <= 0 or width <= 0:
            raise ValueError("Image height/width khong hop le")

        data = np.frombuffer(msg.data, dtype=np.uint8)

        if encoding in ("bgr8", "rgb8", "8uc3"):
            channels = 3
            row_bytes = int(msg.step)
            expected_min_step = width * channels

            if row_bytes < expected_min_step:
                raise ValueError(
                    f"Image step khong hop le: step={row_bytes}, expected>={expected_min_step}"
                )

            image_2d = data.reshape(height, row_bytes)
            image = image_2d[:, :expected_min_step].reshape(height, width, channels)

            if encoding == "rgb8":
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            return image.copy()

        if encoding in ("bgra8", "rgba8", "8uc4"):
            channels = 4
            row_bytes = int(msg.step)
            expected_min_step = width * channels

            if row_bytes < expected_min_step:
                raise ValueError(
                    f"Image step khong hop le: step={row_bytes}, expected>={expected_min_step}"
                )

            image_2d = data.reshape(height, row_bytes)
            image = image_2d[:, :expected_min_step].reshape(height, width, channels)

            if encoding == "rgba8":
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if encoding in ("mono8", "8uc1"):
            row_bytes = int(msg.step)
            expected_min_step = width

            if row_bytes < expected_min_step:
                raise ValueError(
                    f"Image step khong hop le: step={row_bytes}, expected>={expected_min_step}"
                )

            image_2d = data.reshape(height, row_bytes)
            gray = image_2d[:, :expected_min_step].reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        raise ValueError(f"Encoding chua ho tro: {msg.encoding}")

    def bgr_to_ros_image(self, frame: np.ndarray, source_msg: Image) -> Image:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Debug frame phai la anh BGR 3 kenh")

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        out_msg = Image()
        out_msg.header = source_msg.header
        out_msg.height = int(frame.shape[0])
        out_msg.width = int(frame.shape[1])
        out_msg.encoding = "bgr8"
        out_msg.is_bigendian = 0
        out_msg.step = int(frame.shape[1] * 3)
        out_msg.data = frame.tobytes()
        return out_msg

    def publish_pose(self, image_msg: Image, tvec_m: np.ndarray) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = image_msg.header.stamp
        pose_msg.header.frame_id = self.frame_id

        pose_msg.pose.position.x = float(tvec_m[0])
        pose_msg.pose.position.y = float(tvec_m[1])
        pose_msg.pose.position.z = float(tvec_m[2])

        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)

    def handle_target_loss(self) -> None:
        if self.last_detect_time is None:
            return

        now = self.get_clock().now()
        lost_duration = (now - self.last_detect_time).nanoseconds * 1e-9

        if lost_duration >= self.marker_timeout_s and not self.reset_published:
            reset_msg = String()
            reset_msg.data = "RESET"
            self.state_pub.publish(reset_msg)
            self.reset_published = True
            self.get_logger().warn("Helipad lost too long, published RESET")

    def publish_debug_image(self, image_msg: Image, display: np.ndarray) -> None:
        try:
            out_msg = self.bgr_to_ros_image(display, image_msg)
            self.image_pub.publish(out_msg)
        except Exception as exc:
            self.get_logger().warn(f"publish debug image failed: {exc}")

    def publish_timing(self, start_time, image_stamp, found: bool, confidence: float) -> None:
        end_time = self.get_clock().now()
        processing_dt = (end_time - start_time).nanoseconds * 1e-9

        image_time_sec = float(image_stamp.sec) + float(image_stamp.nanosec) * 1e-9
        now_sec = float(end_time.nanoseconds) * 1e-9
        image_to_cb_dt = max(0.0, now_sec - image_time_sec)

        msg = String()
        msg.data = (
            "{"
            f"\"node\":\"helipad\","
            f"\"found\":{str(found).lower()},"
            f"\"confidence\":{confidence:.4f},"
            f"\"frame_count\":{self.frame_count},"
            f"\"det_count\":{self.det_count},"
            f"\"image_to_cb_dt\":{image_to_cb_dt:.6f},"
            f"\"processing_dt\":{processing_dt:.6f}"
            "}"
        )
        self.debug_pub.publish(msg)

    def draw_detection(self, display: np.ndarray, detection, rvec_deg: np.ndarray, tvec_m: np.ndarray) -> None:
        x1, y1 = int(detection.x1), int(detection.y1)
        x2, y2 = int(detection.x2), int(detection.y2)

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display, (detection.center_x, detection.center_y), 5, (0, 255, 255), -1)

        cv2.putText(
            display,
            f"Helipad {detection.conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            f"X:{tvec_m[0]:+.2f} Y:{tvec_m[1]:+.2f} Z:{tvec_m[2]:+.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            display,
            f"frame={self.frame_count} det={self.det_count}",
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        try:
            rvec_rad = np.deg2rad(rvec_deg).flatten().astype(np.float64)
            tvec_draw = tvec_m.flatten().astype(np.float64)
            axis_length = min(self.marker_size_w, self.marker_size_h) * 0.5

            cv2.drawFrameAxes(
                display,
                self.camera_matrix,
                np.zeros(5),
                rvec_rad,
                tvec_draw,
                axis_length,
                thickness=2,
            )
        except Exception as exc:
            if self.debug:
                self.get_logger().warn(f"drawFrameAxes failed: {exc}")

    def draw_lost(self, display: np.ndarray) -> None:
        cv2.putText(
            display,
            "NO HELIPAD",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            display,
            f"frame={self.frame_count} det={self.det_count}",
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HelipadTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_image:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
