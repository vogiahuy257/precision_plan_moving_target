#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String


class TimingCollectorNode(Node):
    """
    Node gom timing từ Aruco, Kalman, PrecisionLand theo key timestamp chung.
    Mục tiêu:
    - đo thời gian node nhận
    - đo thời gian node xử lý
    - đo thời gian node gửi
    - đo dt giữa các node
    - đo tổng end-to-end nếu upstream có publish đủ field
    """

    def __init__(self):
        super().__init__("timing_collector")

        self.declare_parameter(
            "output_csv",
            os.path.expanduser("~/precision_landing/tracktor-beam/pipeline_timing_nodes.csv")
        )
        self.declare_parameter("flush_period_sec", 0.5)
        self.declare_parameter("stale_timeout_sec", 5.0)

        self.output_csv = str(self.get_parameter("output_csv").value)
        self.flush_period_sec = float(self.get_parameter("flush_period_sec").value)
        self.stale_timeout_sec = float(self.get_parameter("stale_timeout_sec").value)

        self.records = defaultdict(dict)

        self.header_written = (
            os.path.exists(self.output_csv) and
            os.path.getsize(self.output_csv) > 0
        )

        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.create_subscription(String, "/debug_dt/aruco", self.aruco_callback, qos)
        self.create_subscription(String, "/debug_dt/kalman", self.kalman_callback, qos)
        self.create_subscription(String, "/debug_dt/precision_land", self.precision_land_callback, qos)

        self.create_timer(self.flush_period_sec, self.flush_to_csv)

        self.get_logger().info("Timing collector started")
        self.get_logger().info(f"Output CSV: {self.output_csv}")

    def parse_json(self, msg: String):
        """
        Parse JSON từ std_msgs/String.

        Input:
            msg: std_msgs.msg.String

        Logic:
            Chuyển chuỗi JSON từ topic debug_dt thành dict Python.

        Output:
            dict hoặc None nếu parse lỗi.
        """
        try:
            return json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"JSON parse error: {exc} | raw={msg.data}")
            return None

    def get_float(self, data: dict, key: str, default: float = -1.0) -> float:
        """
        Lấy giá trị float từ dict.

        Input:
            data: dict
            key: str
            default: float

        Logic:
            Ép kiểu float an toàn.

        Output:
            float
        """
        try:
            return float(data.get(key, default))
        except Exception:
            return default

    def get_first_valid_stamp(self, data: dict, keys: list[str]) -> float:
        """
        Lấy stamp đầu tiên hợp lệ từ danh sách key.

        Input:
            data: dict
            keys: list[str]

        Logic:
            Duyệt lần lượt từng key, key nào có giá trị >= 0 thì dùng.

        Output:
            float
        """
        for key in keys:
            value = self.get_float(data, key, -1.0)
            if value >= 0.0:
                return value
        return -1.0

    def make_key(self, stamp: float) -> str:
        """
        Tạo key gom dữ liệu theo timestamp.

        Input:
            stamp: float

        Logic:
            Làm tròn tới 6 số lẻ để tránh lệch nhỏ khi stringify float.

        Output:
            str
        """
        return f"{stamp:.6f}"

    def update_last_seen(self, record: dict):
        """
        Cập nhật thời điểm collector nhận dữ liệu cuối cùng của record.

        Input:
            record: dict

        Logic:
            Lưu thời gian local để drop record stale nếu thiếu dữ liệu.

        Output:
            Không có.
        """
        record["collector_last_seen"] = self.get_clock().now().nanoseconds / 1e9

    def has_complete_record(self, record: dict) -> bool:
        """
        Kiểm tra record đã đủ dữ liệu để ghi CSV chưa.

        Input:
            record: dict

        Logic:
            Chỉ ghi khi đã có đủ:
            - aruco pub
            - kalman pub
            - precision land ctrl start
            - precision land cmd pub

        Output:
            bool
        """
        required = [
            "record_key_stamp",
            "aruco_pub_now",
            "kalman_pub_now",
            "pl_ctrl_start_now",
            "pl_cmd_pub_now",
        ]
        return all(field in record for field in required)

    def is_stale_record(self, record: dict, now_sec: float) -> bool:
        """
        Kiểm tra record stale.

        Input:
            record: dict
            now_sec: float

        Logic:
            Nếu quá stale_timeout_sec kể từ lần update cuối thì coi là stale.

        Output:
            bool
        """
        last_seen = float(record.get("collector_last_seen", 0.0))
        return (now_sec - last_seen) > self.stale_timeout_sec

    def aruco_callback(self, msg: String):
        """
        Nhận timing từ node Aruco.

        Input:
            msg: std_msgs.msg.String

        Logic:
            Parse JSON và lưu toàn bộ timing của Aruco.

        Output:
            Không có.
        """
        data = self.parse_json(msg)
        if data is None:
            return

        image_stamp = self.get_float(data, "image_stamp", -1.0)
        if image_stamp < 0.0:
            return

        key = self.make_key(image_stamp)
        record = self.records[key]

        record["record_key_stamp"] = image_stamp
        record["image_stamp"] = image_stamp

        record["aruco_rx_now"] = self.get_float(data, "rx_now")
        record["aruco_proc_start"] = self.get_float(data, "proc_start")
        record["aruco_proc_end"] = self.get_float(data, "proc_end")
        record["aruco_pub_now"] = self.get_float(data, "pub_now")

        record["aruco_rx_wait_dt"] = self.get_float(data, "rx_wait_dt")
        record["aruco_queue_before_proc_dt"] = self.get_float(data, "queue_before_proc_dt")
        record["aruco_processing_dt"] = self.get_float(data, "processing_dt")
        record["aruco_send_dt"] = self.get_float(data, "send_dt")
        record["aruco_total_node_dt"] = self.get_float(data, "total_node_dt")
        record["aruco_found"] = int(data.get("found", 0))

        self.update_last_seen(record)

    def kalman_callback(self, msg: String):
        """
        Nhận timing từ node Kalman.

        Input:
            msg: std_msgs.msg.String

        Logic:
            Chỉ lưu stage PUB vì đây là output cuối của Kalman.

        Output:
            Không có.
        """
        data = self.parse_json(msg)
        if data is None:
            return

        image_stamp = self.get_float(data, "image_stamp", -1.0)
        if image_stamp < 0.0:
            return

        stage = str(data.get("stage", "UNKNOWN"))
        if stage != "PUB":
            return

        key = self.make_key(image_stamp)
        record = self.records[key]

        record["record_key_stamp"] = image_stamp
        record["image_stamp"] = image_stamp

        record["kalman_rx_now"] = self.get_float(data, "rx_now")
        record["kalman_proc_start"] = self.get_float(data, "proc_start")
        record["kalman_proc_end"] = self.get_float(data, "proc_end")
        record["kalman_pub_now"] = self.get_float(data, "pub_now")
        record["kalman_processing_dt"] = self.get_float(data, "processing_dt")
        record["kalman_send_dt"] = self.get_float(data, "send_dt")
        record["kalman_measurement_dt"] = self.get_float(data, "measurement_dt")
        record["kalman_predict_dt"] = self.get_float(data, "predict_dt")

        self.update_last_seen(record)

    def precision_land_callback(self, msg: String):
        """
        Nhận timing từ node PrecisionLand.

        Input:
            msg: std_msgs.msg.String

        Logic:
            Tương thích cả 2 schema:
            - image_stamp / total_image_to_cmd_dt
            - state_stamp / total_state_to_cmd_dt

        Output:
            Không có.
        """
        data = self.parse_json(msg)
        if data is None:
            return

        join_stamp = self.get_first_valid_stamp(
            data,
            ["image_stamp", "state_stamp", "pose_stamp"]
        )
        if join_stamp < 0.0:
            return

        key = self.make_key(join_stamp)
        record = self.records[key]

        record["record_key_stamp"] = join_stamp
        record["image_stamp"] = join_stamp

        record["pl_pose_stamp"] = self.get_float(data, "pose_stamp")
        record["pl_vel_stamp"] = self.get_float(data, "vel_stamp")
        record["pl_pose_rx_now"] = self.get_float(data, "pose_rx_now")
        record["pl_vel_rx_now"] = self.get_float(data, "vel_rx_now")
        record["pl_ctrl_start_now"] = self.get_float(data, "ctrl_start_now")
        record["pl_ctrl_end_now"] = self.get_float(data, "ctrl_end_now")
        record["pl_cmd_pub_now"] = self.get_float(data, "cmd_pub_now")
        record["pl_pose_wait_dt"] = self.get_float(data, "pose_wait_dt")
        record["pl_vel_wait_dt"] = self.get_float(data, "vel_wait_dt")
        record["pl_control_processing_dt"] = self.get_float(data, "control_processing_dt")
        record["pl_send_cmd_dt"] = self.get_float(data, "send_cmd_dt")

        total_dt = self.get_first_valid_stamp(
            data,
            ["total_image_to_cmd_dt", "total_state_to_cmd_dt"]
        )
        record["dt_total_image_to_cmd"] = total_dt

        self.update_last_seen(record)

    def build_row(self, record: dict) -> dict:
        """
        Tạo 1 dòng CSV hoàn chỉnh từ record đã gom.

        Input:
            record: dict

        Logic:
            Tính các dt giữa node dựa trên timestamp pub/rx.

        Output:
            dict
        """
        aruco_pub_now = float(record.get("aruco_pub_now", -1.0))
        kalman_rx_now = float(record.get("kalman_rx_now", -1.0))
        kalman_pub_now = float(record.get("kalman_pub_now", -1.0))
        pl_pose_rx_now = float(record.get("pl_pose_rx_now", -1.0))
        pl_vel_rx_now = float(record.get("pl_vel_rx_now", -1.0))

        dt_aruco_to_kalman = -1.0
        dt_kalman_to_pl_pose = -1.0
        dt_kalman_to_pl_vel = -1.0

        if aruco_pub_now >= 0.0 and kalman_rx_now >= 0.0:
            dt_aruco_to_kalman = max(0.0, kalman_rx_now - aruco_pub_now)

        if kalman_pub_now >= 0.0 and pl_pose_rx_now >= 0.0:
            dt_kalman_to_pl_pose = max(0.0, pl_pose_rx_now - kalman_pub_now)

        if kalman_pub_now >= 0.0 and pl_vel_rx_now >= 0.0:
            dt_kalman_to_pl_vel = max(0.0, pl_vel_rx_now - kalman_pub_now)

        return {
            "image_stamp": float(record.get("image_stamp", -1.0)),
            "aruco_found": int(record.get("aruco_found", -1)),

            "aruco_rx_now": float(record.get("aruco_rx_now", -1.0)),
            "aruco_proc_start": float(record.get("aruco_proc_start", -1.0)),
            "aruco_proc_end": float(record.get("aruco_proc_end", -1.0)),
            "aruco_pub_now": float(record.get("aruco_pub_now", -1.0)),
            "aruco_rx_wait_dt": float(record.get("aruco_rx_wait_dt", -1.0)),
            "aruco_queue_before_proc_dt": float(record.get("aruco_queue_before_proc_dt", -1.0)),
            "aruco_processing_dt": float(record.get("aruco_processing_dt", -1.0)),
            "aruco_send_dt": float(record.get("aruco_send_dt", -1.0)),
            "aruco_total_node_dt": float(record.get("aruco_total_node_dt", -1.0)),

            "kalman_rx_now": float(record.get("kalman_rx_now", -1.0)),
            "kalman_proc_start": float(record.get("kalman_proc_start", -1.0)),
            "kalman_proc_end": float(record.get("kalman_proc_end", -1.0)),
            "kalman_pub_now": float(record.get("kalman_pub_now", -1.0)),
            "kalman_processing_dt": float(record.get("kalman_processing_dt", -1.0)),
            "kalman_send_dt": float(record.get("kalman_send_dt", -1.0)),
            "kalman_measurement_dt": float(record.get("kalman_measurement_dt", -1.0)),
            "kalman_predict_dt": float(record.get("kalman_predict_dt", -1.0)),

            "pl_pose_stamp": float(record.get("pl_pose_stamp", -1.0)),
            "pl_vel_stamp": float(record.get("pl_vel_stamp", -1.0)),
            "pl_pose_rx_now": float(record.get("pl_pose_rx_now", -1.0)),
            "pl_vel_rx_now": float(record.get("pl_vel_rx_now", -1.0)),
            "pl_ctrl_start_now": float(record.get("pl_ctrl_start_now", -1.0)),
            "pl_ctrl_end_now": float(record.get("pl_ctrl_end_now", -1.0)),
            "pl_cmd_pub_now": float(record.get("pl_cmd_pub_now", -1.0)),
            "pl_pose_wait_dt": float(record.get("pl_pose_wait_dt", -1.0)),
            "pl_vel_wait_dt": float(record.get("pl_vel_wait_dt", -1.0)),
            "pl_control_processing_dt": float(record.get("pl_control_processing_dt", -1.0)),
            "pl_send_cmd_dt": float(record.get("pl_send_cmd_dt", -1.0)),

            "dt_aruco_to_kalman": dt_aruco_to_kalman,
            "dt_kalman_to_pl_pose": dt_kalman_to_pl_pose,
            "dt_kalman_to_pl_vel": dt_kalman_to_pl_vel,
            "dt_total_image_to_cmd": float(record.get("dt_total_image_to_cmd", -1.0)),
        }

    def flush_to_csv(self):
        """
        Flush các record complete ra CSV.

        Input:
            Không có.

        Logic:
            - Ghi record complete
            - Drop record stale nhưng không đủ dữ liệu

        Output:
            Không có.
        """
        now_sec = self.get_clock().now().nanoseconds / 1e9

        keys_complete = []
        keys_drop = []

        for key, record in self.records.items():
            if self.has_complete_record(record):
                keys_complete.append(key)
            elif self.is_stale_record(record, now_sec):
                keys_drop.append(key)

        if not keys_complete and not keys_drop:
            return

        keys_complete.sort(key=lambda item: float(item))
        rows = [self.build_row(self.records[key]) for key in keys_complete]

        fieldnames = [
            "image_stamp",
            "aruco_found",

            "aruco_rx_now",
            "aruco_proc_start",
            "aruco_proc_end",
            "aruco_pub_now",
            "aruco_rx_wait_dt",
            "aruco_queue_before_proc_dt",
            "aruco_processing_dt",
            "aruco_send_dt",
            "aruco_total_node_dt",

            "kalman_rx_now",
            "kalman_proc_start",
            "kalman_proc_end",
            "kalman_pub_now",
            "kalman_processing_dt",
            "kalman_send_dt",
            "kalman_measurement_dt",
            "kalman_predict_dt",

            "pl_pose_stamp",
            "pl_vel_stamp",
            "pl_pose_rx_now",
            "pl_vel_rx_now",
            "pl_ctrl_start_now",
            "pl_ctrl_end_now",
            "pl_cmd_pub_now",
            "pl_pose_wait_dt",
            "pl_vel_wait_dt",
            "pl_control_processing_dt",
            "pl_send_cmd_dt",

            "dt_aruco_to_kalman",
            "dt_kalman_to_pl_pose",
            "dt_kalman_to_pl_vel",
            "dt_total_image_to_cmd",
        ]

        if rows:
            output_dir = os.path.dirname(self.output_csv)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(self.output_csv, "a", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fieldnames)

                if not self.header_written:
                    writer.writeheader()
                    self.header_written = True

                for row in rows:
                    writer.writerow(row)

        for key in keys_complete:
            del self.records[key]

        for key in keys_drop:
            del self.records[key]

        if rows:
            self.get_logger().info(f"Wrote {len(rows)} rows to {self.output_csv}")

        if keys_drop:
            self.get_logger().warn(f"Dropped {len(keys_drop)} stale incomplete records")


def main(args=None):
    rclpy.init(args=args)
    node = TimingCollectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()