#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, Imu


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class SyncCheck(Node):
    def __init__(self):
        super().__init__("check_vins_sync")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        self.last_imu_t = None
        self.last_img_t = None
        self.imu_count = 0

        self.create_subscription(Imu, "/imu0", self.imu_cb, qos)
        self.create_subscription(Image, "/camera/image", self.img_cb, qos)

    def imu_cb(self, msg):
        self.last_imu_t = stamp_to_sec(msg.header.stamp)
        self.imu_count += 1

    def img_cb(self, msg):
        img_t = stamp_to_sec(msg.header.stamp)
        self.last_img_t = img_t

        if self.last_imu_t is None:
            print("IMG received but no IMU yet")
            return

        dt = self.last_imu_t - img_t

        print(
            f"img_t={img_t:.6f}  "
            f"last_imu_t={self.last_imu_t:.6f}  "
            f"imu_minus_img={dt:+.6f}s  "
            f"imu_count={self.imu_count}"
        )


def main():
    rclpy.init()
    node = SyncCheck()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()