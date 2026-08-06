#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ImuDtCheck(Node):
    def __init__(self):
        super().__init__("check_imu_dt")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.last_t = None
        self.count = 0
        self.bad_count = 0

        self.create_subscription(Imu, "/imu0", self.cb, qos)

    def cb(self, msg):
        t = stamp_to_sec(msg.header.stamp)

        if self.last_t is not None:
            dt = t - self.last_t
            self.count += 1

            if dt <= 0.0 or dt > 0.03:
                self.bad_count += 1
                print(f"BAD dt={dt:+.9f}s  t={t:.9f}  bad={self.bad_count}/{self.count}")
            elif self.count % 100 == 0:
                print(f"OK dt={dt:.6f}s")

        self.last_t = t


def main():
    rclpy.init()
    node = ImuDtCheck()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()