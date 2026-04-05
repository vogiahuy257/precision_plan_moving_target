#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class KalmanTuningReader(Node):
    def __init__(self):
        super().__init__('kalman_tuning_reader')

        self.declare_parameter('topic_name', '/target_pose')
        self.declare_parameter('max_samples', 100)
        self.declare_parameter('output_csv', 'target_pose_samples.csv')
        self.declare_parameter('use_msg_stamp', True)

        topic_name = self.get_parameter('topic_name').value
        self.max_samples = int(self.get_parameter('max_samples').value)
        self.output_csv = str(self.get_parameter('output_csv').value)
        self.use_msg_stamp = bool(self.get_parameter('use_msg_stamp').value)

        self.samples = []  # [t, x, y, z]

        self.sub = self.create_subscription(
            PoseStamped,
            topic_name,
            self.callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"Listening on {topic_name}, collecting {self.max_samples} samples..."
        )

    def callback(self, msg: PoseStamped):
        if self.use_msg_stamp:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        else:
            t = self.get_clock().now().nanoseconds * 1e-9

        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        self.samples.append([t, x, y, z])

        n = len(self.samples)
        if n % 10 == 0:
            self.get_logger().info(f"Collected {n}/{self.max_samples} samples")

        if n >= self.max_samples:
            self.finish()

    def finish(self):
        data = np.array(self.samples, dtype=np.float64)

        output_path = Path(self.output_csv)
        with output_path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['t', 'x', 'y', 'z'])
            writer.writerows(self.samples)

        t = data[:, 0]
        pos = data[:, 1:4]  # x,y,z

        dt = np.diff(t)
        valid_dt_mask = dt > 1e-9

        if np.count_nonzero(valid_dt_mask) < 3:
            self.get_logger().error("Not enough valid dt samples to estimate velocity/acceleration.")
            rclpy.shutdown()
            return

        t0 = t[0]
        duration = t[-1] - t[0]
        mean_dt = np.mean(dt[valid_dt_mask])

        # ===== ƯỚC LƯỢNG R =====
        mean_pos = np.mean(pos, axis=0)
        var_pos = np.var(pos, axis=0, ddof=1)
        std_pos = np.std(pos, axis=0, ddof=1)

        # ===== ƯỚC LƯỢNG v, a BẰNG SAI PHÂN =====
        vel = []
        vel_t = []
        for i in range(len(t) - 1):
            dti = t[i + 1] - t[i]
            if dti <= 1e-9:
                continue
            vi = (pos[i + 1] - pos[i]) / dti
            vel.append(vi)
            vel_t.append(0.5 * (t[i + 1] + t[i]))

        vel = np.array(vel, dtype=np.float64)
        vel_t = np.array(vel_t, dtype=np.float64)

        if len(vel) < 3:
            self.get_logger().error("Not enough velocity samples.")
            rclpy.shutdown()
            return

        acc = []
        for i in range(len(vel_t) - 1):
            dti = vel_t[i + 1] - vel_t[i]
            if dti <= 1e-9:
                continue
            ai = (vel[i + 1] - vel[i]) / dti
            acc.append(ai)

        acc = np.array(acc, dtype=np.float64)

        if len(acc) < 3:
            self.get_logger().error("Not enough acceleration samples.")
            rclpy.shutdown()
            return

        mean_vel = np.mean(vel, axis=0)
        var_vel = np.var(vel, axis=0, ddof=1)
        std_vel = np.std(vel, axis=0, ddof=1)

        mean_acc = np.mean(acc, axis=0)
        var_acc = np.var(acc, axis=0, ddof=1)
        std_acc = np.std(acc, axis=0, ddof=1)

        q_acc_x = var_acc[0]
        q_acc_y = var_acc[1]
        q_acc_z = var_acc[2]

        r_pos_x = 0.1 * var_pos[0]
        r_pos_y = 0.1 * var_pos[1]
        r_pos_z = 0.1 * var_pos[2]

        self.get_logger().info("========== KALMAN TUNING REPORT ==========")
        self.get_logger().info(f"Saved CSV     : {output_path.resolve()}")
        self.get_logger().info(f"Samples       : {len(data)}")
        self.get_logger().info(f"Start time    : {t0:.6f}")
        self.get_logger().info(f"Duration      : {duration:.6f} s")
        self.get_logger().info(f"Mean dt       : {mean_dt:.6f} s")
        self.get_logger().info("")

        self.get_logger().info("----- POSITION STATS (for R) -----")
        self.get_logger().info(
            f"mean_pos  = x:{mean_pos[0]:.6f}, y:{mean_pos[1]:.6f}, z:{mean_pos[2]:.6f}"
        )
        self.get_logger().info(
            f"std_pos   = x:{std_pos[0]:.6f}, y:{std_pos[1]:.6f}, z:{std_pos[2]:.6f}"
        )
        self.get_logger().info(
            f"var_pos   = x:{var_pos[0]:.8f}, y:{var_pos[1]:.8f}, z:{var_pos[2]:.8f}"
        )
        self.get_logger().info("")

        self.get_logger().info("----- VELOCITY STATS -----")
        self.get_logger().info(
            f"mean_vel  = x:{mean_vel[0]:.6f}, y:{mean_vel[1]:.6f}, z:{mean_vel[2]:.6f}"
        )
        self.get_logger().info(
            f"std_vel   = x:{std_vel[0]:.6f}, y:{std_vel[1]:.6f}, z:{std_vel[2]:.6f}"
        )
        self.get_logger().info(
            f"var_vel   = x:{var_vel[0]:.8f}, y:{var_vel[1]:.8f}, z:{var_vel[2]:.8f}"
        )
        self.get_logger().info("")

        self.get_logger().info("----- ACCELERATION STATS (for Q) -----")
        self.get_logger().info(
            f"mean_acc  = x:{mean_acc[0]:.6f}, y:{mean_acc[1]:.6f}, z:{mean_acc[2]:.6f}"
        )
        self.get_logger().info(
            f"std_acc   = x:{std_acc[0]:.6f}, y:{std_acc[1]:.6f}, z:{std_acc[2]:.6f}"
        )
        self.get_logger().info(
            f"var_acc   = x:{var_acc[0]:.8f}, y:{var_acc[1]:.8f}, z:{var_acc[2]:.8f}"
        )
        self.get_logger().info("")

        self.get_logger().info("----- SUGGESTED PARAMETERS -----")
        self.get_logger().info(f"r_pos_x = {r_pos_x:.8f}")
        self.get_logger().info(f"r_pos_y = {r_pos_y:.8f}")
        self.get_logger().info(f"r_pos_z = {r_pos_z:.8f}")
        self.get_logger().info(f"q_acc_x = {q_acc_x:.8f}")
        self.get_logger().info(f"q_acc_y = {q_acc_y:.8f}")
        self.get_logger().info(f"q_acc_z = {q_acc_z:.8f}")
        self.get_logger().info("=========================================")

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = KalmanTuningReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()