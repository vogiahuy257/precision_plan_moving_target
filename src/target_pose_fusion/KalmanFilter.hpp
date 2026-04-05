#pragma once

#include <atomic>

#include <rclcpp/rclcpp.hpp>

#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <opencv2/video/tracking.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

class TargetPoseFusionNode : public rclcpp::Node
{
public:
    TargetPoseFusionNode();

private:
    static constexpr int STATE_SIZE = 6;
    static constexpr int MEASUREMENT_SIZE = 3;

    void declareParameters();
    void initKalman();
    void resetState();

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void resetCallback(const std_msgs::msg::String::SharedPtr msg);
    void validCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    void processAndPublish();
    void predict(double dt);
    void publishEstimatedState(const rclcpp::Time& nowTimestamp, double dt);
    void publishZero(const rclcpp::Time& nowTimestamp);

    Eigen::Matrix3d opticalToNedRotation() const;
    Eigen::Vector3d measurementOpticalToWorldPosition(const Eigen::Vector3d& opticalPosition) const;
    Eigen::Quaterniond transformTagOrientationToWorld(
        const geometry_msgs::msg::Quaternion& quaternionMessage) const;

private:
    // ===== Subscribers =====
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr reset_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr valid_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicle_odom_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicle_local_pos_sub_;

    // ===== Publishers =====
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr kalman_residual_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_world_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_error_pred_raw_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_error_pred_fusion_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_rel_vel_pub_;

    // ===== Timer =====
    rclcpp::TimerBase::SharedPtr timer_;

    // ===== Kalman filter =====
    cv::KalmanFilter kf_;

    // ===== Parameters =====
    double q_acc_x_{0.0002};
    double q_acc_y_{0.0002};
    double q_acc_z_{0.001};

    double r_pos_x_{0.0008};
    double r_pos_y_{0.0008};
    double r_pos_z_{0.004};

    double cam_offset_x_{0.0};
    double cam_offset_y_{0.0};
    double cam_offset_z_{-0.1};

    // ===== State flags =====
    bool initialized_{false};
    bool vehicle_odom_valid_{false};
    bool vehicle_local_pos_valid_{false};

    std::atomic<bool> force_zero_{false};
    std::atomic<bool> target_valid_{false};

    // ===== Vehicle state =====
    Eigen::Quaterniond vehicle_q_ned_{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d vehicle_pos_ned_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vehicle_vel_ned_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d vehicle_acc_ned_{Eigen::Vector3d::Zero()};

    // ===== Last target data =====
    Eigen::Vector3d last_measurement_world_{Eigen::Vector3d::Zero()};
    geometry_msgs::msg::Quaternion last_orientation_{};

    // ===== Timing =====
    rclcpp::Time last_predict_time_{0, 0, RCL_ROS_TIME};
    rclcpp::Time last_measurement_time_{0, 0, RCL_ROS_TIME};
};