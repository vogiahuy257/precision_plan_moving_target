#pragma once

#include <atomic>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/video/tracking.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

class KalmanFilterNode : public rclcpp::Node
{
public:
    KalmanFilterNode();

private:
    static constexpr int stateSize = 6;
    static constexpr int measurementSize = 3;

    void declareParameters();
    void loadParameters();
    void initKalman();
    void resetState();

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void resetCallback(const std_msgs::msg::String::SharedPtr msg);
    void validCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

    void processAndPublish();

    void predict(double dt);
    double sanitizeDt(double dt) const;

    Eigen::Matrix3d opticalToNedRotation() const;
    Eigen::Vector3d measurementOpticalToWorldPosition(const Eigen::Vector3d &opticalPosition) const;
    Eigen::Quaterniond transformTagOrientationToWorld(const geometry_msgs::msg::Quaternion &quaternionMessage) const;

    void publishEstimatedState(const rclcpp::Time &publishTimestamp);
    void publishZero(const rclcpp::Time &publishTimestamp);

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr poseSub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr resetSub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr validSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicleOdomSub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoseRawPub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoseFilteredPub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetVelocityFilteredPub_;

    rclcpp::TimerBase::SharedPtr timer_;

    std::string inputTargetPoseTopic_;
    std::string resetCommandTopic_;
    std::string targetValidTopic_;
    std::string vehicleOdometryTopic_;

    std::string targetPoseRawTopic_;
    std::string targetPoseFilteredTopic_;
    std::string targetVelocityTopic_;

    std::string outputFrameId_;

    double qAccX_{0.0005};
    double qAccY_{0.0005};
    double qAccZ_{0.0010};

    double rPosX_{0.000025};
    double rPosY_{0.000025};
    double rPosZ_{0.0040};

    double camOffsetX_{0.0};
    double camOffsetY_{0.0};
    double camOffsetZ_{-0.1};

    double maxPredictDt_{0.1};
    double staleMeasurementThresholdSec_{0.2};
    double smallNegativeDtToleranceSec_{0.02};

    cv::KalmanFilter kf_;
    bool initialized_{false};

    // Timestamp của measurement cuối cùng đã correct vào Kalman
    rclcpp::Time lastMeasurementTimestamp_{0, 0, RCL_ROS_TIME};

    Eigen::Quaterniond vehicleQned_{1.0, 0.0, 0.0, 0.0};
    Eigen::Vector3d vehiclePosNed_{0.0, 0.0, 0.0};
    Eigen::Vector3d vehicleVelNed_{0.0, 0.0, 0.0};
    bool vehicleOdomValid_{false};

    Eigen::Vector3d lastMeasurementWorld_{0.0, 0.0, 0.0};
    geometry_msgs::msg::Quaternion lastOrientation_;

    std::atomic<bool> forceZero_{false};
};