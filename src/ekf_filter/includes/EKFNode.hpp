#pragma once

#include <deque>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "CtraEkf.hpp"
#include "FrameTransformer.hpp"

class EKFNode : public rclcpp::Node
{
public:
    EKFNode();

private:
    struct BootstrapData
    {
        Eigen::Vector3d positionNed{Eigen::Vector3d::Zero()};
        Eigen::Quaterniond orientationNed{Eigen::Quaterniond::Identity()};
        rclcpp::Time stamp{0, 0, RCL_ROS_TIME};
        bool valid{false};
    };

    void declareParameters();
    void loadParameters();
    void setupRosInterfaces();
    void resetFilter();

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void resetCallback(const std_msgs::msg::String::SharedPtr msg);
    void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    bool bootstrapFilter(
        const rclcpp::Time &stamp,
        const Eigen::Vector3d &positionNed,
        const Eigen::Quaterniond &orientationNed);

    void publishRaw(const rclcpp::Time &stamp);
    void publishEstimate(const rclcpp::Time &stamp);
    void publishEstimateFromFilter(const CtraEkf &filter, const rclcpp::Time &stamp);
    void publishLostPrediction(const rclcpp::Time &stamp);
    void publishHold(const rclcpp::Time &stamp);
    void publishCovariance(const CtraEkf::Matrix6d &covariance);
    void publishProcessNoise();

    static double yawFromQuaternion(const Eigen::Quaterniond &q);
    static geometry_msgs::msg::Quaternion quaternionFromYaw(double yawRad);

    // Topics.
    std::string inputTargetPoseTopic_;
    std::string resetCommandTopic_;
    std::string vehicleOdometryTopic_;
    std::string vehicleLocalPositionTopic_;
    std::string rawPoseTopic_;
    std::string filteredPoseTopic_;
    std::string velocityTopic_;
    std::string motionTopic_;
    std::string covarianceTopic_;
    std::string processNoiseTopic_;
    std::string frameId_;

    // Filter parameters.
    CtraEkf::Config ekfConfig_{};
    double initMinSpeedMps_{0.10};
    double initMotionNisThreshold_{9.21};
    int initWindowSize_{15};
    CtraEkf::Matrix6d initialCovariance_{CtraEkf::Matrix6d::Identity()};

    Eigen::Vector3d cameraOffsetBody_{0.2, 0.0, -0.12};

    // Runtime.
    CtraEkf ekf_{};
    FrameTransformer frameTransformer_{};
    std::deque<BootstrapData> bootstrapSamples_{};

    Eigen::Vector3d vehiclePositionNed_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond worldFromBody_{Eigen::Quaterniond::Identity()};
    bool vehiclePositionValid_{false};
    bool vehicleAttitudeValid_{false};

    Eigen::Vector3d rawMeasurementNed_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond rawOrientationNed_{Eigen::Quaterniond::Identity()};
    double targetDown_{0.0};

    rclcpp::Time lastPredictTime_{0, 0, RCL_ROS_TIME};
    bool forceHold_{false};

    // LOST mode keeps the main EKF untouched. A snapshot is propagated only
    // for output until ArUco is reacquired or sends RESET after 5 seconds.
    bool targetLost_{false};
    CtraEkf lostEkfSnapshot_{};
    rclcpp::Time lostStateStamp_{0, 0, RCL_ROS_TIME};

    // ROS interfaces.
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr poseSub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr resetSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicleOdomSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicleLocalPosSub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr rawPosePub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr filteredPosePub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr velocityPub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr motionPub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr covariancePub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr processNoisePub_;
};
