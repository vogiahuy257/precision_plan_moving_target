#pragma once

#include <cstdint>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kalman_filter_data
{
enum class MountMode
{
    BellyFixedCamera,
    BellyFixedCameraRight90,
    BellyGimbalCamera
};

struct TopicConfig
{
    std::string inputTargetPoseTopic{"/Aruco/target_pose_FRD"};
    std::string resetCommandTopic{"/Aruco/target_state"};
    std::string targetValidTopic{"/target_valid"};
    std::string vehicleOdometryTopic{"/fmu/out/vehicle_odometry"};
    std::string vehicleLocalPositionTopic{"/fmu/out/vehicle_local_position"};
    std::string relativePositionRawTopic{"/KalmanFilter/target_pose_NED"};
    std::string relativePositionPredictedTopic{"/KalmanFilter/target_pose_est_NED"};
    std::string relativeVelocityTopic{"/KalmanFilter/target_velocity_est_NED"};
    std::string outputFrameId{"map"};
};

struct NoiseConfig
{
    double qAccX{0.0002};
    double qAccY{0.0002};
    double qAccZ{0.0010};

    double rPosX{0.0008};
    double rPosY{0.0008};
    double rPosZ{0.0040};

    bool dynamicREnabled{true};
    double nearRange{0.7};
    double nearNoiseGain{0.08};
    double maxExtraRxy{0.20};
    double minDynamicRange{0.1};
};

struct DebugConfig
{
    bool enabled{false};
    std::string csvPath{"logs/kalman_filter_debug.csv"};
};

struct TransformConfig
{
    MountMode mountMode{MountMode::BellyFixedCamera};
    std::string mountModeString{"belly_fixed_camera"};

    Eigen::Vector3d cameraOffsetBody{0.0, 0.0, -0.1};
    Eigen::Matrix3d opticalToMountRotation{Eigen::Matrix3d::Identity()};
};

struct NodeConfig
{
    TopicConfig topics{};
    NoiseConfig noise{};
    DebugConfig debug{};
    TransformConfig transform{};
    double poseTimeoutSec{3.0};
};

struct RuntimeFlags
{
    bool initialized{false};
    bool forceZero{false};
    bool targetValid{false};

    bool vehicleOdomValid{false};
    bool vehicleLocalPosValid{false};

    std::string lastResetCommand{"NONE"};
};

struct VehicleStateData
{
    Eigen::Quaterniond worldFromBody{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d positionWorld{Eigen::Vector3d::Zero()};
    Eigen::Vector3d velocityWorld{Eigen::Vector3d::Zero()};
    bool valid{false};
};

struct MountStateData
{
    Eigen::Quaterniond bodyFromMount{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d eulerDeg{Eigen::Vector3d::Zero()};
    bool valid{false};
};

struct TargetMeasurementData
{
    rclcpp::Time stamp{0, 0, RCL_ROS_TIME};

    Eigen::Vector3d positionOptical{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientationOptical{Eigen::Quaterniond::Identity()};

    Eigen::Vector3d positionWorld{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientationWorld{Eigen::Quaterniond::Identity()};

    bool valid{false};
};

struct KalmanEstimateData
{
    Eigen::Vector3d rawMeasurementWorld{Eigen::Vector3d::Zero()};
    Eigen::Vector3d estimatedPositionWorld{Eigen::Vector3d::Zero()};
    Eigen::Vector3d estimatedVelocityWorld{Eigen::Vector3d::Zero()};

    double predictDt{0.0};
    std::uint64_t predictCount{0};

    double dynamicRx{0.0008};
    double dynamicRy{0.0008};
    double dynamicRz{0.0040};
    double dynamicRExtraXY{0.0};
    double dynamicRangeToTarget{0.0};
    double dynamicNearRangeError{0.0};
};

struct TimingData
{
    rclcpp::Time lastPredictTime{0, 0, RCL_ROS_TIME};
    rclcpp::Time lastMeasurementTime{0, 0, RCL_ROS_TIME};
};

struct SystemData
{
    NodeConfig config{};
    RuntimeFlags runtime{};
    VehicleStateData vehicle{};
    MountStateData mount{};
    TargetMeasurementData targetMeasurement{};
    KalmanEstimateData kalman{};
    TimingData timing{};
};

struct DebugLogRow
{
    double stampSec{0.0};

    bool initialized{false};
    bool forceZero{false};
    bool targetValid{false};
    bool vehicleOdomValid{false};
    bool vehicleLocalPosValid{false};

    double vehiclePosX{0.0};
    double vehiclePosY{0.0};
    double vehiclePosZ{0.0};

    double vehicleVelX{0.0};
    double vehicleVelY{0.0};
    double vehicleVelZ{0.0};

    double measOptX{0.0};
    double measOptY{0.0};
    double measOptZ{0.0};

    double measWorldX{0.0};
    double measWorldY{0.0};
    double measWorldZ{0.0};

    double estPosX{0.0};
    double estPosY{0.0};
    double estPosZ{0.0};

    double estVelX{0.0};
    double estVelY{0.0};
    double estVelZ{0.0};

    double predictDt{0.0};
    std::uint64_t predictCount{0};

    std::string mountMode{"belly_fixed_camera"};
    std::string lastResetCommand{"NONE"};
};

} // namespace kalman_filter_data