#include "KalmanFilter.hpp"

#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/twist_stamped.hpp>

namespace
{
constexpr char kInputTargetPoseTopic[] = "/target_pose";
constexpr char kResetCommandTopic[] = "/reset";
constexpr char kTargetValidTopic[] = "/target_valid";

constexpr char kVehicleOdometryTopic[] = "/fmu/out/vehicle_odometry";
constexpr char kVehicleLocalPositionTopic[] = "/fmu/out/vehicle_local_position_v1";

constexpr char kKalmanResidualTopic[] = "/target_fusion/kalman_residual";
constexpr char kTargetWorldPoseTopic[] = "/target_fusion/target_world_pose";
constexpr char kRelativePositionRawTopic[] = "/target_fusion/relative_position_raw";
constexpr char kRelativePositionPredictedTopic[] = "/target_fusion/relative_position_predicted";
constexpr char kRelativeVelocityTopic[] = "/target_fusion/relative_velocity";

constexpr char kFrameId[] = "map";
}

TargetPoseFusionNode::TargetPoseFusionNode()
    : Node("target_pose_fusion_node")
{
    const auto subQos = rclcpp::QoS(1).best_effort();
    const auto pubQos = rclcpp::QoS(1).best_effort();

    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        kInputTargetPoseTopic,
        subQos,
        std::bind(&TargetPoseFusionNode::poseCallback, this, std::placeholders::_1));

    reset_sub_ = create_subscription<std_msgs::msg::String>(
        kResetCommandTopic,
        subQos,
        std::bind(&TargetPoseFusionNode::resetCallback, this, std::placeholders::_1));

    valid_sub_ = create_subscription<std_msgs::msg::Bool>(
        kTargetValidTopic,
        subQos,
        std::bind(&TargetPoseFusionNode::validCallback, this, std::placeholders::_1));

    vehicle_odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        kVehicleOdometryTopic,
        subQos,
        std::bind(&TargetPoseFusionNode::vehicleOdometryCallback, this, std::placeholders::_1));

    vehicle_local_pos_sub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        kVehicleLocalPositionTopic,
        subQos,
        std::bind(&TargetPoseFusionNode::vehicleLocalPositionCallback, this, std::placeholders::_1));

    kalman_residual_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kKalmanResidualTopic,
        pubQos);

    target_pose_world_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kTargetWorldPoseTopic,
        pubQos);

    target_error_pred_raw_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kRelativePositionRawTopic,
        pubQos);

    target_error_pred_fusion_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kRelativePositionPredictedTopic,
        pubQos);

    target_rel_vel_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        kRelativeVelocityTopic,
        pubQos);

    declareParameters();
    initKalman();

    timer_ = create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&TargetPoseFusionNode::processAndPublish, this));

    last_predict_time_ = now();
    last_measurement_time_ = now();
}

void TargetPoseFusionNode::declareParameters()
{
    declare_parameter<double>("q_acc_x", 0.0002);
    declare_parameter<double>("q_acc_y", 0.0002);
    declare_parameter<double>("q_acc_z", 0.001);

    declare_parameter<double>("r_pos_x", 0.0008);
    declare_parameter<double>("r_pos_y", 0.0008);
    declare_parameter<double>("r_pos_z", 0.004);

    declare_parameter<double>("cam_offset_x", 0.0);
    declare_parameter<double>("cam_offset_y", 0.0);
    declare_parameter<double>("cam_offset_z", -0.1);
}

void TargetPoseFusionNode::initKalman()
{
    q_acc_x_ = get_parameter("q_acc_x").as_double();
    q_acc_y_ = get_parameter("q_acc_y").as_double();
    q_acc_z_ = get_parameter("q_acc_z").as_double();

    r_pos_x_ = get_parameter("r_pos_x").as_double();
    r_pos_y_ = get_parameter("r_pos_y").as_double();
    r_pos_z_ = get_parameter("r_pos_z").as_double();

    cam_offset_x_ = get_parameter("cam_offset_x").as_double();
    cam_offset_y_ = get_parameter("cam_offset_y").as_double();
    cam_offset_z_ = get_parameter("cam_offset_z").as_double();

    kf_ = cv::KalmanFilter(STATE_SIZE, MEASUREMENT_SIZE, 0, CV_64F);

    kf_.transitionMatrix = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);

    kf_.measurementMatrix = cv::Mat::zeros(MEASUREMENT_SIZE, STATE_SIZE, CV_64F);
    kf_.measurementMatrix.at<double>(0, 0) = 1.0;
    kf_.measurementMatrix.at<double>(1, 1) = 1.0;
    kf_.measurementMatrix.at<double>(2, 2) = 1.0;

    kf_.processNoiseCov = cv::Mat::zeros(STATE_SIZE, STATE_SIZE, CV_64F);

    kf_.measurementNoiseCov = cv::Mat::eye(MEASUREMENT_SIZE, MEASUREMENT_SIZE, CV_64F);
    kf_.measurementNoiseCov.at<double>(0, 0) = r_pos_x_;
    kf_.measurementNoiseCov.at<double>(1, 1) = r_pos_y_;
    kf_.measurementNoiseCov.at<double>(2, 2) = r_pos_z_;

    kf_.errorCovPost = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);
    kf_.errorCovPost.at<double>(3, 3) = 10.0;
    kf_.errorCovPost.at<double>(4, 4) = 10.0;
    kf_.errorCovPost.at<double>(5, 5) = 10.0;

    kf_.statePost = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);
}

void TargetPoseFusionNode::vehicleOdometryCallback(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    vehicle_q_ned_ = Eigen::Quaterniond(
        static_cast<double>(msg->q[0]),
        static_cast<double>(msg->q[1]),
        static_cast<double>(msg->q[2]),
        static_cast<double>(msg->q[3]));

    if (vehicle_q_ned_.norm() > 1e-9)
    {
        vehicle_q_ned_.normalize();
        vehicle_odom_valid_ = true;
    }
}

void TargetPoseFusionNode::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    vehicle_pos_ned_.x() = static_cast<double>(msg->x);
    vehicle_pos_ned_.y() = static_cast<double>(msg->y);
    vehicle_pos_ned_.z() = static_cast<double>(msg->z);

    vehicle_vel_ned_.x() = static_cast<double>(msg->vx);
    vehicle_vel_ned_.y() = static_cast<double>(msg->vy);
    vehicle_vel_ned_.z() = static_cast<double>(msg->vz);

    vehicle_acc_ned_.x() = static_cast<double>(msg->ax);
    vehicle_acc_ned_.y() = static_cast<double>(msg->ay);
    vehicle_acc_ned_.z() = static_cast<double>(msg->az);

    vehicle_local_pos_valid_ =
        msg->xy_valid &&
        msg->z_valid &&
        msg->v_xy_valid &&
        msg->v_z_valid;
}

void TargetPoseFusionNode::resetCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (msg->data == "RESET")
    {
        resetState();
        force_zero_.store(true, std::memory_order_relaxed);
        return;
    }

    if (msg->data == "ACTIVE")
    {
        force_zero_.store(false, std::memory_order_relaxed);
    }
}

void TargetPoseFusionNode::validCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    target_valid_.store(msg->data, std::memory_order_relaxed);

    if (msg->data)
    {
        force_zero_.store(false, std::memory_order_relaxed);
    }
}

void TargetPoseFusionNode::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!vehicle_odom_valid_ || !vehicle_local_pos_valid_)
    {
        return;
    }

    rclcpp::Time measurementTimestamp = msg->header.stamp;
    if (measurementTimestamp.nanoseconds() == 0)
    {
        measurementTimestamp = now();
    }

    last_measurement_time_ = measurementTimestamp;
    last_orientation_ = msg->pose.orientation;

    const Eigen::Vector3d measurementOptical(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    const Eigen::Vector3d measurementWorld =
        measurementOpticalToWorldPosition(measurementOptical);

    last_measurement_world_ = measurementWorld;

    if (!initialized_)
    {
        kf_.statePost.at<double>(0, 0) = measurementWorld.x();
        kf_.statePost.at<double>(1, 0) = measurementWorld.y();
        kf_.statePost.at<double>(2, 0) = measurementWorld.z();
        kf_.statePost.at<double>(3, 0) = 0.0;
        kf_.statePost.at<double>(4, 0) = 0.0;
        kf_.statePost.at<double>(5, 0) = 0.0;

        kf_.statePre = kf_.statePost.clone();
        initialized_ = true;
        last_predict_time_ = measurementTimestamp;
        return;
    }

    cv::Mat measurement(MEASUREMENT_SIZE, 1, CV_64F);
    measurement.at<double>(0, 0) = measurementWorld.x();
    measurement.at<double>(1, 0) = measurementWorld.y();
    measurement.at<double>(2, 0) = measurementWorld.z();

    const cv::Mat predictedMeasurement = kf_.measurementMatrix * kf_.statePre;
    const cv::Mat residual = measurement - predictedMeasurement;

    kf_.correct(measurement);

    geometry_msgs::msg::PoseStamped residualMsg;
    residualMsg.header.stamp = measurementTimestamp;
    residualMsg.header.frame_id = kFrameId;
    residualMsg.pose.position.x = residual.at<double>(0, 0);
    residualMsg.pose.position.y = residual.at<double>(1, 0);
    residualMsg.pose.position.z = residual.at<double>(2, 0);
    residualMsg.pose.orientation.w = 1.0;
    residualMsg.pose.orientation.x = 0.0;
    residualMsg.pose.orientation.y = 0.0;
    residualMsg.pose.orientation.z = 0.0;
    kalman_residual_pub_->publish(residualMsg);
}

void TargetPoseFusionNode::processAndPublish()
{
    const rclcpp::Time nowTimestamp = now();

    if (force_zero_.load(std::memory_order_relaxed))
    {
        publishZero(nowTimestamp);
        return;
    }

    if (!initialized_ || !vehicle_odom_valid_ || !vehicle_local_pos_valid_)
    {
        return;
    }

    double dt = (nowTimestamp - last_predict_time_).seconds();
    if (dt <= 0.0)
    {
        dt = 1e-3;
    }

    dt = std::min(dt, 0.1);
    last_predict_time_ = nowTimestamp;

    predict(dt);
    publishEstimatedState(nowTimestamp, dt);
}

void TargetPoseFusionNode::predict(double dt)
{
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    kf_.transitionMatrix = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);
    kf_.transitionMatrix.at<double>(0, 3) = dt;
    kf_.transitionMatrix.at<double>(1, 4) = dt;
    kf_.transitionMatrix.at<double>(2, 5) = dt;

    kf_.processNoiseCov = cv::Mat::zeros(STATE_SIZE, STATE_SIZE, CV_64F);

    kf_.processNoiseCov.at<double>(0, 0) = 0.25 * dt4 * q_acc_x_;
    kf_.processNoiseCov.at<double>(0, 3) = 0.5 * dt3 * q_acc_x_;
    kf_.processNoiseCov.at<double>(3, 0) = 0.5 * dt3 * q_acc_x_;
    kf_.processNoiseCov.at<double>(3, 3) = dt2 * q_acc_x_;

    kf_.processNoiseCov.at<double>(1, 1) = 0.25 * dt4 * q_acc_y_;
    kf_.processNoiseCov.at<double>(1, 4) = 0.5 * dt3 * q_acc_y_;
    kf_.processNoiseCov.at<double>(4, 1) = 0.5 * dt3 * q_acc_y_;
    kf_.processNoiseCov.at<double>(4, 4) = dt2 * q_acc_y_;

    kf_.processNoiseCov.at<double>(2, 2) = 0.25 * dt4 * q_acc_z_;
    kf_.processNoiseCov.at<double>(2, 5) = 0.5 * dt3 * q_acc_z_;
    kf_.processNoiseCov.at<double>(5, 2) = 0.5 * dt3 * q_acc_z_;
    kf_.processNoiseCov.at<double>(5, 5) = dt2 * q_acc_z_;

    kf_.predict();
}

Eigen::Matrix3d TargetPoseFusionNode::opticalToNedRotation() const
{
    Eigen::Matrix3d rotation;
    rotation << 0.0, -1.0, 0.0,
                1.0,  0.0, 0.0,
                0.0,  0.0, 1.0;
    return rotation;
}

Eigen::Vector3d TargetPoseFusionNode::measurementOpticalToWorldPosition(
    const Eigen::Vector3d& opticalPosition) const
{
    const Eigen::Matrix3d opticalToNed = opticalToNedRotation();
    const Eigen::Vector3d cameraPositionNed = opticalToNed * opticalPosition;

    const Eigen::Vector3d cameraOffsetBody(
        cam_offset_x_,
        cam_offset_y_,
        cam_offset_z_);

    const Eigen::Matrix3d worldFromBody = vehicle_q_ned_.toRotationMatrix();

    return vehicle_pos_ned_ + worldFromBody * (cameraOffsetBody + cameraPositionNed);
}

Eigen::Quaterniond TargetPoseFusionNode::transformTagOrientationToWorld(
    const geometry_msgs::msg::Quaternion& quaternionMessage) const
{
    Eigen::Quaterniond targetOrientationOptical(
        quaternionMessage.w,
        quaternionMessage.x,
        quaternionMessage.y,
        quaternionMessage.z);

    if (targetOrientationOptical.norm() > 1e-9)
    {
        targetOrientationOptical.normalize();
    }

    const Eigen::Quaterniond opticalToNedQuaternion(opticalToNedRotation());
    Eigen::Quaterniond targetOrientationWorld =
        vehicle_q_ned_ * opticalToNedQuaternion * targetOrientationOptical;

    targetOrientationWorld.normalize();
    return targetOrientationWorld;
}

void TargetPoseFusionNode::publishEstimatedState(const rclcpp::Time& nowTimestamp, double dt)
{
    (void)dt;

    const Eigen::Vector3d targetWorldPosition(
        kf_.statePost.at<double>(0, 0),
        kf_.statePost.at<double>(1, 0),
        kf_.statePost.at<double>(2, 0));

    const Eigen::Vector3d targetWorldVelocity(
        kf_.statePost.at<double>(3, 0),
        kf_.statePost.at<double>(4, 0),
        kf_.statePost.at<double>(5, 0));

    const Eigen::Vector3d relativePositionRaw =
        last_measurement_world_ - vehicle_pos_ned_;

    const Eigen::Vector3d relativePositionPredicted =
        targetWorldPosition - vehicle_pos_ned_;

    const Eigen::Vector3d relativeVelocity =
        targetWorldVelocity - vehicle_vel_ned_;

    const Eigen::Quaterniond targetOrientationWorld =
        transformTagOrientationToWorld(last_orientation_);

    geometry_msgs::msg::PoseStamped targetWorldPoseMsg;
    targetWorldPoseMsg.header.stamp = nowTimestamp;
    targetWorldPoseMsg.header.frame_id = kFrameId;
    targetWorldPoseMsg.pose.position.x = targetWorldPosition.x();
    targetWorldPoseMsg.pose.position.y = targetWorldPosition.y();
    targetWorldPoseMsg.pose.position.z = targetWorldPosition.z();
    targetWorldPoseMsg.pose.orientation.w = targetOrientationWorld.w();
    targetWorldPoseMsg.pose.orientation.x = targetOrientationWorld.x();
    targetWorldPoseMsg.pose.orientation.y = targetOrientationWorld.y();
    targetWorldPoseMsg.pose.orientation.z = targetOrientationWorld.z();
    target_pose_world_pub_->publish(targetWorldPoseMsg);

    geometry_msgs::msg::PoseStamped relativePositionRawMsg;
    relativePositionRawMsg.header.stamp = nowTimestamp;
    relativePositionRawMsg.header.frame_id = kFrameId;
    relativePositionRawMsg.pose.position.x = relativePositionRaw.x();
    relativePositionRawMsg.pose.position.y = relativePositionRaw.y();
    relativePositionRawMsg.pose.position.z = relativePositionRaw.z();
    relativePositionRawMsg.pose.orientation.w = 1.0;
    relativePositionRawMsg.pose.orientation.x = 0.0;
    relativePositionRawMsg.pose.orientation.y = 0.0;
    relativePositionRawMsg.pose.orientation.z = 0.0;
    target_error_pred_raw_pub_->publish(relativePositionRawMsg);

    geometry_msgs::msg::PoseStamped relativePositionPredictedMsg;
    relativePositionPredictedMsg.header.stamp = nowTimestamp;
    relativePositionPredictedMsg.header.frame_id = kFrameId;
    relativePositionPredictedMsg.pose.position.x = relativePositionPredicted.x();
    relativePositionPredictedMsg.pose.position.y = relativePositionPredicted.y();
    relativePositionPredictedMsg.pose.position.z = relativePositionPredicted.z();
    relativePositionPredictedMsg.pose.orientation.w = targetOrientationWorld.w();
    relativePositionPredictedMsg.pose.orientation.x = targetOrientationWorld.x();
    relativePositionPredictedMsg.pose.orientation.y = targetOrientationWorld.y();
    relativePositionPredictedMsg.pose.orientation.z = targetOrientationWorld.z();
    target_error_pred_fusion_pub_->publish(relativePositionPredictedMsg);

    geometry_msgs::msg::PoseStamped relativeVelocityMsg;
    relativeVelocityMsg.header.stamp = nowTimestamp;
    relativeVelocityMsg.header.frame_id = kFrameId;
    relativeVelocityMsg.pose.position.x = relativeVelocity.x();
    relativeVelocityMsg.pose.position.y = relativeVelocity.y();
    relativeVelocityMsg.pose.position.z = relativeVelocity.z();
    relativeVelocityMsg.pose.orientation.w = 1.0;
    relativeVelocityMsg.pose.orientation.x = 0.0;
    relativeVelocityMsg.pose.orientation.y = 0.0;
    relativeVelocityMsg.pose.orientation.z = 0.0;
    target_rel_vel_pub_->publish(relativeVelocityMsg);
}

void TargetPoseFusionNode::publishZero(const rclcpp::Time& nowTimestamp)
{
    geometry_msgs::msg::PoseStamped zeroPoseMsg;
    zeroPoseMsg.header.stamp = nowTimestamp;
    zeroPoseMsg.header.frame_id = kFrameId;
    zeroPoseMsg.pose.position.x = 0.0;
    zeroPoseMsg.pose.position.y = 0.0;
    zeroPoseMsg.pose.position.z = 0.0;
    zeroPoseMsg.pose.orientation.w = 1.0;
    zeroPoseMsg.pose.orientation.x = 0.0;
    zeroPoseMsg.pose.orientation.y = 0.0;
    zeroPoseMsg.pose.orientation.z = 0.0;

    geometry_msgs::msg::PoseStamped zeroVelocityMsg;
    zeroVelocityMsg.header.stamp = nowTimestamp;
    zeroVelocityMsg.header.frame_id = kFrameId;
    zeroVelocityMsg.pose.position.x = 0.0;
    zeroVelocityMsg.pose.position.y = 0.0;
    zeroVelocityMsg.pose.position.z = 0.0;
    zeroVelocityMsg.pose.orientation.w = 1.0;
    zeroVelocityMsg.pose.orientation.x = 0.0;
    zeroVelocityMsg.pose.orientation.y = 0.0;
    zeroVelocityMsg.pose.orientation.z = 0.0;

    target_pose_world_pub_->publish(zeroPoseMsg);
    target_error_pred_raw_pub_->publish(zeroPoseMsg);
    target_error_pred_fusion_pub_->publish(zeroPoseMsg);
    target_rel_vel_pub_->publish(zeroVelocityMsg);
}

void TargetPoseFusionNode::resetState()
{
    initialized_ = false;

    kf_.statePost = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(STATE_SIZE, 1, CV_64F);

    kf_.errorCovPost = cv::Mat::eye(STATE_SIZE, STATE_SIZE, CV_64F);
    kf_.errorCovPost.at<double>(3, 3) = 10.0;
    kf_.errorCovPost.at<double>(4, 4) = 10.0;
    kf_.errorCovPost.at<double>(5, 5) = 10.0;

    last_measurement_world_.setZero();
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TargetPoseFusionNode>());
    rclcpp::shutdown();
    return 0;
}