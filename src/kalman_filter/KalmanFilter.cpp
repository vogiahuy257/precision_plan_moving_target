#include "KalmanFilter.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <std_msgs/msg/string.hpp>

namespace
{
void publishKalmanTiming(
    const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr &pub,
    const std::string &stage,
    double imageStampSec,
    double rxNowSec,
    double procStartSec,
    double procEndSec,
    double pubNowSec,
    double measurementDtSec,
    double predictDtSec)
{
    if (!pub)
    {
        return;
    }

    std_msgs::msg::String msg;
    std::ostringstream ss;

    const double processingDt =
        (procEndSec >= 0.0 && procStartSec >= 0.0) ? (procEndSec - procStartSec) : -1.0;

    const double sendDt =
        (pubNowSec >= 0.0 && procEndSec >= 0.0) ? (pubNowSec - procEndSec) : -1.0;

    ss << std::fixed << std::setprecision(6)
       << "{"
       << "\"node\":\"kalman\","
       << "\"stage\":\"" << stage << "\","
       << "\"image_stamp\":" << imageStampSec << ","
       << "\"rx_now\":" << rxNowSec << ","
       << "\"proc_start\":" << procStartSec << ","
       << "\"proc_end\":" << procEndSec << ","
       << "\"pub_now\":" << pubNowSec << ","
       << "\"processing_dt\":" << processingDt << ","
       << "\"send_dt\":" << sendDt << ","
       << "\"measurement_dt\":" << measurementDtSec << ","
       << "\"predict_dt\":" << predictDtSec
       << "}";

    msg.data = ss.str();
    pub->publish(msg);
}
}

KalmanFilterNode::KalmanFilterNode()
    : Node("kalman_filter_node")
{
    declareParameters();
    loadParameters();

    const auto subQos = rclcpp::QoS(1).best_effort();
    const auto pubQos = rclcpp::QoS(1).best_effort();

    poseSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        inputTargetPoseTopic_,
        subQos,
        std::bind(&KalmanFilterNode::poseCallback, this, std::placeholders::_1));

    resetSub_ = create_subscription<std_msgs::msg::String>(
        resetCommandTopic_,
        subQos,
        std::bind(&KalmanFilterNode::resetCallback, this, std::placeholders::_1));

    validSub_ = create_subscription<std_msgs::msg::Bool>(
        targetValidTopic_,
        subQos,
        std::bind(&KalmanFilterNode::validCallback, this, std::placeholders::_1));

    vehicleOdomSub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        vehicleOdometryTopic_,
        subQos,
        std::bind(&KalmanFilterNode::vehicleOdometryCallback, this, std::placeholders::_1));

    targetPoseRawPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        targetPoseRawTopic_,
        pubQos);

    targetPoseFilteredPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        targetPoseFilteredTopic_,
        pubQos);

    targetVelocityFilteredPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
        targetVelocityTopic_,
        pubQos);

    debugDtPub_ = create_publisher<std_msgs::msg::String>(
        "/debug_dt/kalman",
        rclcpp::QoS(10).best_effort());

    initKalman();

    RCLCPP_INFO(
        get_logger(),
        "Loaded topics | input_pose=%s, reset=%s, valid=%s, odom=%s, raw=%s, filtered=%s, vel=%s, frame_id=%s",
        inputTargetPoseTopic_.c_str(),
        resetCommandTopic_.c_str(),
        targetValidTopic_.c_str(),
        vehicleOdometryTopic_.c_str(),
        targetPoseRawTopic_.c_str(),
        targetPoseFilteredTopic_.c_str(),
        targetVelocityTopic_.c_str(),
        outputFrameId_.c_str());

    RCLCPP_INFO(
        get_logger(),
        "Loaded params | q_acc=[%.6f %.6f %.6f], q_bias=[%.6f %.6f], r_pos=[%.6f %.6f %.6f], r_vel=[%.6f %.6f], cam_offset=[%.3f %.3f %.3f], max_predict_dt=%.3f, stale_threshold=%.3f, negative_dt_tol=%.3f, rel_vel_gain=%.3f, rel_pos_gain=%.3f, vel_meas_limit=%.3f",
        qAccX_,
        qAccY_,
        qAccZ_,
        qBiasVx_,
        qBiasVy_,
        rPosX_,
        rPosY_,
        rPosZ_,
        rVelX_,
        rVelY_,
        camOffsetX_,
        camOffsetY_,
        camOffsetZ_,
        maxPredictDt_,
        staleMeasurementThresholdSec_,
        smallNegativeDtToleranceSec_,
        biasFromRelVelGain_,
        biasFromRelPosGain_,
        biasLimit_);
}

void KalmanFilterNode::declareParameters()
{
    declare_parameter<std::string>("topics.input_target_pose", "/Aruco/target_pose_FRD");
    declare_parameter<std::string>("topics.reset_command", "/Aruco/target_state");
    declare_parameter<std::string>("topics.target_valid", "/target_valid");
    declare_parameter<std::string>("topics.vehicle_odometry", "/fmu/out/vehicle_odometry");

    declare_parameter<std::string>("topics.target_pose_raw", "/KalmanFilter/target_pose_NED");
    declare_parameter<std::string>("topics.target_pose_filtered", "/KalmanFilter/target_pose_est_NED");
    declare_parameter<std::string>("topics.target_velocity_filtered", "/KalmanFilter/target_velocity_est_NED");

    declare_parameter<std::string>("frame_id", "map");

    declare_parameter<double>("q_acc_x", 0.02);
    declare_parameter<double>("q_acc_y", 0.02);
    declare_parameter<double>("q_acc_z", 0.0010);

    declare_parameter<double>("q_bias_vx", 0.0001);
    declare_parameter<double>("q_bias_vy", 0.0001);

    declare_parameter<double>("r_pos_x", 0.0008);
    declare_parameter<double>("r_pos_y", 0.0008);
    declare_parameter<double>("r_pos_z", 0.0040);

    declare_parameter<double>("r_vel_x", 0.2);
    declare_parameter<double>("r_vel_y", 0.2);

    declare_parameter<double>("cam_offset_x", 0.0);
    declare_parameter<double>("cam_offset_y", 0.0);
    declare_parameter<double>("cam_offset_z", -0.1);

    declare_parameter<double>("max_predict_dt", 0.15);
    declare_parameter<double>("stale_measurement_threshold_sec", 0.2);
    declare_parameter<double>("small_negative_dt_tolerance_sec", 0.02);

    declare_parameter<double>("bias_from_rel_vel_gain", 1.7);
    declare_parameter<double>("bias_from_rel_pos_gain", 0.18);
    declare_parameter<double>("bias_limit", 5.0);
}

void KalmanFilterNode::loadParameters()
{
    get_parameter("topics.input_target_pose", inputTargetPoseTopic_);
    get_parameter("topics.reset_command", resetCommandTopic_);
    get_parameter("topics.target_valid", targetValidTopic_);
    get_parameter("topics.vehicle_odometry", vehicleOdometryTopic_);

    get_parameter("topics.target_pose_raw", targetPoseRawTopic_);
    get_parameter("topics.target_pose_filtered", targetPoseFilteredTopic_);
    get_parameter("topics.target_velocity_filtered", targetVelocityTopic_);

    get_parameter("frame_id", outputFrameId_);

    get_parameter("q_acc_x", qAccX_);
    get_parameter("q_acc_y", qAccY_);
    get_parameter("q_acc_z", qAccZ_);

    get_parameter("q_bias_vx", qBiasVx_);
    get_parameter("q_bias_vy", qBiasVy_);

    get_parameter("r_pos_x", rPosX_);
    get_parameter("r_pos_y", rPosY_);
    get_parameter("r_pos_z", rPosZ_);

    get_parameter("r_vel_x", rVelX_);
    get_parameter("r_vel_y", rVelY_);

    get_parameter("cam_offset_x", camOffsetX_);
    get_parameter("cam_offset_y", camOffsetY_);
    get_parameter("cam_offset_z", camOffsetZ_);

    get_parameter("max_predict_dt", maxPredictDt_);
    get_parameter("stale_measurement_threshold_sec", staleMeasurementThresholdSec_);
    get_parameter("small_negative_dt_tolerance_sec", smallNegativeDtToleranceSec_);

    get_parameter("bias_from_rel_vel_gain", biasFromRelVelGain_);
    get_parameter("bias_from_rel_pos_gain", biasFromRelPosGain_);
    get_parameter("bias_limit", biasLimit_);
}

void KalmanFilterNode::initKalman()
{
    kf_ = cv::KalmanFilter(stateSize, measurementSize, 0, CV_64F);

    kf_.transitionMatrix = cv::Mat::eye(stateSize, stateSize, CV_64F);

    kf_.measurementMatrix = cv::Mat::zeros(measurementSize, stateSize, CV_64F);

    // position measurement
    kf_.measurementMatrix.at<double>(0, IDX_PX) = 1.0;
    kf_.measurementMatrix.at<double>(1, IDX_PY) = 1.0;
    kf_.measurementMatrix.at<double>(2, IDX_PZ) = 1.0;

    // velocity pseudo-measurement: vx_out = vx + bvx, vy_out = vy + bvy
    kf_.measurementMatrix.at<double>(3, IDX_VX) = 1.0;
    kf_.measurementMatrix.at<double>(3, IDX_BVX) = 1.0;

    kf_.measurementMatrix.at<double>(4, IDX_VY) = 1.0;
    kf_.measurementMatrix.at<double>(4, IDX_BVY) = 1.0;

    kf_.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_64F);

    kf_.measurementNoiseCov = cv::Mat::eye(measurementSize, measurementSize, CV_64F);
    kf_.measurementNoiseCov.at<double>(0, 0) = rPosX_;
    kf_.measurementNoiseCov.at<double>(1, 1) = rPosY_;
    kf_.measurementNoiseCov.at<double>(2, 2) = rPosZ_;
    kf_.measurementNoiseCov.at<double>(3, 3) = rVelX_;
    kf_.measurementNoiseCov.at<double>(4, 4) = rVelY_;

    kf_.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_64F);
    kf_.errorCovPost.at<double>(IDX_VX, IDX_VX) = 10.0;
    kf_.errorCovPost.at<double>(IDX_VY, IDX_VY) = 10.0;
    kf_.errorCovPost.at<double>(IDX_VZ, IDX_VZ) = 10.0;
    kf_.errorCovPost.at<double>(IDX_BVX, IDX_BVX) = 2.0;
    kf_.errorCovPost.at<double>(IDX_BVY, IDX_BVY) = 2.0;

    kf_.statePost = cv::Mat::zeros(stateSize, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(stateSize, 1, CV_64F);
}

double KalmanFilterNode::sanitizeDt(double dt) const
{
    if (dt <= 0.0)
    {
        return 0.0;
    }

    return dt;
}

void KalmanFilterNode::vehicleOdometryCallback(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    vehicleQned_ = Eigen::Quaterniond(
        static_cast<double>(msg->q[0]),
        static_cast<double>(msg->q[1]),
        static_cast<double>(msg->q[2]),
        static_cast<double>(msg->q[3]));

    if (vehicleQned_.norm() <= 1e-9)
    {
        return;
    }

    vehicleQned_.normalize();

    vehiclePosNed_.x() = static_cast<double>(msg->position[0]);
    vehiclePosNed_.y() = static_cast<double>(msg->position[1]);
    vehiclePosNed_.z() = static_cast<double>(msg->position[2]);

    vehicleVelNed_.x() = static_cast<double>(msg->velocity[0]);
    vehicleVelNed_.y() = static_cast<double>(msg->velocity[1]);
    vehicleVelNed_.z() = static_cast<double>(msg->velocity[2]);

    vehicleOdomValid_ = true;
}

void KalmanFilterNode::resetCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (msg->data == "RESET")
    {
        resetState();
        forceZero_.store(true, std::memory_order_relaxed);
        publishZero(now());
        return;
    }

    if (msg->data == "ACTIVE")
    {
        forceZero_.store(false, std::memory_order_relaxed);
    }
}

void KalmanFilterNode::validCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    if (msg->data)
    {
        forceZero_.store(false, std::memory_order_relaxed);
    }
}

void KalmanFilterNode::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!vehicleOdomValid_)
    {
        return;
    }

    if (forceZero_.load(std::memory_order_relaxed))
    {
        publishZero(now());
        return;
    }

    rclcpp::Time sourceTimestamp = msg->header.stamp;
    if (sourceTimestamp.nanoseconds() == 0)
    {
        sourceTimestamp = now();
    }

    const rclcpp::Time rxNow = now();
    const rclcpp::Time procStart = now();

    lastOrientation_ = msg->pose.orientation;

    const Eigen::Vector3d measurementOptical(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    const Eigen::Vector3d measurementWorld =
        measurementOpticalToWorldPosition(measurementOptical);

    lastMeasurementWorld_ = measurementWorld;

    if (!initialized_)
    {
        kf_.statePost.at<double>(IDX_PX, 0) = measurementWorld.x();
        kf_.statePost.at<double>(IDX_PY, 0) = measurementWorld.y();
        kf_.statePost.at<double>(IDX_PZ, 0) = measurementWorld.z();

        kf_.statePost.at<double>(IDX_VX, 0) = 0.0;
        kf_.statePost.at<double>(IDX_VY, 0) = 0.0;
        kf_.statePost.at<double>(IDX_VZ, 0) = 0.0;

        kf_.statePost.at<double>(IDX_BVX, 0) = 0.0;
        kf_.statePost.at<double>(IDX_BVY, 0) = 0.0;

        lastRelativeWorld_ = measurementWorld - vehiclePosNed_;
        lastRelativeWorldValid_ = true;
        lastMeasurementWorldSync_ = measurementWorld;

        kf_.statePre = kf_.statePost.clone();
        initialized_ = true;

        lastMeasurementTimestamp_ = sourceTimestamp;
        lastPredictTimestamp_ = rxNow;

        const rclcpp::Time procEnd = now();
        const rclcpp::Time pubNow = now();

        publishKalmanTiming(
            debugDtPub_,
            "PUB",
            sourceTimestamp.seconds(),
            rxNow.seconds(),
            procStart.seconds(),
            procEnd.seconds(),
            pubNow.seconds(),
            0.0,
            0.0);

        publishEstimatedState(pubNow);
        return;
    }

    double rawMeasurementDt = (sourceTimestamp - lastMeasurementTimestamp_).seconds();

    if (rawMeasurementDt < 0.0 && rawMeasurementDt >= -smallNegativeDtToleranceSec_)
    {
        rawMeasurementDt = 0.0;
    }

    if (rawMeasurementDt < 0.0)
    {
        const rclcpp::Time procEnd = now();

        publishKalmanTiming(
            debugDtPub_,
            "DROP_OOS",
            sourceTimestamp.seconds(),
            rxNow.seconds(),
            procStart.seconds(),
            procEnd.seconds(),
            -1.0,
            rawMeasurementDt,
            -1.0);
        return;
    }

    const double measurementDt = sanitizeDt(rawMeasurementDt);

    double rawPredictDt = (rxNow - lastPredictTimestamp_).seconds();

    if (rawPredictDt < 0.0 && rawPredictDt >= -smallNegativeDtToleranceSec_)
    {
        rawPredictDt = 0.0;
    }

    if (rawPredictDt < 0.0)
    {
        rawPredictDt = 0.0;
    }

    const double predictDt = sanitizeDt(rawPredictDt);

    if (predictDt > 0.0)
    {
        predict(predictDt);
    }

    const Eigen::Vector3d measurementWorldSync =
        syncMeasurementPositionToCurrentTime(measurementWorld, sourceTimestamp);

    lastMeasurementWorldSync_ = measurementWorldSync;

    double targetVelMeasX = 0.0;
    double targetVelMeasY = 0.0;
    const bool hasVelocityMeasurement =
        computeVelocityMeasurementFromRelativeMotion(
            measurementWorld,
            measurementDt,
            targetVelMeasX,
            targetVelMeasY);

    cv::Mat measurement(measurementSize, 1, CV_64F);
    measurement.at<double>(0, 0) = measurementWorldSync.x();
    measurement.at<double>(1, 0) = measurementWorldSync.y();
    measurement.at<double>(2, 0) = measurementWorldSync.z();

    if (hasVelocityMeasurement)
    {
        measurement.at<double>(3, 0) = targetVelMeasX;
        measurement.at<double>(4, 0) = targetVelMeasY;
    }
    else
    {
        measurement.at<double>(3, 0) =
            kf_.statePost.at<double>(IDX_VX, 0) +
            kf_.statePost.at<double>(IDX_BVX, 0);

        measurement.at<double>(4, 0) =
            kf_.statePost.at<double>(IDX_VY, 0) +
            kf_.statePost.at<double>(IDX_BVY, 0);
    }

    kf_.correct(measurement);

    lastMeasurementTimestamp_ = sourceTimestamp;
    lastPredictTimestamp_ = rxNow;

    const rclcpp::Time procEnd = now();
    const rclcpp::Time pubNow = now();

    publishKalmanTiming(
        debugDtPub_,
        "PUB",
        sourceTimestamp.seconds(),
        rxNow.seconds(),
        procStart.seconds(),
        procEnd.seconds(),
        pubNow.seconds(),
        measurementDt,
        predictDt);

    publishEstimatedState(pubNow);
}

void KalmanFilterNode::predict(double dt)
{
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    kf_.transitionMatrix = cv::Mat::eye(stateSize, stateSize, CV_64F);

    kf_.transitionMatrix.at<double>(IDX_PX, IDX_VX) = dt;
    kf_.transitionMatrix.at<double>(IDX_PX, IDX_BVX) = dt;

    kf_.transitionMatrix.at<double>(IDX_PY, IDX_VY) = dt;
    kf_.transitionMatrix.at<double>(IDX_PY, IDX_BVY) = dt;

    kf_.transitionMatrix.at<double>(IDX_PZ, IDX_VZ) = dt;

    kf_.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_64F);

    kf_.processNoiseCov.at<double>(IDX_PX, IDX_PX) = 0.25 * dt4 * qAccX_;
    kf_.processNoiseCov.at<double>(IDX_PX, IDX_VX) = 0.5 * dt3 * qAccX_;
    kf_.processNoiseCov.at<double>(IDX_VX, IDX_PX) = 0.5 * dt3 * qAccX_;
    kf_.processNoiseCov.at<double>(IDX_VX, IDX_VX) = dt2 * qAccX_;

    kf_.processNoiseCov.at<double>(IDX_PY, IDX_PY) = 0.25 * dt4 * qAccY_;
    kf_.processNoiseCov.at<double>(IDX_PY, IDX_VY) = 0.5 * dt3 * qAccY_;
    kf_.processNoiseCov.at<double>(IDX_VY, IDX_PY) = 0.5 * dt3 * qAccY_;
    kf_.processNoiseCov.at<double>(IDX_VY, IDX_VY) = dt2 * qAccY_;

    kf_.processNoiseCov.at<double>(IDX_PZ, IDX_PZ) = 0.25 * dt4 * qAccZ_;
    kf_.processNoiseCov.at<double>(IDX_PZ, IDX_VZ) = 0.5 * dt3 * qAccZ_;
    kf_.processNoiseCov.at<double>(IDX_VZ, IDX_PZ) = 0.5 * dt3 * qAccZ_;
    kf_.processNoiseCov.at<double>(IDX_VZ, IDX_VZ) = dt2 * qAccZ_;

    kf_.processNoiseCov.at<double>(IDX_BVX, IDX_BVX) = dt * qBiasVx_;
    kf_.processNoiseCov.at<double>(IDX_BVY, IDX_BVY) = dt * qBiasVy_;

    kf_.predict();
}

bool KalmanFilterNode::computeVelocityMeasurementFromRelativeMotion(
    const Eigen::Vector3d &measurementWorld,
    double dt,
    double &targetVelMeasX,
    double &targetVelMeasY)
{
    const Eigen::Vector3d relativeWorld = measurementWorld - vehiclePosNed_;

    if (dt <= 1e-6)
    {
        lastRelativeWorld_ = relativeWorld;
        lastRelativeWorldValid_ = true;
        return false;
    }

    if (!lastRelativeWorldValid_)
    {
        lastRelativeWorld_ = relativeWorld;
        lastRelativeWorldValid_ = true;
        return false;
    }

    const Eigen::Vector3d relativeVelocity =
        (relativeWorld - lastRelativeWorld_) / dt;

    targetVelMeasX =
        vehicleVelNed_.x() +
        biasFromRelVelGain_ * relativeVelocity.x() +
        biasFromRelPosGain_ * relativeWorld.x();

    targetVelMeasY =
        vehicleVelNed_.y() +
        biasFromRelVelGain_ * relativeVelocity.y() +
        biasFromRelPosGain_ * relativeWorld.y();

    targetVelMeasX = std::clamp(targetVelMeasX, -biasLimit_, biasLimit_);
    targetVelMeasY = std::clamp(targetVelMeasY, -biasLimit_, biasLimit_);

    lastRelativeWorld_ = relativeWorld;
    lastRelativeWorldValid_ = true;
    return true;
}

Eigen::Vector3d KalmanFilterNode::syncMeasurementPositionToCurrentTime(
    const Eigen::Vector3d &measurementWorld,
    const rclcpp::Time &measurementTimestamp) const
{
    double measurementAgeSec = (now() - measurementTimestamp).seconds();
    if (measurementAgeSec < 0.0)
    {
        measurementAgeSec = 0.0;
    }

    const Eigen::Vector3d targetVelocityEstimated(
        kf_.statePost.at<double>(IDX_VX, 0) + kf_.statePost.at<double>(IDX_BVX, 0),
        kf_.statePost.at<double>(IDX_VY, 0) + kf_.statePost.at<double>(IDX_BVY, 0),
        kf_.statePost.at<double>(IDX_VZ, 0));

    const Eigen::Vector3d relativeVelocityEstimated =
        targetVelocityEstimated - vehicleVelNed_;

    return measurementWorld + relativeVelocityEstimated * measurementAgeSec;
}

Eigen::Matrix3d KalmanFilterNode::opticalToNedRotation() const
{
    Eigen::Matrix3d rotation;
    rotation << 0.0, -1.0, 0.0,
                1.0,  0.0, 0.0,
                0.0,  0.0, 1.0;
    return rotation;
}

Eigen::Vector3d KalmanFilterNode::measurementOpticalToWorldPosition(
    const Eigen::Vector3d &opticalPosition) const
{
    const Eigen::Matrix3d opticalToNed = opticalToNedRotation();
    const Eigen::Vector3d cameraPositionNed = opticalToNed * opticalPosition;

    const Eigen::Vector3d cameraOffsetBody(
        camOffsetX_,
        camOffsetY_,
        camOffsetZ_);

    const Eigen::Matrix3d worldFromBody = vehicleQned_.toRotationMatrix();

    return vehiclePosNed_ + worldFromBody * (cameraOffsetBody + cameraPositionNed);
}

Eigen::Quaterniond KalmanFilterNode::transformTagOrientationToWorld(
    const geometry_msgs::msg::Quaternion &quaternionMessage) const
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
        vehicleQned_ * opticalToNedQuaternion * targetOrientationOptical;

    targetOrientationWorld.normalize();
    return targetOrientationWorld;
}

void KalmanFilterNode::publishEstimatedState(const rclcpp::Time &publishTimestamp)
{
    const Eigen::Vector3d targetWorldPositionFiltered(
        kf_.statePost.at<double>(IDX_PX, 0),
        kf_.statePost.at<double>(IDX_PY, 0),
        kf_.statePost.at<double>(IDX_PZ, 0));

    const double vx = kf_.statePost.at<double>(IDX_VX, 0);
    const double vy = kf_.statePost.at<double>(IDX_VY, 0);
    const double vz = kf_.statePost.at<double>(IDX_VZ, 0);
    const double bvx = kf_.statePost.at<double>(IDX_BVX, 0);
    const double bvy = kf_.statePost.at<double>(IDX_BVY, 0);

    const Eigen::Vector3d targetWorldVelocityFiltered(
        vx + bvx,
        vy + bvy,
        vz);

    const Eigen::Quaterniond targetOrientationWorld =
        transformTagOrientationToWorld(lastOrientation_);

    geometry_msgs::msg::PoseStamped rawPoseMsg;
    rawPoseMsg.header.stamp = lastMeasurementTimestamp_;
    rawPoseMsg.header.frame_id = outputFrameId_;
    rawPoseMsg.pose.position.x = lastMeasurementWorld_.x();
    rawPoseMsg.pose.position.y = lastMeasurementWorld_.y();
    rawPoseMsg.pose.position.z = lastMeasurementWorld_.z();
    rawPoseMsg.pose.orientation.w = targetOrientationWorld.w();
    rawPoseMsg.pose.orientation.x = targetOrientationWorld.x();
    rawPoseMsg.pose.orientation.y = targetOrientationWorld.y();
    rawPoseMsg.pose.orientation.z = targetOrientationWorld.z();
    targetPoseRawPub_->publish(rawPoseMsg);

    geometry_msgs::msg::PoseStamped filteredPoseMsg;
    filteredPoseMsg.header.stamp = publishTimestamp;
    filteredPoseMsg.header.frame_id = outputFrameId_;
    filteredPoseMsg.pose.position.x = targetWorldPositionFiltered.x();
    filteredPoseMsg.pose.position.y = targetWorldPositionFiltered.y();
    filteredPoseMsg.pose.position.z = targetWorldPositionFiltered.z();
    filteredPoseMsg.pose.orientation.w = targetOrientationWorld.w();
    filteredPoseMsg.pose.orientation.x = targetOrientationWorld.x();
    filteredPoseMsg.pose.orientation.y = targetOrientationWorld.y();
    filteredPoseMsg.pose.orientation.z = targetOrientationWorld.z();
    targetPoseFilteredPub_->publish(filteredPoseMsg);

    geometry_msgs::msg::PoseStamped filteredVelocityMsg;
    filteredVelocityMsg.header.stamp = publishTimestamp;
    filteredVelocityMsg.header.frame_id = outputFrameId_;
    filteredVelocityMsg.pose.position.x = targetWorldVelocityFiltered.x();
    filteredVelocityMsg.pose.position.y = targetWorldVelocityFiltered.y();
    filteredVelocityMsg.pose.position.z = targetWorldVelocityFiltered.z();
    filteredVelocityMsg.pose.orientation.w = 1.0;
    filteredVelocityMsg.pose.orientation.x = 0.0;
    filteredVelocityMsg.pose.orientation.y = 0.0;
    filteredVelocityMsg.pose.orientation.z = 0.0;
    targetVelocityFilteredPub_->publish(filteredVelocityMsg);
}

void KalmanFilterNode::publishZero(const rclcpp::Time &publishTimestamp)
{
    const Eigen::Quaterniond targetOrientationWorld =
        transformTagOrientationToWorld(lastOrientation_);

    const rclcpp::Time outputStamp =
        (lastMeasurementTimestamp_.nanoseconds() != 0) ? lastMeasurementTimestamp_ : publishTimestamp;

    geometry_msgs::msg::PoseStamped holdPoseMsg;
    holdPoseMsg.header.stamp = outputStamp;
    holdPoseMsg.header.frame_id = outputFrameId_;
    holdPoseMsg.pose.position.x = lastMeasurementWorld_.x();
    holdPoseMsg.pose.position.y = lastMeasurementWorld_.y();
    holdPoseMsg.pose.position.z = lastMeasurementWorld_.z();
    holdPoseMsg.pose.orientation.w = targetOrientationWorld.w();
    holdPoseMsg.pose.orientation.x = targetOrientationWorld.x();
    holdPoseMsg.pose.orientation.y = targetOrientationWorld.y();
    holdPoseMsg.pose.orientation.z = targetOrientationWorld.z();

    geometry_msgs::msg::PoseStamped zeroVelMsg;
    zeroVelMsg.header.stamp = outputStamp;
    zeroVelMsg.header.frame_id = outputFrameId_;
    zeroVelMsg.pose.position.x = 0.0;
    zeroVelMsg.pose.position.y = 0.0;
    zeroVelMsg.pose.position.z = 0.0;
    zeroVelMsg.pose.orientation.w = 1.0;
    zeroVelMsg.pose.orientation.x = 0.0;
    zeroVelMsg.pose.orientation.y = 0.0;
    zeroVelMsg.pose.orientation.z = 0.0;

    targetPoseRawPub_->publish(holdPoseMsg);
    targetPoseFilteredPub_->publish(holdPoseMsg);
    targetVelocityFilteredPub_->publish(zeroVelMsg);
}

void KalmanFilterNode::resetState()
{
    initialized_ = false;
    lastMeasurementTimestamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    lastPredictTimestamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    kf_.statePost = cv::Mat::zeros(stateSize, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(stateSize, 1, CV_64F);

    kf_.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_64F);
    kf_.errorCovPost.at<double>(IDX_VX, IDX_VX) = 10.0;
    kf_.errorCovPost.at<double>(IDX_VY, IDX_VY) = 10.0;
    kf_.errorCovPost.at<double>(IDX_VZ, IDX_VZ) = 10.0;
    kf_.errorCovPost.at<double>(IDX_BVX, IDX_BVX) = 2.0;
    kf_.errorCovPost.at<double>(IDX_BVY, IDX_BVY) = 2.0;

    lastMeasurementWorld_.setZero();
    lastMeasurementWorldSync_.setZero();
    lastRelativeWorld_.setZero();
    lastRelativeWorldValid_ = false;

    lastOrientation_.x = 0.0;
    lastOrientation_.y = 0.0;
    lastOrientation_.z = 0.0;
    lastOrientation_.w = 1.0;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanFilterNode>());
    rclcpp::shutdown();
    return 0;
}