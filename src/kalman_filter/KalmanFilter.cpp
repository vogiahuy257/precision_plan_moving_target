#include "KalmanFilter.hpp"

#include <algorithm>
#include <cmath>

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

    initKalman();

    timer_ = create_wall_timer(
        std::chrono::milliseconds(33),
        std::bind(&KalmanFilterNode::processAndPublish, this));

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
        "Loaded params | q_acc=[%.6f %.6f %.6f], r_pos=[%.6f %.6f %.6f], cam_offset=[%.3f %.3f %.3f], max_predict_dt=%.3f, stale_threshold=%.3f, negative_dt_tol=%.3f",
        qAccX_,
        qAccY_,
        qAccZ_,
        rPosX_,
        rPosY_,
        rPosZ_,
        camOffsetX_,
        camOffsetY_,
        camOffsetZ_,
        maxPredictDt_,
        staleMeasurementThresholdSec_,
        smallNegativeDtToleranceSec_);
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

    declare_parameter<double>("q_acc_x", 0.0005);
    declare_parameter<double>("q_acc_y", 0.0005);
    declare_parameter<double>("q_acc_z", 0.0010);

    declare_parameter<double>("r_pos_x", 0.000025);
    declare_parameter<double>("r_pos_y", 0.000025);
    declare_parameter<double>("r_pos_z", 0.0040);

    declare_parameter<double>("cam_offset_x", 0.0);
    declare_parameter<double>("cam_offset_y", 0.0);
    declare_parameter<double>("cam_offset_z", -0.1);

    declare_parameter<double>("max_predict_dt", 0.1);
    declare_parameter<double>("stale_measurement_threshold_sec", 0.2);
    declare_parameter<double>("small_negative_dt_tolerance_sec", 0.02);
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
    get_parameter("max_predict_dt", maxPredictDt_);
    get_parameter("stale_measurement_threshold_sec", staleMeasurementThresholdSec_);
    get_parameter("small_negative_dt_tolerance_sec", smallNegativeDtToleranceSec_);
}

void KalmanFilterNode::initKalman()
{
    get_parameter("q_acc_x", qAccX_);
    get_parameter("q_acc_y", qAccY_);
    get_parameter("q_acc_z", qAccZ_);

    get_parameter("r_pos_x", rPosX_);
    get_parameter("r_pos_y", rPosY_);
    get_parameter("r_pos_z", rPosZ_);

    get_parameter("cam_offset_x", camOffsetX_);
    get_parameter("cam_offset_y", camOffsetY_);
    get_parameter("cam_offset_z", camOffsetZ_);

    kf_ = cv::KalmanFilter(stateSize, measurementSize, 0, CV_64F);

    kf_.transitionMatrix = cv::Mat::eye(stateSize, stateSize, CV_64F);

    kf_.measurementMatrix = cv::Mat::zeros(measurementSize, stateSize, CV_64F);
    kf_.measurementMatrix.at<double>(0, 0) = 1.0;
    kf_.measurementMatrix.at<double>(1, 1) = 1.0;
    kf_.measurementMatrix.at<double>(2, 2) = 1.0;

    kf_.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_64F);

    kf_.measurementNoiseCov = cv::Mat::eye(measurementSize, measurementSize, CV_64F);
    kf_.measurementNoiseCov.at<double>(0, 0) = rPosX_;
    kf_.measurementNoiseCov.at<double>(1, 1) = rPosY_;
    kf_.measurementNoiseCov.at<double>(2, 2) = rPosZ_;

    kf_.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_64F);
    kf_.errorCovPost.at<double>(3, 3) = 10.0;
    kf_.errorCovPost.at<double>(4, 4) = 10.0;
    kf_.errorCovPost.at<double>(5, 5) = 10.0;

    kf_.statePost = cv::Mat::zeros(stateSize, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(stateSize, 1, CV_64F);
}

double KalmanFilterNode::sanitizeDt(double dt) const
{
    if (dt <= 0.0)
    {
        return 0.0;
    }

    return std::min(dt, maxPredictDt_);
}

void KalmanFilterNode::vehicleOdometryCallback(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    vehicleQned_ = Eigen::Quaterniond(
        static_cast<double>(msg->q[0]),
        static_cast<double>(msg->q[1]),
        static_cast<double>(msg->q[2]),
        static_cast<double>(msg->q[3]));

    if (vehicleQned_.norm() > 1e-9)
    {
        vehicleQned_.normalize();
    }
    else
    {
        return;
    }

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

    rclcpp::Time measurementTimestamp = msg->header.stamp;
    if (measurementTimestamp.nanoseconds() == 0)
    {
        measurementTimestamp = now();
    }

    const double measurementAgeSec = (now() - measurementTimestamp).seconds();
    if (measurementAgeSec > staleMeasurementThresholdSec_)
    {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            2000,
            "Drop stale measurement: age=%.3f s > threshold=%.3f s",
            measurementAgeSec,
            staleMeasurementThresholdSec_);
        return;
    }

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
        kf_.statePost.at<double>(0, 0) = measurementWorld.x();
        kf_.statePost.at<double>(1, 0) = measurementWorld.y();
        kf_.statePost.at<double>(2, 0) = measurementWorld.z();
        kf_.statePost.at<double>(3, 0) = 0.0;
        kf_.statePost.at<double>(4, 0) = 0.0;
        kf_.statePost.at<double>(5, 0) = 0.0;

        kf_.statePre = kf_.statePost.clone();
        initialized_ = true;
        lastMeasurementTimestamp_ = measurementTimestamp;
        return;
    }

    double rawDt = (measurementTimestamp - lastMeasurementTimestamp_).seconds();

    if (rawDt < 0.0 && rawDt >= -smallNegativeDtToleranceSec_)
    {
        rawDt = 0.0;
    }

    if (rawDt < 0.0)
    {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            2000,
            "Out-of-sequence measurement ignored: dt=%.6f s",
            rawDt);
        return;
    }

    const double dt = sanitizeDt(rawDt);
    if (dt > 0.0)
    {
        predict(dt);
    }

    cv::Mat measurement(measurementSize, 1, CV_64F);
    measurement.at<double>(0, 0) = measurementWorld.x();
    measurement.at<double>(1, 0) = measurementWorld.y();
    measurement.at<double>(2, 0) = measurementWorld.z();

    kf_.correct(measurement);

    lastMeasurementTimestamp_ = measurementTimestamp;
}

void KalmanFilterNode::processAndPublish()
{
    const rclcpp::Time publishTimestamp = now();

    if (forceZero_.load(std::memory_order_relaxed))
    {
        publishZero(publishTimestamp);
        return;
    }

    if (!initialized_ || !vehicleOdomValid_)
    {
        return;
    }

    publishEstimatedState(publishTimestamp);
}

void KalmanFilterNode::predict(double dt)
{
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;

    kf_.transitionMatrix = cv::Mat::eye(stateSize, stateSize, CV_64F);
    kf_.transitionMatrix.at<double>(0, 3) = dt;
    kf_.transitionMatrix.at<double>(1, 4) = dt;
    kf_.transitionMatrix.at<double>(2, 5) = dt;

    kf_.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_64F);

    kf_.processNoiseCov.at<double>(0, 0) = 0.25 * dt4 * qAccX_;
    kf_.processNoiseCov.at<double>(0, 3) = 0.5 * dt3 * qAccX_;
    kf_.processNoiseCov.at<double>(3, 0) = 0.5 * dt3 * qAccX_;
    kf_.processNoiseCov.at<double>(3, 3) = dt2 * qAccX_;

    kf_.processNoiseCov.at<double>(1, 1) = 0.25 * dt4 * qAccY_;
    kf_.processNoiseCov.at<double>(1, 4) = 0.5 * dt3 * qAccY_;
    kf_.processNoiseCov.at<double>(4, 1) = 0.5 * dt3 * qAccY_;
    kf_.processNoiseCov.at<double>(4, 4) = dt2 * qAccY_;

    kf_.processNoiseCov.at<double>(2, 2) = 0.25 * dt4 * qAccZ_;
    kf_.processNoiseCov.at<double>(2, 5) = 0.5 * dt3 * qAccZ_;
    kf_.processNoiseCov.at<double>(5, 2) = 0.5 * dt3 * qAccZ_;
    kf_.processNoiseCov.at<double>(5, 5) = dt2 * qAccZ_;

    kf_.predict();
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
    (void)publishTimestamp;

    const Eigen::Vector3d targetWorldPositionFiltered(
        kf_.statePost.at<double>(0, 0),
        kf_.statePost.at<double>(1, 0),
        kf_.statePost.at<double>(2, 0));

    const Eigen::Vector3d targetWorldVelocityFiltered(
        kf_.statePost.at<double>(3, 0),
        kf_.statePost.at<double>(4, 0),
        kf_.statePost.at<double>(5, 0));

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
    filteredPoseMsg.header.stamp = lastMeasurementTimestamp_;
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
    filteredVelocityMsg.header.stamp = lastMeasurementTimestamp_;
    filteredVelocityMsg.header.frame_id = outputFrameId_;
    filteredVelocityMsg.pose.position.x = targetWorldVelocityFiltered.x();
    filteredVelocityMsg.pose.position.y = targetWorldVelocityFiltered.y();
    filteredVelocityMsg.pose.position.z = targetWorldVelocityFiltered.z();
    filteredVelocityMsg.pose.orientation.w = targetOrientationWorld.w();
    filteredVelocityMsg.pose.orientation.x = targetOrientationWorld.x();
    filteredVelocityMsg.pose.orientation.y = targetOrientationWorld.y();
    filteredVelocityMsg.pose.orientation.z = targetOrientationWorld.z();
    targetVelocityFilteredPub_->publish(filteredVelocityMsg);
}

void KalmanFilterNode::publishZero(const rclcpp::Time &publishTimestamp)
{
    geometry_msgs::msg::PoseStamped zeroPoseMsg;
    zeroPoseMsg.header.stamp = publishTimestamp;
    zeroPoseMsg.header.frame_id = outputFrameId_;
    zeroPoseMsg.pose.position.x = 0.0;
    zeroPoseMsg.pose.position.y = 0.0;
    zeroPoseMsg.pose.position.z = 0.0;
    zeroPoseMsg.pose.orientation.w = 1.0;
    zeroPoseMsg.pose.orientation.x = 0.0;
    zeroPoseMsg.pose.orientation.y = 0.0;
    zeroPoseMsg.pose.orientation.z = 0.0;

    targetPoseRawPub_->publish(zeroPoseMsg);
    targetPoseFilteredPub_->publish(zeroPoseMsg);
    targetVelocityFilteredPub_->publish(zeroPoseMsg);
}

void KalmanFilterNode::resetState()
{
    initialized_ = false;
    lastMeasurementTimestamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

    kf_.statePost = cv::Mat::zeros(stateSize, 1, CV_64F);
    kf_.statePre = cv::Mat::zeros(stateSize, 1, CV_64F);

    kf_.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_64F);
    kf_.errorCovPost.at<double>(3, 3) = 10.0;
    kf_.errorCovPost.at<double>(4, 4) = 10.0;
    kf_.errorCovPost.at<double>(5, 5) = 10.0;

    lastMeasurementWorld_.setZero();
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