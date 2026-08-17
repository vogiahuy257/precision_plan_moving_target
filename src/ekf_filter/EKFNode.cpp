#include "EKFNode.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace
{
constexpr const char *kPrefix = "[EKF-CTRA]";
}

EKFNode::EKFNode()
    : Node("ekf_filter_node")
{
    declareParameters();
    loadParameters();

    ekf_.setConfig(ekfConfig_);
    frameTransformer_.setCameraOffsetBody(cameraOffsetBody_);
    setupRosInterfaces();
    publishProcessNoise();

    RCLCPP_INFO(
        get_logger(),
        "%s started | q_a=%.4f q_omega=%.4f R_NE=(%.4f, %.4f) NIS=%.3f",
        kPrefix,
        ekfConfig_.qAcc,
        ekfConfig_.qTurnRate,
        ekfConfig_.rPosN,
        ekfConfig_.rPosE,
        ekfConfig_.nisThreshold);
}

void EKFNode::declareParameters()
{
    declare_parameter<std::string>("topics.input_target_pose", "/Aruco/target_pose_optical");
    declare_parameter<std::string>("topics.reset_command", "/Aruco/target_state");
    declare_parameter<std::string>("topics.vehicle_odometry", "/fmu/out/vehicle_odometry");
    declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position_v1");

    declare_parameter<std::string>("topics.relative_position_raw", "/EKF/target_pose_NED");
    declare_parameter<std::string>("topics.relative_position_predicted", "/EKF/target_pose_est_NED");
    declare_parameter<std::string>("topics.relative_velocity", "/EKF/target_velocity_est_NED");
    declare_parameter<std::string>("topics.motion", "/EKF/target_motion");
    declare_parameter<std::string>("topics.covariance", "/EKF/target_covariance_NE");
    declare_parameter<std::string>("topics.process_noise", "/EKF/process_noise");

    declare_parameter<std::string>("frame_id", "map");

    declare_parameter<double>("q_acc", 0.20);
    declare_parameter<double>("q_turn_rate", 0.20);
    declare_parameter<double>("r_pos_n", 0.008);
    declare_parameter<double>("r_pos_e", 0.008);
    declare_parameter<double>("nis_threshold", 9.21);
    declare_parameter<double>("turn_rate_epsilon", 0.001);

    declare_parameter<double>("init.min_speed_m_s", 0.10);
    declare_parameter<double>("init.motion_nis_threshold", 9.21);
    declare_parameter<int>("init.window_size", 15);
    declare_parameter<double>("init.var_p_n", 0.008);
    declare_parameter<double>("init.var_p_e", 0.008);
    declare_parameter<double>("init.var_speed", 1.0);
    declare_parameter<double>("init.var_heading", 3.0);
    declare_parameter<double>("init.var_acc", 1.0);
    declare_parameter<double>("init.var_turn_rate", 0.25);

    declare_parameter<double>("transform.camera_offset_x", 0.2);
    declare_parameter<double>("transform.camera_offset_y", 0.0);
    declare_parameter<double>("transform.camera_offset_z", -0.12);
}

void EKFNode::loadParameters()
{
    get_parameter("topics.input_target_pose", inputTargetPoseTopic_);
    get_parameter("topics.reset_command", resetCommandTopic_);
    get_parameter("topics.vehicle_odometry", vehicleOdometryTopic_);
    get_parameter("topics.vehicle_local_position", vehicleLocalPositionTopic_);
    get_parameter("topics.relative_position_raw", rawPoseTopic_);
    get_parameter("topics.relative_position_predicted", filteredPoseTopic_);
    get_parameter("topics.relative_velocity", velocityTopic_);
    get_parameter("topics.motion", motionTopic_);
    get_parameter("topics.covariance", covarianceTopic_);
    get_parameter("topics.process_noise", processNoiseTopic_);
    get_parameter("frame_id", frameId_);

    get_parameter("q_acc", ekfConfig_.qAcc);
    get_parameter("q_turn_rate", ekfConfig_.qTurnRate);
    get_parameter("r_pos_n", ekfConfig_.rPosN);
    get_parameter("r_pos_e", ekfConfig_.rPosE);
    get_parameter("nis_threshold", ekfConfig_.nisThreshold);
    get_parameter("turn_rate_epsilon", ekfConfig_.turnRateEps);

    get_parameter("init.min_speed_m_s", initMinSpeedMps_);
    get_parameter("init.motion_nis_threshold", initMotionNisThreshold_);
    get_parameter("init.window_size", initWindowSize_);

    double varPN = 0.008;
    double varPE = 0.008;
    double varSpeed = 1.0;
    double varHeading = 3.0;
    double varAcc = 1.0;
    double varTurnRate = 0.25;

    get_parameter("init.var_p_n", varPN);
    get_parameter("init.var_p_e", varPE);
    get_parameter("init.var_speed", varSpeed);
    get_parameter("init.var_heading", varHeading);
    get_parameter("init.var_acc", varAcc);
    get_parameter("init.var_turn_rate", varTurnRate);

    initialCovariance_.setZero();
    initialCovariance_(0, 0) = varPN;
    initialCovariance_(1, 1) = varPE;
    initialCovariance_(2, 2) = varSpeed;
    initialCovariance_(3, 3) = varHeading;
    initialCovariance_(4, 4) = varAcc;
    initialCovariance_(5, 5) = varTurnRate;

    get_parameter("transform.camera_offset_x", cameraOffsetBody_.x());
    get_parameter("transform.camera_offset_y", cameraOffsetBody_.y());
    get_parameter("transform.camera_offset_z", cameraOffsetBody_.z());

    if (!std::isfinite(initMinSpeedMps_) || initMinSpeedMps_ < 0.0 ||
        !std::isfinite(initMotionNisThreshold_) || initMotionNisThreshold_ <= 0.0 ||
        initWindowSize_ < 2 || initWindowSize_ > 60 ||
        !initialCovariance_.allFinite() ||
        (initialCovariance_.diagonal().array() <= 0.0).any() ||
        !cameraOffsetBody_.allFinite())
    {
        throw std::runtime_error("Invalid EKF node parameters");
    }
}

void EKFNode::setupRosInterfaces()
{
    const auto qos = rclcpp::QoS(1).best_effort();

    poseSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        inputTargetPoseTopic_,
        qos,
        std::bind(&EKFNode::poseCallback, this, std::placeholders::_1));

    const auto stateQos = rclcpp::QoS(10).reliable();
    resetSub_ = create_subscription<std_msgs::msg::String>(
        resetCommandTopic_,
        stateQos,
        std::bind(&EKFNode::resetCallback, this, std::placeholders::_1));

    vehicleOdomSub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        vehicleOdometryTopic_,
        qos,
        std::bind(&EKFNode::vehicleOdometryCallback, this, std::placeholders::_1));

    vehicleLocalPosSub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        vehicleLocalPositionTopic_,
        qos,
        std::bind(&EKFNode::vehicleLocalPositionCallback, this, std::placeholders::_1));

    rawPosePub_ = create_publisher<geometry_msgs::msg::PoseStamped>(rawPoseTopic_, qos);
    filteredPosePub_ = create_publisher<geometry_msgs::msg::PoseStamped>(filteredPoseTopic_, qos);
    velocityPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(velocityTopic_, qos);
    motionPub_ = create_publisher<std_msgs::msg::Float64MultiArray>(motionTopic_, qos);
    covariancePub_ = create_publisher<std_msgs::msg::Float64MultiArray>(covarianceTopic_, qos);

    const auto configQos = rclcpp::QoS(1).reliable().transient_local();
    processNoisePub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        processNoiseTopic_, configQos);
}

void EKFNode::resetFilter()
{
    ekf_.reset();
    bootstrapSamples_.clear();
    lastPredictTime_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());

    targetLost_ = false;
    lostEkfSnapshot_.reset();
    lostStateStamp_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
}

void EKFNode::vehicleOdometryCallback(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    if (msg == nullptr ||
        !std::isfinite(msg->q[0]) ||
        !std::isfinite(msg->q[1]) ||
        !std::isfinite(msg->q[2]) ||
        !std::isfinite(msg->q[3]))
    {
        return;
    }

    worldFromBody_ = Eigen::Quaterniond(
        static_cast<double>(msg->q[0]),
        static_cast<double>(msg->q[1]),
        static_cast<double>(msg->q[2]),
        static_cast<double>(msg->q[3]));

    if (worldFromBody_.norm() <= 1e-9)
    {
        vehicleAttitudeValid_ = false;
        return;
    }

    worldFromBody_.normalize();
    vehicleAttitudeValid_ = true;

    if (vehiclePositionValid_)
    {
        frameTransformer_.setVehicleState(vehiclePositionNed_, worldFromBody_);
    }
}

void EKFNode::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (msg == nullptr ||
        !std::isfinite(msg->x) ||
        !std::isfinite(msg->y) ||
        !std::isfinite(msg->z))
    {
        return;
    }

    vehiclePositionNed_ = Eigen::Vector3d(
        static_cast<double>(msg->x),
        static_cast<double>(msg->y),
        static_cast<double>(msg->z));
    vehiclePositionValid_ = true;

    if (vehicleAttitudeValid_)
    {
        frameTransformer_.setVehicleState(vehiclePositionNed_, worldFromBody_);
    }

    if (targetLost_)
    {
        publishLostPrediction(now());
    }
}

void EKFNode::resetCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

    if (msg->data == "RESET")
    {
        resetFilter();
        forceHold_ = true;
        publishHold(now());

        RCLCPP_INFO(get_logger(), "%s target RESET -> filter reset", kPrefix);
        return;
    }

    if (msg->data == "LOST")
    {
        if (!targetLost_)
        {
            targetLost_ = true;

            if (ekf_.initialized() && lastPredictTime_.nanoseconds() > 0)
            {
                lostEkfSnapshot_ = ekf_;
                lostStateStamp_ = lastPredictTime_;

                RCLCPP_INFO(
                    get_logger(),
                    "%s target LOST -> prediction-only",
                    kPrefix);
            }
        }
        return;
    }

    if (msg->data == "ACTIVE")
    {
        forceHold_ = false;
        targetLost_ = false;
        lostEkfSnapshot_.reset();
        lostStateStamp_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    }
}

void EKFNode::poseCallback(
    const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (msg == nullptr || !vehiclePositionValid_ || !vehicleAttitudeValid_)
    {
        return;
    }

    if (!std::isfinite(msg->pose.position.x) ||
        !std::isfinite(msg->pose.position.y) ||
        !std::isfinite(msg->pose.position.z))
    {
        return;
    }

    if (targetLost_)
    {
        targetLost_ = false;
        lostEkfSnapshot_.reset();
        lostStateStamp_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());

        RCLCPP_INFO(get_logger(), "%s target reacquired -> measurement update", kPrefix);
    }

    rclcpp::Time stamp = msg->header.stamp;
    if (stamp.nanoseconds() == 0)
    {
        stamp = now();
    }

    Eigen::Quaterniond opticalOrientation(
        msg->pose.orientation.w,
        msg->pose.orientation.x,
        msg->pose.orientation.y,
        msg->pose.orientation.z);
    if (!opticalOrientation.coeffs().allFinite())
    {
        opticalOrientation.setIdentity();
    }

    rawMeasurementNed_ = frameTransformer_.opticalPositionToWorld(
        Eigen::Vector3d(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z));

    rawOrientationNed_ = frameTransformer_.opticalOrientationToWorld(opticalOrientation);
    targetDown_ = rawMeasurementNed_.z();
    forceHold_ = false;

    publishRaw(stamp);

    if (!ekf_.initialized())
    {
        bootstrapFilter(stamp, rawMeasurementNed_, rawOrientationNed_);
        return;
    }

    double dtSec = 0.0;
    if (lastPredictTime_.nanoseconds() > 0)
    {
        dtSec = (stamp - lastPredictTime_).seconds();
    }

    if (!std::isfinite(dtSec) || dtSec < 0.0)
    {
        return;
    }

    if (dtSec > 1e-9)
    {
        ekf_.predict(dtSec);
    }
    lastPredictTime_ = stamp;

    CtraEkf::Vector2d measurement;
    measurement << rawMeasurementNed_.x(), rawMeasurementNed_.y();

    const CtraEkf::UpdateResult update = ekf_.correct(measurement);
    if (!update.accepted)
    {
        RCLCPP_WARN_THROTTLE(
            get_logger(),
            *get_clock(),
            1000,
            "%s measurement rejected by NIS | nis=%.3f threshold=%.3f",
            kPrefix,
            update.nis,
            ekfConfig_.nisThreshold);
        return;
    }

    publishEstimate(stamp);
}

bool EKFNode::bootstrapFilter(
    const rclcpp::Time &stamp,
    const Eigen::Vector3d &positionNed,
    const Eigen::Quaterniond &orientationNed)
{
    BootstrapData sample;
    sample.positionNed = positionNed;
    sample.orientationNed = orientationNed;
    sample.stamp = stamp;
    sample.valid = true;
    bootstrapSamples_.push_back(sample);

    while (static_cast<int>(bootstrapSamples_.size()) > initWindowSize_)
    {
        bootstrapSamples_.pop_front();
    }

    if (static_cast<int>(bootstrapSamples_.size()) < initWindowSize_)
    {
        return false;
    }

    const rclcpp::Time t0 = bootstrapSamples_.front().stamp;
    const double spanSec =
        (bootstrapSamples_.back().stamp - t0).seconds();

    if (!std::isfinite(spanSec) || spanSec <= 0.0)
    {
        return false;
    }

    // Least-squares velocity over the bootstrap window. This is much less
    // sensitive to frame-to-frame camera noise than a two-sample derivative.
    double meanT = 0.0;
    Eigen::Vector2d meanPos = Eigen::Vector2d::Zero();

    for (const BootstrapData &entry : bootstrapSamples_)
    {
        meanT += (entry.stamp - t0).seconds();
        meanPos += entry.positionNed.head<2>();
    }

    const double count = static_cast<double>(bootstrapSamples_.size());
    meanT /= count;
    meanPos /= count;

    double timeVariance = 0.0;
    Eigen::Vector2d timePositionCov = Eigen::Vector2d::Zero();

    for (const BootstrapData &entry : bootstrapSamples_)
    {
        const double centeredTime = (entry.stamp - t0).seconds() - meanT;
        timeVariance += centeredTime * centeredTime;
        timePositionCov +=
            centeredTime * (entry.positionNed.head<2>() - meanPos);
    }

    if (timeVariance <= 1e-9)
    {
        return false;
    }

    const Eigen::Vector2d velocityNE = timePositionCov / timeVariance;
    double speed = velocityNE.norm();
    double heading = yawFromQuaternion(orientationNed);

    // Significance test for the fitted velocity. With position-noise variances
    // R_N/R_E, the least-squares slope variances are R/Sxx.
    const double varVN = ekfConfig_.rPosN / timeVariance;
    const double varVE = ekfConfig_.rPosE / timeVariance;
    const double velocityNis =
        velocityNE.x() * velocityNE.x() / varVN +
        velocityNE.y() * velocityNE.y() / varVE;

    if (speed >= initMinSpeedMps_ &&
        std::isfinite(velocityNis) &&
        velocityNis > initMotionNisThreshold_)
    {
        heading = std::atan2(velocityNE.y(), velocityNE.x());
    }
    else
    {
        speed = 0.0;
    }

    ekf_.initialize(
        positionNed.x(),
        positionNed.y(),
        speed,
        heading,
        0.0,
        0.0,
        initialCovariance_);

    lastPredictTime_ = stamp;
    bootstrapSamples_.clear();

    RCLCPP_INFO(
        get_logger(),
        "%s initialized | p=(%.3f, %.3f) speed=%.3f psi=%.3f",
        kPrefix,
        positionNed.x(),
        positionNed.y(),
        speed,
        heading);

    publishEstimate(stamp);
    return true;
}

void EKFNode::publishRaw(const rclcpp::Time &stamp)
{
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frameId_;
    msg.pose.position.x = rawMeasurementNed_.x();
    msg.pose.position.y = rawMeasurementNed_.y();
    msg.pose.position.z = rawMeasurementNed_.z();
    msg.pose.orientation.w = rawOrientationNed_.w();
    msg.pose.orientation.x = rawOrientationNed_.x();
    msg.pose.orientation.y = rawOrientationNed_.y();
    msg.pose.orientation.z = rawOrientationNed_.z();
    rawPosePub_->publish(msg);
}

void EKFNode::publishEstimate(const rclcpp::Time &stamp)
{
    publishEstimateFromFilter(ekf_, stamp);
}

void EKFNode::publishEstimateFromFilter(
    const CtraEkf &filter,
    const rclcpp::Time &stamp)
{
    if (!filter.initialized())
    {
        return;
    }

    const CtraEkf::Vector6d &x = filter.state();
    const double speed = x(2);
    const double psi = x(3);
    const double vN = speed * std::cos(psi);
    const double vE = speed * std::sin(psi);
    const auto headingQuaternion = quaternionFromYaw(psi);

    geometry_msgs::msg::PoseStamped poseMsg;
    poseMsg.header.stamp = stamp;
    poseMsg.header.frame_id = frameId_;
    poseMsg.pose.position.x = x(0);
    poseMsg.pose.position.y = x(1);
    poseMsg.pose.position.z = targetDown_;
    poseMsg.pose.orientation = headingQuaternion;
    filteredPosePub_->publish(poseMsg);

    geometry_msgs::msg::PoseStamped velocityMsg;
    velocityMsg.header.stamp = stamp;
    velocityMsg.header.frame_id = frameId_;
    velocityMsg.pose.position.x = vN;
    velocityMsg.pose.position.y = vE;
    velocityMsg.pose.position.z = 0.0;
    velocityMsg.pose.orientation = headingQuaternion;
    velocityPub_->publish(velocityMsg);

    std_msgs::msg::Float64MultiArray motionMsg;
    motionMsg.data = {x(4), x(5)};
    motionPub_->publish(motionMsg);

    publishCovariance(filter.covariance());
}

void EKFNode::publishLostPrediction(const rclcpp::Time &stamp)
{
    if (!targetLost_ ||
        !lostEkfSnapshot_.initialized() ||
        lostStateStamp_.nanoseconds() == 0)
    {
        return;
    }

    const double dtSec = (stamp - lostStateStamp_).seconds();
    if (!std::isfinite(dtSec) || dtSec <= 0.0)
    {
        return;
    }

    // Predict a temporary copy from the last measurement state.
    // The main ekf_ stays untouched so reacquisition can predict directly
    // from the last measurement timestamp and then correct cleanly.
    CtraEkf predicted = lostEkfSnapshot_;
    predicted.predict(dtSec);

    publishEstimateFromFilter(predicted, stamp);

    const CtraEkf::Vector6d &x = predicted.state();
    RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "%s LOST prediction | dt=%.3f p=(%.3f, %.3f) v=%.3f psi=%.3f",
        kPrefix,
        dtSec,
        x(0),
        x(1),
        x(2),
        x(3));
}

void EKFNode::publishHold(const rclcpp::Time &stamp)
{
    const Eigen::Vector3d holdPosition =
        vehiclePositionValid_ ? vehiclePositionNed_ : Eigen::Vector3d::Zero();
    const auto q = quaternionFromYaw(0.0);

    geometry_msgs::msg::PoseStamped poseMsg;
    poseMsg.header.stamp = stamp;
    poseMsg.header.frame_id = frameId_;
    poseMsg.pose.position.x = holdPosition.x();
    poseMsg.pose.position.y = holdPosition.y();
    poseMsg.pose.position.z = holdPosition.z();
    poseMsg.pose.orientation = q;
    rawPosePub_->publish(poseMsg);
    filteredPosePub_->publish(poseMsg);

    geometry_msgs::msg::PoseStamped velocityMsg;
    velocityMsg.header = poseMsg.header;
    velocityMsg.pose.orientation = q;
    velocityPub_->publish(velocityMsg);

    std_msgs::msg::Float64MultiArray motionMsg;
    motionMsg.data = {0.0, 0.0};
    motionPub_->publish(motionMsg);

    CtraEkf::Matrix6d holdCovariance = CtraEkf::Matrix6d::Zero();
    holdCovariance.diagonal().setConstant(1e6);
    publishCovariance(holdCovariance);
}

void EKFNode::publishCovariance(const CtraEkf::Matrix6d &covariance)
{
    std_msgs::msg::Float64MultiArray msg;
    msg.data.resize(36);

    for (int row = 0; row < 6; ++row)
    {
        for (int col = 0; col < 6; ++col)
        {
            msg.data[static_cast<std::size_t>(row * 6 + col)] =
                covariance(row, col);
        }
    }

    covariancePub_->publish(msg);
}

void EKFNode::publishProcessNoise()
{
    if (!processNoisePub_)
    {
        return;
    }

    // CTRA process-noise parameters used by target_drop only for
    // future covariance propagation. They remain owned/tuned by EKF.
    // data[0] = q_acc, data[1] = q_turn_rate.
    std_msgs::msg::Float64MultiArray msg;
    msg.data = {ekfConfig_.qAcc, ekfConfig_.qTurnRate};
    processNoisePub_->publish(msg);
}

double EKFNode::yawFromQuaternion(const Eigen::Quaterniond &qInput)
{
    Eigen::Quaterniond q = qInput;
    if (q.norm() <= 1e-9)
    {
        return 0.0;
    }
    q.normalize();

    const double sinYaw = 2.0 * (q.w() * q.z() + q.x() * q.y());
    const double cosYaw = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    return std::atan2(sinYaw, cosYaw);
}

geometry_msgs::msg::Quaternion EKFNode::quaternionFromYaw(double yawRad)
{
    geometry_msgs::msg::Quaternion q;
    q.w = std::cos(0.5 * yawRad);
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(0.5 * yawRad);
    return q;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EKFNode>());
    rclcpp::shutdown();
    return 0;
}
