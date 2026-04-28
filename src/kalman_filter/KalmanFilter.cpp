#include "KalmanFilter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace
{
constexpr const char *kKalmanNodePrefix = "[KalmanFilter-Node]";

#define KF_INFO(logger, fmt, ...) \
    RCLCPP_INFO(logger, "%s " fmt, kKalmanNodePrefix, ##__VA_ARGS__)

#define KF_WARN(logger, fmt, ...) \
    RCLCPP_WARN(logger, "%s " fmt, kKalmanNodePrefix, ##__VA_ARGS__)

#define KF_WARN_THROTTLE(logger, clock, duration_ms, fmt, ...) \
    RCLCPP_WARN_THROTTLE(logger, clock, duration_ms, "%s " fmt, kKalmanNodePrefix, ##__VA_ARGS__)

#define KF_ERROR(logger, fmt, ...) \
    RCLCPP_ERROR(logger, "%s " fmt, kKalmanNodePrefix, ##__VA_ARGS__)

#define KF_FATAL(logger, fmt, ...) \
    RCLCPP_FATAL(logger, "%s " fmt, kKalmanNodePrefix, ##__VA_ARGS__)

#define KF_DEBUG(logger, fmt, ...) \
    RCLCPP_DEBUG(logger, "%s " fmt, kKalmanNodePrefix, ##__VA_ARGS__)
}

/**
 * Khoi tao node KalmanFilter.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     - Khai bao va doc parameter
 *     - Khoi tao FrameTransformer va Kalman
 *     - Cau hinh DebugLogger
 *     - Tao subscriber, publisher va timer xu ly chinh
 *
 * Output:
 *     Node san sang nhan data va publish output.
 */
KalmanFilterNode::KalmanFilterNode()
    : Node("kalman_filter_node")
{
    try
    {
        declareParameters();
        loadParameters();
        initFrameTransformer();
        initKalman();

        setForceZeroReason("Kalman paused: startup");

        debugLogger_.configure(
            get_logger(),
            data_.config.debug.enabled,
            data_.config.debug.csvPath);

        const auto subQos = rclcpp::QoS(1).best_effort();
        const auto pubQos = rclcpp::QoS(1).best_effort();

        poseSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            data_.config.topics.inputTargetPoseTopic,
            subQos,
            std::bind(&KalmanFilterNode::poseCallback, this, std::placeholders::_1));

        resetSub_ = create_subscription<std_msgs::msg::String>(
            data_.config.topics.resetCommandTopic,
            subQos,
            std::bind(&KalmanFilterNode::resetCallback, this, std::placeholders::_1));

        validSub_ = create_subscription<std_msgs::msg::Bool>(
            data_.config.topics.targetValidTopic,
            subQos,
            std::bind(&KalmanFilterNode::validCallback, this, std::placeholders::_1));

        vehicleOdomSub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
            data_.config.topics.vehicleOdometryTopic,
            subQos,
            std::bind(&KalmanFilterNode::vehicleOdometryCallback, this, std::placeholders::_1));

        vehicleLocalPosSub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            data_.config.topics.vehicleLocalPositionTopic,
            subQos,
            std::bind(&KalmanFilterNode::vehicleLocalPositionCallback, this, std::placeholders::_1));

        targetPoseRawPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            data_.config.topics.relativePositionRawTopic,
            pubQos);

        targetPoseFilteredPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            data_.config.topics.relativePositionPredictedTopic,
            pubQos);

        targetRelVelPub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            data_.config.topics.relativeVelocityTopic,
            pubQos);

        data_.timing.lastPredictTime = rclcpp::Time(0, 0, get_clock()->get_clock_type());
        data_.timing.lastMeasurementTime = rclcpp::Time(0, 0, get_clock()->get_clock_type());

        KF_INFO(
            get_logger(),
            "Params loaded | node=%s debug=%s csv=%s timeout=%.2f input_pose=%s odom=%s local_pos=%s out_raw=%s out_filtered=%s out_vel=%s",
            this->get_name(),
            data_.config.debug.enabled ? "true" : "false",
            data_.config.debug.csvPath.c_str(),
            data_.config.poseTimeoutSec,
            data_.config.topics.inputTargetPoseTopic.c_str(),
            data_.config.topics.vehicleOdometryTopic.c_str(),
            data_.config.topics.vehicleLocalPositionTopic.c_str(),
            data_.config.topics.relativePositionRawTopic.c_str(),
            data_.config.topics.relativePositionPredictedTopic.c_str(),
            data_.config.topics.relativeVelocityTopic.c_str());

        KF_INFO(get_logger(), "KalmanFilterNode started");
    }
    catch (const std::exception &exception)
    {
        KF_FATAL(
            get_logger(),
            "KalmanFilterNode constructor failed: %s",
            exception.what());
        throw;
    }
    catch (...)
    {
        KF_FATAL(
            get_logger(),
            "KalmanFilterNode constructor failed: unknown exception");
        throw;
    }
}

/**
 * Khai bao toan bo parameter su dung trong node.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Khai bao nhom topic, frame_id, timeout, debug, noise va transform.
 *
 * Output:
 *     Parameter duoc dang ky de co the override bang yaml.
 */
void KalmanFilterNode::declareParameters()
{
    try
    {
        declare_parameter<std::string>("topics.input_target_pose", "/Aruco/target_pose_FRD");
        declare_parameter<std::string>("topics.reset_command", "/Aruco/target_state");
        declare_parameter<std::string>("topics.target_valid", "/target_valid");
        declare_parameter<std::string>("topics.vehicle_odometry", "/fmu/out/vehicle_odometry");
        declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position");
        declare_parameter<std::string>("topics.relative_position_raw", "/KalmanFilter/target_pose_NED");
        declare_parameter<std::string>("topics.relative_position_predicted", "/KalmanFilter/target_pose_est_NED");
        declare_parameter<std::string>("topics.relative_velocity", "/KalmanFilter/target_velocity_est_NED");

        declare_parameter<std::string>("frame_id", "map");
        declare_parameter<double>("pose_timeout_s", 3.0);

        declare_parameter<bool>("debug", false);
        declare_parameter<std::string>("debug_csv_path", "kalman_logs/");

        declare_parameter<double>("q_acc_x", 0.0002);
        declare_parameter<double>("q_acc_y", 0.0002);
        declare_parameter<double>("q_acc_z", 0.0010);

        declare_parameter<double>("r_pos_x", 0.0008);
        declare_parameter<double>("r_pos_y", 0.0008);
        declare_parameter<double>("r_pos_z", 0.0040);

        declare_parameter<bool>("dynamic_r.enable", true);
        declare_parameter<double>("dynamic_r.near_range_m", 0.7);
        declare_parameter<double>("dynamic_r.near_noise_gain", 0.08);
        declare_parameter<double>("dynamic_r.max_extra_r_xy", 0.20);
        declare_parameter<double>("dynamic_r.min_range_m", 0.1);

        declare_parameter<std::string>("transform.mount_mode", "belly_fixed_camera");
        declare_parameter<double>("transform.camera_offset_x", 0.0);
        declare_parameter<double>("transform.camera_offset_y", 0.0);
        declare_parameter<double>("transform.camera_offset_z", -0.1);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "declareParameters failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "declareParameters failed: unknown exception");
        throw;
    }
}

/**
 * Doc parameter tu ROS parameter server vao data trung tam.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Lay cac gia tri param va luu vao data_.config.
 *
 * Output:
 *     data_.config duoc cap nhat day du.
 */
void KalmanFilterNode::loadParameters()
{
    try
    {
        get_parameter("topics.input_target_pose", data_.config.topics.inputTargetPoseTopic);
        get_parameter("topics.reset_command", data_.config.topics.resetCommandTopic);
        get_parameter("topics.target_valid", data_.config.topics.targetValidTopic);
        get_parameter("topics.vehicle_odometry", data_.config.topics.vehicleOdometryTopic);
        get_parameter("topics.vehicle_local_position", data_.config.topics.vehicleLocalPositionTopic);
        get_parameter("topics.relative_position_raw", data_.config.topics.relativePositionRawTopic);
        get_parameter("topics.relative_position_predicted", data_.config.topics.relativePositionPredictedTopic);
        get_parameter("topics.relative_velocity", data_.config.topics.relativeVelocityTopic);

        get_parameter("pose_timeout_s", data_.config.poseTimeoutSec);
        get_parameter("frame_id", data_.config.topics.outputFrameId);

        get_parameter("debug", data_.config.debug.enabled);
        get_parameter("debug_csv_path", data_.config.debug.csvPath);

        get_parameter("q_acc_x", data_.config.noise.qAccX);
        get_parameter("q_acc_y", data_.config.noise.qAccY);
        get_parameter("q_acc_z", data_.config.noise.qAccZ);

        get_parameter("r_pos_x", data_.config.noise.rPosX);
        get_parameter("r_pos_y", data_.config.noise.rPosY);
        get_parameter("r_pos_z", data_.config.noise.rPosZ);

        get_parameter("dynamic_r.enable", data_.config.noise.dynamicREnabled);
        get_parameter("dynamic_r.near_range_m", data_.config.noise.nearRange);
        get_parameter("dynamic_r.near_noise_gain", data_.config.noise.nearNoiseGain);
        get_parameter("dynamic_r.max_extra_r_xy", data_.config.noise.maxExtraRxy);
        get_parameter("dynamic_r.min_range_m", data_.config.noise.minDynamicRange);

        get_parameter("transform.mount_mode", data_.config.transform.mountModeString);
        get_parameter("transform.camera_offset_x", data_.config.transform.cameraOffsetBody.x());
        get_parameter("transform.camera_offset_y", data_.config.transform.cameraOffsetBody.y());
        get_parameter("transform.camera_offset_z", data_.config.transform.cameraOffsetBody.z());
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "loadParameters failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "loadParameters failed: unknown exception");
        throw;
    }
}

/**
 * Khoi tao FrameTransformer theo config transform hien tai.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Parse mount mode, tao config tuong ung va dong bo vehicle state.
 *     Neu parse loi thi fallback ve belly_fixed_camera.
 *
 * Output:
 *     frameTransformer_ san sang cho phep bien doi.
 */
void KalmanFilterNode::initFrameTransformer()
{
    try
    {
        try
        {
            data_.config.transform.mountMode =
                frame_transform::FrameTransformer::parseMountMode(
                    data_.config.transform.mountModeString);

            if (data_.config.transform.mountMode == kalman_filter_data::MountMode::BellyFixedCamera)
            {
                data_.config.transform =
                    frame_transform::FrameTransformer::makeBellyFixedCameraConfig(
                        data_.config.transform.cameraOffsetBody);
            }
            else
            {
                data_.config.transform =
                    frame_transform::FrameTransformer::makeBellyGimbalCameraConfig(
                        data_.config.transform.cameraOffsetBody);
            }
        }
        catch (const std::exception &exception)
        {
            KF_WARN(
                get_logger(),
                "Invalid transform.mount_mode='%s', fallback to belly_fixed_camera. reason=%s",
                data_.config.transform.mountModeString.c_str(),
                exception.what());

            data_.config.transform =
                frame_transform::FrameTransformer::makeBellyFixedCameraConfig(
                    data_.config.transform.cameraOffsetBody);
        }

        frameTransformer_.setConfig(data_.config.transform);
        frameTransformer_.setVehicleState(data_.vehicle);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "initFrameTransformer failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "initFrameTransformer failed: unknown exception");
        throw;
    }
}

/**
 * Khoi tao bo loc Kalman voi state 6 bien va measurement 3 bien.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Tao measurement matrix chi do truc tiep vi tri, nap measurement noise
 *     va gan covariance ban dau cho state van toc lon hon de filter mem hon.
 *
 * Output:
 *     kf_ duoc khoi tao day du.
 */
void KalmanFilterNode::initKalman()
{
    try
    {
        kf_ = cv::KalmanFilter(stateSize, measurementSize, 0, CV_64F);

        kf_.transitionMatrix = cv::Mat::eye(stateSize, stateSize, CV_64F);

        kf_.measurementMatrix = cv::Mat::zeros(measurementSize, stateSize, CV_64F);
        kf_.measurementMatrix.at<double>(0, 0) = 1.0;
        kf_.measurementMatrix.at<double>(1, 1) = 1.0;
        kf_.measurementMatrix.at<double>(2, 2) = 1.0;

        kf_.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_64F);

        kf_.measurementNoiseCov = cv::Mat::eye(measurementSize, measurementSize, CV_64F);
        kf_.measurementNoiseCov.at<double>(0, 0) = data_.config.noise.rPosX;
        kf_.measurementNoiseCov.at<double>(1, 1) = data_.config.noise.rPosY;
        kf_.measurementNoiseCov.at<double>(2, 2) = data_.config.noise.rPosZ;
        data_.kalman.dynamicRx = data_.config.noise.rPosX;
        data_.kalman.dynamicRy = data_.config.noise.rPosY;
        data_.kalman.dynamicRz = data_.config.noise.rPosZ;

        kf_.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_64F);
        kf_.errorCovPost.at<double>(3, 3) = 10.0;
        kf_.errorCovPost.at<double>(4, 4) = 10.0;
        kf_.errorCovPost.at<double>(5, 5) = 10.0;

        kf_.statePost = cv::Mat::zeros(stateSize, 1, CV_64F);
        kf_.statePre = cv::Mat::zeros(stateSize, 1, CV_64F);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "initKalman failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "initKalman failed: unknown exception");
        throw;
    }
}

/**
 * Reset toan bo state runtime cua node.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Xoa trang thai Kalman, measurement cu va reset timing de bat dau lai sach.
 *
 * Output:
 *     data_ va kf_ tro ve trang thai moi khoi dong.
 */
void KalmanFilterNode::resetState()
{
    try
    {
        data_.runtime.initialized = false;

        kf_.statePost = cv::Mat::zeros(stateSize, 1, CV_64F);
        kf_.statePre = cv::Mat::zeros(stateSize, 1, CV_64F);

        kf_.errorCovPost = cv::Mat::eye(stateSize, stateSize, CV_64F);
        kf_.errorCovPost.at<double>(3, 3) = 10.0;
        kf_.errorCovPost.at<double>(4, 4) = 10.0;
        kf_.errorCovPost.at<double>(5, 5) = 10.0;

        data_.targetMeasurement = kalman_filter_data::TargetMeasurementData{};
        data_.kalman = kalman_filter_data::KalmanEstimateData{};

        data_.timing.lastPredictTime = rclcpp::Time(0, 0, get_clock()->get_clock_type());
        data_.timing.lastMeasurementTime = rclcpp::Time(0, 0, get_clock()->get_clock_type());
        data_.kalman.predictDt = 0.0;

        data_.runtime.targetValid = false;
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "resetState failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "resetState failed: unknown exception");
        throw;
    }
}

/**
 * Dong bo vehicle state sang FrameTransformer.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Danh dau vehicle valid khi da co ca odom va local position.
 *
 * Output:
 *     frameTransformer_ nhan vehicle state moi nhat.
 */
void KalmanFilterNode::updateFrameTransformerVehicleState()
{
    try
    {
        data_.vehicle.valid =
            data_.runtime.vehicleOdomValid &&
            data_.runtime.vehicleLocalPosValid;

        frameTransformer_.setVehicleState(data_.vehicle);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "updateFrameTransformerVehicleState failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "updateFrameTransformerVehicleState failed: unknown exception");
        throw;
    }
}

/**
 * Callback nhan odometry quaternion cua drone.
 *
 * Input:
 *     msg: px4_msgs::msg::VehicleOdometry::SharedPtr
 *
 * Logic:
 *     Chuyen quaternion q[] sang Eigen, chuan hoa va cap nhat flag odom valid.
 *
 * Output:
 *     data_.vehicle.worldFromBody duoc cap nhat.
 */
void KalmanFilterNode::vehicleOdometryCallback(
    const px4_msgs::msg::VehicleOdometry::SharedPtr msg)
{
    try
    {
        if (msg == nullptr)
        {
            KF_WARN(get_logger(), "vehicleOdometryCallback received null msg");
            return;
        }

        if (!std::isfinite(msg->q[0]) ||
            !std::isfinite(msg->q[1]) ||
            !std::isfinite(msg->q[2]) ||
            !std::isfinite(msg->q[3]))
        {
            KF_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "vehicleOdometryCallback received non-finite quaternion");
            return;
        }

        data_.vehicle.worldFromBody = Eigen::Quaterniond(
            static_cast<double>(msg->q[0]),
            static_cast<double>(msg->q[1]),
            static_cast<double>(msg->q[2]),
            static_cast<double>(msg->q[3]));

        if (data_.vehicle.worldFromBody.norm() > 1e-9)
        {
            data_.vehicle.worldFromBody.normalize();
            data_.runtime.vehicleOdomValid = true;
            updateFrameTransformerVehicleState();
        }
        else
        {
            KF_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "vehicleOdometryCallback received invalid quaternion norm");
        }
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "vehicleOdometryCallback failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "vehicleOdometryCallback failed: unknown exception");
    }
}

/**
 * Callback nhan local position NED cua drone.
 *
 * Input:
 *     msg: px4_msgs::msg::VehicleLocalPosition::SharedPtr
 *
 * Logic:
 *     Kiem tra finite, cap nhat vi tri va van toc world/NED hien tai.
 *
 * Output:
 *     data_.vehicle.positionWorld va velocityWorld duoc cap nhat.
 */
void KalmanFilterNode::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    try
    {
        if (msg == nullptr)
        {
            KF_WARN(get_logger(), "vehicleLocalPositionCallback received null msg");
            return;
        }

        if (!std::isfinite(msg->x) || !std::isfinite(msg->y) || !std::isfinite(msg->z) ||
            !std::isfinite(msg->vx) || !std::isfinite(msg->vy) || !std::isfinite(msg->vz))
        {
            KF_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "vehicleLocalPositionCallback received non-finite data");
            return;
        }

        data_.vehicle.positionWorld.x() = static_cast<double>(msg->x);
        data_.vehicle.positionWorld.y() = static_cast<double>(msg->y);
        data_.vehicle.positionWorld.z() = static_cast<double>(msg->z);

        data_.vehicle.velocityWorld.x() = static_cast<double>(msg->vx);
        data_.vehicle.velocityWorld.y() = static_cast<double>(msg->vy);
        data_.vehicle.velocityWorld.z() = static_cast<double>(msg->vz);

        data_.runtime.vehicleLocalPosValid = true;
        updateFrameTransformerVehicleState();
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "vehicleLocalPositionCallback failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "vehicleLocalPositionCallback failed: unknown exception");
    }
}

/**
 * Callback nhan lenh RESET hoac ACTIVE.
 *
 * Input:
 *     msg: std_msgs::msg::String::SharedPtr
 *
 * Logic:
 *     RESET thi reset filter va bat che do hold.
 *     ACTIVE thi tat forceZero va cho phep publish lai.
 *
 * Output:
 *     data_.runtime duoc cap nhat.
 */
void KalmanFilterNode::resetCallback(const std_msgs::msg::String::SharedPtr msg)
{
    try
    {
        if (msg == nullptr)
        {
            KF_WARN(get_logger(), "resetCallback received null msg");
            return;
        }

        data_.runtime.lastResetCommand = msg->data;

        if (msg->data == "RESET")
        {
            resetState();
            data_.runtime.forceZero = true;
            setForceZeroReason("Kalman paused: external RESET command");

            const rclcpp::Time nowTimestamp = now();
            reportProcessingBlockState(ProcessingBlockState::ForceZeroHold);
            publishZero(nowTimestamp);
            debugLogger_.log(data_, nowTimestamp);
            return;
        }

        if (msg->data == "ACTIVE")
        {
            data_.runtime.forceZero = false;
        }
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "resetCallback failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "resetCallback failed: unknown exception");
    }
}

/**
 * Callback nhan target_valid.
 *
 * Input:
 *     msg: std_msgs::msg::Bool::SharedPtr
 *
 * Logic:
 *     Neu target khong hop le thi bat che do hold.
 *     Neu target hop le tro lai thi tat forceZero.
 *
 * Output:
 *     data_.runtime duoc cap nhat.
 */
void KalmanFilterNode::validCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    try
    {
        if (msg == nullptr)
        {
            KF_WARN(get_logger(), "validCallback received null msg");
            return;
        }

        data_.runtime.targetValid = msg->data;
        data_.runtime.forceZero = !msg->data;

        if (!msg->data)
        {
            setForceZeroReason("Kalman paused: target_valid=false");

            const rclcpp::Time nowTimestamp = now();
            reportProcessingBlockState(ProcessingBlockState::ForceZeroHold);
            publishZero(nowTimestamp);
            debugLogger_.log(data_, nowTimestamp);
        }
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "validCallback failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "validCallback failed: unknown exception");
    }
}

/**
 * Callback nhan measurement pose target trong he optical.
 *
 * Input:
 *     msg: geometry_msgs::msg::PoseStamped::SharedPtr
 *
 * Logic:
 *     Chuyen pose optical sang world, luu measurement moi nhat va correct Kalman.
 *     Neu day la measurement dau tien thi dung no de init state.
 *
 * Output:
 *     data_.targetMeasurement va kf_ duoc cap nhat.
 */
void KalmanFilterNode::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    try
    {
        if (msg == nullptr)
        {
            KF_WARN(get_logger(), "poseCallback received null msg");
            return;
        }

        if (!data_.runtime.vehicleOdomValid)
        {
            return;
        }

        if (!data_.runtime.vehicleLocalPosValid)
        {
            return;
        }

        if (!std::isfinite(msg->pose.position.x) ||
            !std::isfinite(msg->pose.position.y) ||
            !std::isfinite(msg->pose.position.z))
        {
            KF_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "poseCallback received non-finite position");
            return;
        }

        if (!std::isfinite(msg->pose.orientation.w) ||
            !std::isfinite(msg->pose.orientation.x) ||
            !std::isfinite(msg->pose.orientation.y) ||
            !std::isfinite(msg->pose.orientation.z))
        {
            KF_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "poseCallback received non-finite orientation");
            return;
        }

        rclcpp::Time measurementTimestamp = msg->header.stamp;
        if (measurementTimestamp.nanoseconds() == 0)
        {
            measurementTimestamp = now();
        }

        data_.targetMeasurement.stamp = measurementTimestamp;
        data_.timing.lastMeasurementTime = measurementTimestamp;

        data_.targetMeasurement.positionOptical =
            Eigen::Vector3d(
                msg->pose.position.x,
                msg->pose.position.y,
                msg->pose.position.z);

        data_.targetMeasurement.orientationOptical =
            Eigen::Quaterniond(
                msg->pose.orientation.w,
                msg->pose.orientation.x,
                msg->pose.orientation.y,
                msg->pose.orientation.z);

        if (data_.targetMeasurement.orientationOptical.norm() > 1e-9)
        {
            data_.targetMeasurement.orientationOptical.normalize();
        }
        else
        {
            data_.targetMeasurement.orientationOptical.setIdentity();
        }

        data_.targetMeasurement.positionWorld =
            frameTransformer_.opticalPositionToWorld(
                data_.targetMeasurement.positionOptical);

        data_.targetMeasurement.orientationWorld =
            frameTransformer_.opticalOrientationToWorld(
                data_.targetMeasurement.orientationOptical);

        data_.targetMeasurement.valid = true;
        data_.kalman.rawMeasurementWorld = data_.targetMeasurement.positionWorld;

        data_.runtime.forceZero = false;
        setForceZeroReason("Kalman processing active");

        cv::Mat measurement(measurementSize, 1, CV_64F);
        measurement.at<double>(0, 0) = data_.targetMeasurement.positionWorld.x();
        measurement.at<double>(1, 0) = data_.targetMeasurement.positionWorld.y();
        measurement.at<double>(2, 0) = data_.targetMeasurement.positionWorld.z();

        if (!data_.runtime.initialized)
        {
            kf_.statePost.at<double>(0, 0) = data_.targetMeasurement.positionWorld.x();
            kf_.statePost.at<double>(1, 0) = data_.targetMeasurement.positionWorld.y();
            kf_.statePost.at<double>(2, 0) = data_.targetMeasurement.positionWorld.z();
            kf_.statePost.at<double>(3, 0) = 0.0;
            kf_.statePost.at<double>(4, 0) = 0.0;
            kf_.statePost.at<double>(5, 0) = 0.0;

            kf_.statePre = kf_.statePost.clone();
            data_.runtime.initialized = true;
            data_.timing.lastPredictTime = measurementTimestamp;
            data_.kalman.predictDt = 0.0;

            KF_INFO(
                get_logger(),
                "Kalman initialized from first pose | world=(%.3f, %.3f, %.3f)",
                data_.targetMeasurement.positionWorld.x(),
                data_.targetMeasurement.positionWorld.y(),
                data_.targetMeasurement.positionWorld.z());

            reportProcessingBlockState(ProcessingBlockState::None);
            publishEstimatedState(measurementTimestamp);
            debugLogger_.log(data_, now());
            return;
        }

        processAndPublishMeasurement(measurementTimestamp, measurement);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "poseCallback failed: %s", exception.what());
        logStateSummary("poseCallback exception");
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "poseCallback failed: unknown exception");
        logStateSummary("poseCallback unknown exception");
    }
}

/**
 * Xu ly 1 measurement moi va publish ngay lap tuc theo timestamp cua measurement.
 *
 * Input:
 *     measurementTimestamp: timestamp goc cua measurement dang xu ly
 *     measurement: vector measurement vi tri world [x y z]^T
 *
 * Logic:
 *     - Danh gia block state hien tai
 *     - Tinh dt chinh xac tu 2 timestamp measurement lien tiep
 *     - Predict neu dt > 0
 *     - Correct bang measurement moi
 *     - Publish output ngay sau khi xu ly xong, khong qua timer 33 ms
 *
 * Output:
 *     State output duoc publish ngay sau khi xu ly xong, header.stamp la thoi diem publish thuc te.
 *     measurementTimestamp chi duoc dung de tinh predict dt.
 */
void KalmanFilterNode::processAndPublishMeasurement(
    const rclcpp::Time &measurementTimestamp,
    const cv::Mat &measurement)
{
    try
    {
        const ProcessingBlockState blockState = evaluateProcessingBlockState();
        reportProcessingBlockState(blockState);

        if (blockState != ProcessingBlockState::None)
        {
            if (blockState == ProcessingBlockState::ForceZeroHold)
            {
                publishZero(now());
                debugLogger_.log(data_, now());
            }
            return;
        }

        double dt = 0.0;

        if (data_.timing.lastPredictTime.nanoseconds() > 0)
        {
            dt = (measurementTimestamp - data_.timing.lastPredictTime).seconds();
        }

        if (dt < 0.0)
        {
            KF_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                2000,
                "Received out-of-order measurement stamp. skip update | dt=%.6f",
                dt);
            return;
        }

        data_.kalman.predictDt = dt;

        if (dt > 1e-9)
        {
            predict(dt);
        }

        updateDynamicMeasurementNoise();
        kf_.correct(measurement);
        data_.timing.lastPredictTime = measurementTimestamp;

        publishEstimatedState(measurementTimestamp);
        debugLogger_.log(data_, now());
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(
            get_logger(),
            "processAndPublishMeasurement failed: %s | initialized=%d forceZero=%d odomValid=%d localPosValid=%d",
            exception.what(),
            static_cast<int>(data_.runtime.initialized),
            static_cast<int>(data_.runtime.forceZero),
            static_cast<int>(data_.runtime.vehicleOdomValid),
            static_cast<int>(data_.runtime.vehicleLocalPosValid));
        logStateSummary("processAndPublishMeasurement exception");
    }
    catch (...)
    {
        KF_ERROR(
            get_logger(),
            "processAndPublishMeasurement failed: unknown exception | initialized=%d forceZero=%d odomValid=%d localPosValid=%d",
            static_cast<int>(data_.runtime.initialized),
            static_cast<int>(data_.runtime.forceZero),
            static_cast<int>(data_.runtime.vehicleOdomValid),
            static_cast<int>(data_.runtime.vehicleLocalPosValid));
        logStateSummary("processAndPublishMeasurement unknown exception");
    }
}

/**
 * Cap nhat ma tran nhieu do R dong truoc buoc correct.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     - R co dinh phu hop khi camera va ArUco on dinh.
 *     - Khi UAV ha xuong qua gan marker, PnP/ArUco co the dao dong hon.
 *     - Phan R_x, R_y tang them theo khoang cach gan marker.
 *     - Khong su dung gyro/attitude rate trong dynamic R nay.
 *
 * Output:
 *     kf_.measurementNoiseCov duoc cap nhat theo khoang cach camera-target hien tai.
 */
void KalmanFilterNode::updateDynamicMeasurementNoise()
{
    try
    {
        const DynamicMeasurementNoiseResult dynamicNoiseResult =
            dynamicMeasurementNoiseEstimator_.estimate(
                data_.config.noise,
                data_.targetMeasurement.positionOptical);

        dynamicMeasurementNoiseEstimator_.applyToMeasurementNoiseCov(
            kf_.measurementNoiseCov,
            dynamicNoiseResult);

        data_.kalman.dynamicRx = dynamicNoiseResult.rx;
        data_.kalman.dynamicRy = dynamicNoiseResult.ry;
        data_.kalman.dynamicRz = dynamicNoiseResult.rz;
        data_.kalman.dynamicRExtraXY = dynamicNoiseResult.extraRxy;
        data_.kalman.dynamicRangeToTarget = dynamicNoiseResult.rangeToTarget;
        data_.kalman.dynamicNearRangeError = dynamicNoiseResult.nearRangeError;
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "updateDynamicMeasurementNoise failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "updateDynamicMeasurementNoise failed: unknown exception");
        throw;
    }
}
/**
 * Buoc predict cua Kalman theo mo hinh constant velocity.
 *
 * Input:
 *     dt: buoc thoi gian giua 2 lan predict
 *
 * Logic:
 *     Dung mo hinh white acceleration cho tung truc x, y, z de tao process noise.
 *
 * Output:
 *     kf_ duoc predict sang trang thai moi.
 */
void KalmanFilterNode::predict(double dt)
{
    try
    {
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;
        const double dt4 = dt3 * dt;

        kf_.transitionMatrix = cv::Mat::eye(stateSize, stateSize, CV_64F);
        kf_.transitionMatrix.at<double>(0, 3) = dt;
        kf_.transitionMatrix.at<double>(1, 4) = dt;
        kf_.transitionMatrix.at<double>(2, 5) = dt;

        kf_.processNoiseCov = cv::Mat::zeros(stateSize, stateSize, CV_64F);

        kf_.processNoiseCov.at<double>(0, 0) = 0.25 * dt4 * data_.config.noise.qAccX;
        kf_.processNoiseCov.at<double>(0, 3) = 0.5 * dt3 * data_.config.noise.qAccX;
        kf_.processNoiseCov.at<double>(3, 0) = 0.5 * dt3 * data_.config.noise.qAccX;
        kf_.processNoiseCov.at<double>(3, 3) = dt2 * data_.config.noise.qAccX;

        kf_.processNoiseCov.at<double>(1, 1) = 0.25 * dt4 * data_.config.noise.qAccY;
        kf_.processNoiseCov.at<double>(1, 4) = 0.5 * dt3 * data_.config.noise.qAccY;
        kf_.processNoiseCov.at<double>(4, 1) = 0.5 * dt3 * data_.config.noise.qAccY;
        kf_.processNoiseCov.at<double>(4, 4) = dt2 * data_.config.noise.qAccY;

        kf_.processNoiseCov.at<double>(2, 2) = 0.25 * dt4 * data_.config.noise.qAccZ;
        kf_.processNoiseCov.at<double>(2, 5) = 0.5 * dt3 * data_.config.noise.qAccZ;
        kf_.processNoiseCov.at<double>(5, 2) = 0.5 * dt3 * data_.config.noise.qAccZ;
        kf_.processNoiseCov.at<double>(5, 5) = dt2 * data_.config.noise.qAccZ;

        kf_.predict();
        data_.kalman.predictCount++;
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "predict failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "predict failed: unknown exception");
        throw;
    }
}

/**
 * Publish raw measurement, filtered position va estimated velocity.
 *
 * Input:
 *     measurementTimestamp: timestamp goc cua measurement vua duoc xu ly
 *
 * Logic:
 *     - Lay state hien tai tu Kalman roi publish ra 3 topic
 *     - header.stamp cua output dung thoi diem publish thuc te, khong dung measurementTimestamp
 *     - measurementTimestamp chi dung de tinh latency tu measurement den publish
 *     - Orientation dung orientation world gan nhat cua target
 *
 * Output:
 *     Tat ca topic output duoc publish voi timestamp publish thuc te.
 */
void KalmanFilterNode::publishEstimatedState(const rclcpp::Time &measurementTimestamp)
{
    try
    {
        if (!targetPoseRawPub_ || !targetPoseFilteredPub_ || !targetRelVelPub_)
        {
            throw std::runtime_error("One or more publishers are null");
        }

        data_.kalman.estimatedPositionWorld =
            Eigen::Vector3d(
                kf_.statePost.at<double>(0, 0),
                kf_.statePost.at<double>(1, 0),
                kf_.statePost.at<double>(2, 0));

        data_.kalman.estimatedVelocityWorld =
            Eigen::Vector3d(
                kf_.statePost.at<double>(3, 0),
                kf_.statePost.at<double>(4, 0),
                kf_.statePost.at<double>(5, 0));

        const rclcpp::Time publishTimestamp = now();
        const double measurementToPublishLatencySec =(measurementTimestamp.nanoseconds() > 0)? (publishTimestamp - measurementTimestamp).seconds() : -1.0;
        
        geometry_msgs::msg::PoseStamped rawPoseMsg;
        rawPoseMsg.header.stamp = measurementTimestamp;
        rawPoseMsg.header.frame_id = data_.config.topics.outputFrameId;
        rawPoseMsg.pose.position.x = data_.kalman.rawMeasurementWorld.x();
        rawPoseMsg.pose.position.y = data_.kalman.rawMeasurementWorld.y();
        rawPoseMsg.pose.position.z = data_.kalman.rawMeasurementWorld.z();
        rawPoseMsg.pose.orientation.w = data_.targetMeasurement.orientationWorld.w();
        rawPoseMsg.pose.orientation.x = data_.targetMeasurement.orientationWorld.x();
        rawPoseMsg.pose.orientation.y = data_.targetMeasurement.orientationWorld.y();
        rawPoseMsg.pose.orientation.z = data_.targetMeasurement.orientationWorld.z();
        targetPoseRawPub_->publish(rawPoseMsg);

        geometry_msgs::msg::PoseStamped filteredPoseMsg;
        filteredPoseMsg.header.stamp = measurementTimestamp;
        filteredPoseMsg.header.frame_id = data_.config.topics.outputFrameId;
        filteredPoseMsg.pose.position.x = data_.kalman.estimatedPositionWorld.x();
        filteredPoseMsg.pose.position.y = data_.kalman.estimatedPositionWorld.y();
        filteredPoseMsg.pose.position.z = data_.kalman.estimatedPositionWorld.z();
        filteredPoseMsg.pose.orientation.w = data_.targetMeasurement.orientationWorld.w();
        filteredPoseMsg.pose.orientation.x = data_.targetMeasurement.orientationWorld.x();
        filteredPoseMsg.pose.orientation.y = data_.targetMeasurement.orientationWorld.y();
        filteredPoseMsg.pose.orientation.z = data_.targetMeasurement.orientationWorld.z();
        targetPoseFilteredPub_->publish(filteredPoseMsg);

        geometry_msgs::msg::PoseStamped velocityMsg;
        velocityMsg.header.stamp = measurementTimestamp;
        velocityMsg.header.frame_id = data_.config.topics.outputFrameId;
        velocityMsg.pose.position.x = data_.kalman.estimatedVelocityWorld.x();
        velocityMsg.pose.position.y = data_.kalman.estimatedVelocityWorld.y();
        velocityMsg.pose.position.z = data_.kalman.estimatedVelocityWorld.z();
        velocityMsg.pose.orientation.w = data_.targetMeasurement.orientationWorld.w();
        velocityMsg.pose.orientation.x = data_.targetMeasurement.orientationWorld.x();
        velocityMsg.pose.orientation.y = data_.targetMeasurement.orientationWorld.y();
        velocityMsg.pose.orientation.z = data_.targetMeasurement.orientationWorld.z();
        targetRelVelPub_->publish(velocityMsg);

        KF_DEBUG(
            get_logger(),
            "Published Kalman output | pub_stamp=%.6f meas_to_pub=%.6f pos=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)",
            publishTimestamp.seconds(),
            measurementToPublishLatencySec,
            data_.kalman.estimatedPositionWorld.x(),
            data_.kalman.estimatedPositionWorld.y(),
            data_.kalman.estimatedPositionWorld.z(),
            data_.kalman.estimatedVelocityWorld.x(),
            data_.kalman.estimatedVelocityWorld.y(),
            data_.kalman.estimatedVelocityWorld.z());
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "publishEstimatedState failed: %s", exception.what());
        logStateSummary("publishEstimatedState exception");
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "publishEstimatedState failed: unknown exception");
        logStateSummary("publishEstimatedState unknown exception");
        throw;
    }
}

/**
 * Ghi log tong hop state hien tai cua node.
 *
 * Input:
 *     prefix: chuoi mo ta bo canh can in truoc state
 *
 * Logic:
 *     Tong hop cac flag runtime va timing quan trong de debug.
 *
 * Output:
 *     In ra 1 dong state summary.
 */
void KalmanFilterNode::logStateSummary(const std::string &prefix)
{
    try
    {
        const double measurementAgeSec =
            (data_.timing.lastMeasurementTime.nanoseconds() > 0)
                ? (now() - data_.timing.lastMeasurementTime).seconds()
                : -1.0;

        KF_WARN(
            get_logger(),
            "%s | init=%d forceZero=%d targetValid=%d measValid=%d odomValid=%d localPosValid=%d "
            "predictDt=%.4f predictCount=%lu measurementAge=%.3f",
            prefix.c_str(),
            static_cast<int>(data_.runtime.initialized),
            static_cast<int>(data_.runtime.forceZero),
            static_cast<int>(data_.runtime.targetValid),
            static_cast<int>(data_.targetMeasurement.valid),
            static_cast<int>(data_.runtime.vehicleOdomValid),
            static_cast<int>(data_.runtime.vehicleLocalPosValid),
            data_.kalman.predictDt,
            static_cast<unsigned long>(data_.kalman.predictCount),
            measurementAgeSec);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "logStateSummary failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "logStateSummary failed: unknown exception");
    }
}

/**
 * Cap nhat ly do dang o che do forceZero / hold.
 *
 * Input:
 *     reason: ly do kich hoat forceZero
 *
 * Logic:
 *     Luu lai nguyen nhan de report block state co the in dung ngu canh.
 *
 * Output:
 *     forceZeroReason_ duoc cap nhat.
 */
void KalmanFilterNode::setForceZeroReason(const std::string &reason)
{
    try
    {
        forceZeroReason_ = reason;
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "setForceZeroReason failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "setForceZeroReason failed: unknown exception");
    }
}

/**
 * Danh gia trang thai block hien tai cua node.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Chon 1 nguyen nhan block duy nhat, uu tien:
 *     forceZero -> odom -> local position -> first measurement.
 *
 * Output:
 *     Trang thai block hien tai.
 */
KalmanFilterNode::ProcessingBlockState KalmanFilterNode::evaluateProcessingBlockState() const
{
    try
    {
        if (data_.runtime.forceZero)
        {
            return ProcessingBlockState::ForceZeroHold;
        }

        if (!data_.runtime.vehicleOdomValid)
        {
            return ProcessingBlockState::WaitVehicleOdom;
        }

        if (!data_.runtime.vehicleLocalPosValid)
        {
            return ProcessingBlockState::WaitVehicleLocalPos;
        }

        if (!data_.runtime.initialized)
        {
            return ProcessingBlockState::WaitFirstMeasurement;
        }

        return ProcessingBlockState::None;
    }
    catch (...)
    {
        return ProcessingBlockState::WaitFirstMeasurement;
    }
}

/**
 * Tao message log tu trang thai block.
 *
 * Input:
 *     state: trang thai block hien tai
 *
 * Logic:
 *     Chuyen enum sang message ro rang de in ra terminal.
 *
 * Output:
 *     Chuoi mo ta block hien tai.
 */
std::string KalmanFilterNode::buildProcessingBlockMessage(ProcessingBlockState state) const
{
    try
    {
        switch (state)
        {
            case ProcessingBlockState::None:
                return "Kalman processing resumed";

            case ProcessingBlockState::WaitVehicleOdom:
                return "Kalman paused: waiting for vehicle odometry";

            case ProcessingBlockState::WaitVehicleLocalPos:
                return "Kalman paused: waiting for vehicle local position";

            case ProcessingBlockState::WaitFirstMeasurement:
                return "Kalman paused: waiting for first valid target measurement";

            case ProcessingBlockState::ForceZeroHold:
                return forceZeroReason_;
        }

        return "Kalman paused: unknown state";
    }
    catch (...)
    {
        return "Kalman paused: failed to build block message";
    }
}

/**
 * Log block state chi 1 lan khi co thay doi.
 *
 * Input:
 *     state: trang thai block hien tai
 *
 * Logic:
 *     Neu state khong doi thi khong log lai.
 *     Neu state doi thi log 1 lan.
 *
 * Output:
 *     Terminal khong bi spam log.
 */
void KalmanFilterNode::reportProcessingBlockState(ProcessingBlockState state)
{
    try
    {
        if (state == lastProcessingBlockState_)
        {
            return;
        }

        if (state == ProcessingBlockState::None)
        {
            KF_INFO(get_logger(), "%s", buildProcessingBlockMessage(state).c_str());
        }
        else
        {
            KF_WARN(get_logger(), "%s", buildProcessingBlockMessage(state).c_str());
            logStateSummary("processing paused");
        }

        lastProcessingBlockState_ = state;
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "reportProcessingBlockState failed: %s", exception.what());
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "reportProcessingBlockState failed: unknown exception");
    }
}

/**
 * Publish state hold de drone hover tai cho.
 *
 * Input:
 *     nowTimestamp: tham so du phong, output se dong dau theo thoi diem publish thuc te
 *
 * Logic:
 *     - Vi output dang la world pose, neu publish (0,0,0) drone se bi keo ve goc world
 *     - De hover tai cho, pose raw va filtered phai publish vi tri world hien tai cua UAV
 *     - Van toc output duoc dua ve 0 de dung yeu cau di chuyen
 *     - Neu chua co vehicle local position thi moi fallback ve 0
 *     - header.stamp cua output dung thoi diem publish thuc te
 *
 * Output:
 *     Downstream nhan duoc lenh giu vi tri hien tai va van toc bang 0.
 */
void KalmanFilterNode::publishZero(const rclcpp::Time &nowTimestamp)
{
    try
    {
        Eigen::Vector3d holdWorldPosition = Eigen::Vector3d::Zero();

        if (data_.runtime.vehicleLocalPosValid)
        {
            holdWorldPosition = data_.vehicle.positionWorld;
        }

        Eigen::Quaterniond holdOrientation = Eigen::Quaterniond::Identity();

        if (data_.targetMeasurement.valid)
        {
            holdOrientation = data_.targetMeasurement.orientationWorld;
        }

        const rclcpp::Time publishTimestamp = now();
        (void)nowTimestamp;

        geometry_msgs::msg::PoseStamped holdPoseMsg;
        holdPoseMsg.header.stamp = publishTimestamp;
        holdPoseMsg.header.frame_id = data_.config.topics.outputFrameId;
        holdPoseMsg.pose.position.x = holdWorldPosition.x();
        holdPoseMsg.pose.position.y = holdWorldPosition.y();
        holdPoseMsg.pose.position.z = holdWorldPosition.z();
        holdPoseMsg.pose.orientation.w = holdOrientation.w();
        holdPoseMsg.pose.orientation.x = holdOrientation.x();
        holdPoseMsg.pose.orientation.y = holdOrientation.y();
        holdPoseMsg.pose.orientation.z = holdOrientation.z();

        geometry_msgs::msg::PoseStamped zeroVelocityMsg;
        zeroVelocityMsg.header.stamp = publishTimestamp;
        zeroVelocityMsg.header.frame_id = data_.config.topics.outputFrameId;
        zeroVelocityMsg.pose.position.x = 0.0;
        zeroVelocityMsg.pose.position.y = 0.0;
        zeroVelocityMsg.pose.position.z = 0.0;
        zeroVelocityMsg.pose.orientation.w = holdOrientation.w();
        zeroVelocityMsg.pose.orientation.x = holdOrientation.x();
        zeroVelocityMsg.pose.orientation.y = holdOrientation.y();
        zeroVelocityMsg.pose.orientation.z = holdOrientation.z();

        targetPoseRawPub_->publish(holdPoseMsg);
        targetPoseFilteredPub_->publish(holdPoseMsg);
        targetRelVelPub_->publish(zeroVelocityMsg);
    }
    catch (const std::exception &exception)
    {
        KF_ERROR(get_logger(), "publishZero failed: %s", exception.what());
        throw;
    }
    catch (...)
    {
        KF_ERROR(get_logger(), "publishZero failed: unknown exception");
        throw;
    }
}

int main(int argc, char **argv)
{
    try
    {
        rclcpp::init(argc, argv);
        rclcpp::spin(std::make_shared<KalmanFilterNode>());
        rclcpp::shutdown();
        return 0;
    }
    catch (const std::exception &exception)
    {
        KF_FATAL(
            rclcpp::get_logger("kalman_filter_node"),
            "main failed: %s",
            exception.what());
        return -1;
    }
    catch (...)
    {
        KF_FATAL(
            rclcpp::get_logger("kalman_filter_node"),
            "main failed: unknown exception");
        return -1;
    }
}
