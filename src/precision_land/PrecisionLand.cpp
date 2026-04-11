#include "PrecisionLand.hpp"

#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iomanip>
#include <sstream>
#include <std_msgs/msg/string.hpp>

namespace
{
const std::string kModeName = "PLHEOC";
constexpr bool kEnableDebugOutput = true;

/**
 * Publish debug timing cho node PrecisionLand.
 *
 * Input:
 *     pub: publisher debug timing
 *     imageStampSec: stamp anh goc de gom end-to-end
 *     poseStampSec: stamp pose filtered sau Kalman
 *     velStampSec: stamp velocity filtered sau Kalman
 *     poseRxNowSec: thoi diem callback nhan pose
 *     velRxNowSec: thoi diem callback nhan velocity
 *     ctrlStartNowSec: thoi diem bat dau tinh dieu khien
 *     ctrlEndNowSec: thoi diem tinh dieu khien xong
 *     cmdPubNowSec: thoi diem gui setpoint di
 *
 * Logic:
 *     - imageStampSec dung de cong end-to-end toan pipeline
 *     - poseStampSec/velStampSec dung de tinh tuoi du lieu con lai sau Kalman
 *
 * Output:
 *     Publish JSON string len /debug_dt/precision_land
 */
void publishPrecisionLandTiming(
    const rclcpp::Publisher<std_msgs::msg::String>::SharedPtr &pub,
    double imageStampSec,
    double poseStampSec,
    double velStampSec,
    double poseRxNowSec,
    double velRxNowSec,
    double ctrlStartNowSec,
    double ctrlEndNowSec,
    double cmdPubNowSec)
{
    if (!pub)
    {
        return;
    }

    const double poseWaitDt =
        (ctrlStartNowSec >= 0.0 && poseRxNowSec >= 0.0) ? (ctrlStartNowSec - poseRxNowSec) : -1.0;

    const double velWaitDt =
        (ctrlStartNowSec >= 0.0 && velRxNowSec >= 0.0) ? (ctrlStartNowSec - velRxNowSec) : -1.0;

    const double controlProcessingDt =
        (ctrlEndNowSec >= 0.0 && ctrlStartNowSec >= 0.0) ? (ctrlEndNowSec - ctrlStartNowSec) : -1.0;

    const double sendCmdDt =
        (cmdPubNowSec >= 0.0 && ctrlEndNowSec >= 0.0) ? (cmdPubNowSec - ctrlEndNowSec) : -1.0;

    const double totalImageToCmd =
        (cmdPubNowSec >= 0.0 && imageStampSec >= 0.0) ? (cmdPubNowSec - imageStampSec) : -1.0;

    std_msgs::msg::String msg;
    std::ostringstream ss;

    ss << std::fixed << std::setprecision(6)
       << "{"
       << "\"node\":\"precision_land\","
       << "\"image_stamp\":" << imageStampSec << ","
       << "\"pose_stamp\":" << poseStampSec << ","
       << "\"vel_stamp\":" << velStampSec << ","
       << "\"pose_rx_now\":" << poseRxNowSec << ","
       << "\"vel_rx_now\":" << velRxNowSec << ","
       << "\"ctrl_start_now\":" << ctrlStartNowSec << ","
       << "\"ctrl_end_now\":" << ctrlEndNowSec << ","
       << "\"cmd_pub_now\":" << cmdPubNowSec << ","
       << "\"pose_wait_dt\":" << poseWaitDt << ","
       << "\"vel_wait_dt\":" << velWaitDt << ","
       << "\"control_processing_dt\":" << controlProcessingDt << ","
       << "\"send_cmd_dt\":" << sendCmdDt << ","
       << "\"total_image_to_cmd_dt\":" << totalImageToCmd
       << "}";

    msg.data = ss.str();
    pub->publish(msg);
}
}

using namespace px4_ros2::literals;

PrecisionLand::PrecisionLand(rclcpp::Node &node)
    : ModeBase(node, kModeName),
      _node(node)
{
    _trajectory_setpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
    _vehicle_local_position = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
    _vehicle_attitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);

    loadParameters();

    _target_pose_sub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetPoseTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::targetPoseCallback, this, std::placeholders::_1));

    _target_velocity_sub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetVelocityTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::targetVelocityCallback, this, std::placeholders::_1));

    _vehicle_land_detected_sub =
        _node.create_subscription<px4_msgs::msg::VehicleLandDetected>(
            _vehicleLandDetectedTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::vehicleLandDetectedCallback, this, std::placeholders::_1));

    _vehicle_local_pos_sub =
        _node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            _vehicleLocalPositionTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::vehicleLocalPositionCallback, this, std::placeholders::_1));

    _gimbal_sub =
        _node.create_subscription<geometry_msgs::msg::Vector3>(
            _gimbalAttitudeTopic,
            rclcpp::QoS(10).best_effort(),
            std::bind(&PrecisionLand::gimbalAttCallback, this, std::placeholders::_1));

    _gimbal_seq_pub =
        _node.create_publisher<std_msgs::msg::String>(
            _gimbalCommandTopic,
            rclcpp::QoS(1).best_effort());

    _debug_target_pred_pub =
        _node.create_publisher<geometry_msgs::msg::PoseStamped>(
            "/debug/precision_land/target_pose_pred_world",
            rclcpp::QoS(1).best_effort());

    _debug_dt_pub =
        _node.create_publisher<std_msgs::msg::String>(
            "/debug_dt/precision_land",
            rclcpp::QoS(10).best_effort());

    modeRequirements().manual_control = false;
}

void PrecisionLand::loadParameters()
{
    _node.declare_parameter<std::string>("topics.target_pose", "/KalmanFilter/target_pose_est_NED");
    _node.declare_parameter<std::string>("topics.target_velocity", "/KalmanFilter/target_velocity_est_NED");
    _node.declare_parameter<std::string>("topics.vehicle_land_detected", "/fmu/out/vehicle_land_detected");
    _node.declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position");
    _node.declare_parameter<std::string>("topics.gimbal_command", "/gimbal/cmd/sequence");
    _node.declare_parameter<std::string>("topics.gimbal_attitude", "/gimbal/state/attitude");

    _node.declare_parameter<float>("PID_deadband", 0.05f);
    _node.declare_parameter<float>("target_timeout", 3.0f);

    _node.declare_parameter<float>("descent_kp_pid", 0.6f);
    _node.declare_parameter<float>("descent_ki_pid", 0.0f);
    _node.declare_parameter<float>("descent_kd_pid", 0.0f);
    _node.declare_parameter<float>("descent_max_velocity", 10.0f);
    _node.declare_parameter<float>("slew_acc", 18.0f);

    _node.declare_parameter<float>("land_zone_z", 0.5f);
    _node.declare_parameter<float>("descent_vel", 0.4f);

    _node.declare_parameter<float>("descent_gate_radius", 0.3f);
    _node.declare_parameter<float>("vmin", 0.45f);
    _node.declare_parameter<float>("vmax", 0.8f);

    _node.declare_parameter<bool>("use_predictive_error", true);
    _node.declare_parameter<float>("prediction_dt_max", 0.75f);
    _node.declare_parameter<float>("control_extra_lead_sec", 0.18f);

    _node.declare_parameter<float>("predictive_acc_gain", 0.0f);
    _node.declare_parameter<float>("predictive_acc_lpf_alpha", 0.4f);
    _node.declare_parameter<float>("predictive_acc_max", 4.0f);

    _node.get_parameter("topics.target_pose", _targetPoseTopic);
    _node.get_parameter("topics.target_velocity", _targetVelocityTopic);
    _node.get_parameter("topics.vehicle_land_detected", _vehicleLandDetectedTopic);
    _node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);
    _node.get_parameter("topics.gimbal_command", _gimbalCommandTopic);
    _node.get_parameter("topics.gimbal_attitude", _gimbalAttitudeTopic);

    _node.get_parameter("PID_deadband", _param_pid_deadband);
    _node.get_parameter("target_timeout", _param_target_timeout);

    _node.get_parameter("descent_kp_pid", _param_descent_kp);
    _node.get_parameter("descent_ki_pid", _param_descent_ki);
    _node.get_parameter("descent_kd_pid", _param_descent_kd);
    _node.get_parameter("descent_max_velocity", _param_descent_max_velocity);
    _node.get_parameter("slew_acc", _param_slew_acc);

    _node.get_parameter("land_zone_z", _param_land_zone_z);
    _node.get_parameter("descent_vel", _param_descent_vel);

    _node.get_parameter("descent_gate_radius", _param_descent_gate_radius);
    _node.get_parameter("vmin", _param_vmin);
    _node.get_parameter("vmax", _param_vmax);

    _node.get_parameter("use_predictive_error", _param_use_predictive_error);
    _node.get_parameter("prediction_dt_max", _param_prediction_dt_max);
    _node.get_parameter("control_extra_lead_sec", _param_control_extra_lead_sec);

    _node.get_parameter("predictive_acc_gain", _param_predictive_acc_gain);
    _node.get_parameter("predictive_acc_lpf_alpha", _param_predictive_acc_lpf_alpha);
    _node.get_parameter("predictive_acc_max", _param_predictive_acc_max);
}

void PrecisionLand::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    z_dist_bottom = msg->dist_bottom;
}

void PrecisionLand::vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg)
{
    _land_detected = msg->landed;
}

void PrecisionLand::gimbalAttCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    _gimbal_pitch_deg = static_cast<float>(msg->y);
    _gimbal_ready = std::abs(_gimbal_pitch_deg) > 80.0f;

    const double yaw = msg->x * M_PI / 180.0;
    const double pitch = msg->y * M_PI / 180.0;
    const double roll = msg->z * M_PI / 180.0;

    _q_gimbal =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

    _q_gimbal.normalize();
    _gimbal_valid = true;
}

void PrecisionLand::targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_search_started)
    {
        return;
    }

    _targetWorld.position = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    rclcpp::Time msgTimestamp = msg->header.stamp;
    if (msgTimestamp.nanoseconds() == 0)
    {
        msgTimestamp = _node.now();
    }

    _targetWorld.timestamp = msgTimestamp;
    imageTimestamp = msgTimestamp;

    _targetWorld.validPose = true;
    _targetPoseRxNow = _node.now();
}

void PrecisionLand::targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    _targetWorld.velocity = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    rclcpp::Time msgTimestamp = msg->header.stamp;
    if (msgTimestamp.nanoseconds() == 0)
    {
        msgTimestamp = _node.now();
    }

    _targetWorld.velocityTimestamp = msgTimestamp;
    _targetWorld.validVelocity = true;
    _targetVelRxNow = _node.now();
}

void PrecisionLand::onActivate()
{
    _vxFilt = 0.0f;
    _vyFilt = 0.0f;
    _velXIntegral = 0.0f;
    _velYIntegral = 0.0f;

    _prevErrPredX = 0.0f;
    _prevErrPredY = 0.0f;
    _prevErrPredValid = false;

    _prevVehicleVelX = 0.0f;
    _prevVehicleVelY = 0.0f;
    _vehicleAccXFilt = 0.0f;
    _vehicleAccYFilt = 0.0f;
    _prevVehicleVelValid = false;

    _search_started = true;
    _yawSpInit = false;

    switchToState(State::Search);
}

void PrecisionLand::Hover()
{
    RCLCPP_INFO_THROTTLE(_node.get_logger(), *(_node.get_clock()), 2000, "Hovering...");
    _trajectory_setpoint->update(
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        std::nullopt,
        std::nullopt);
}

void PrecisionLand::onDeactivate()
{
    _search_started = false;
}

void PrecisionLand::updateSetpoint(float dt_s)
{
    const bool targetLost = checkTargetTimeout();

    if (targetLost && !_target_lost_prev)
    {
        RCLCPP_INFO(_node.get_logger(), "Target lost (state=%s)", stateName(_state).c_str());
    }
    else if (!targetLost && _target_lost_prev)
    {
        RCLCPP_INFO(_node.get_logger(), "Target acquired");
    }
    _target_lost_prev = targetLost;

    switch (_state)
    {
    case State::Search:
    {
        if (!targetLost && _targetWorld.validPose)
        {
            switchToState(State::Descend);
            break;
        }

        Hover();
        break;
    }

    case State::Descend:
    {
        if (targetLost)
        {
            switchToState(State::Search);
            break;
        }

        const rclcpp::Time ctrlStartNow = _node.now();

        const Eigen::Vector3f dronePosition = _vehicle_local_position->positionNed();
        const Eigen::Vector3f droneVelocity = _vehicle_local_position->velocityNed();
        const Eigen::Vector2f vehicleAccelerationXY = estimateVehicleAccelerationXY(dt_s);

        const Eigen::Vector3f targetPositionWorld(
            static_cast<float>(_targetWorld.position.x()),
            static_cast<float>(_targetWorld.position.y()),
            static_cast<float>(_targetWorld.position.z()));

        const bool hasTargetVelocity =
            _param_use_predictive_error && _targetWorld.validVelocity;

        Eigen::Vector3f targetVelocityWorld(0.0f, 0.0f, 0.0f);
        if (hasTargetVelocity)
        {
            targetVelocityWorld.x() = static_cast<float>(_targetWorld.velocity.x());
            targetVelocityWorld.y() = static_cast<float>(_targetWorld.velocity.y());
            targetVelocityWorld.z() = static_cast<float>(_targetWorld.velocity.z());
        }

        const Eigen::Vector3f relativePositionWorld = targetPositionWorld - dronePosition;
        const float errX = relativePositionWorld.x();
        const float errY = relativePositionWorld.y();

        float poseAgeSec = static_cast<float>((ctrlStartNow - _targetWorld.timestamp).seconds());
        if (poseAgeSec < 0.0f)
        {
            poseAgeSec = 0.0f;
        }

        float velAgeSec = poseAgeSec;
        if (_targetWorld.validVelocity)
        {
            velAgeSec = static_cast<float>((ctrlStartNow - _targetWorld.velocityTimestamp).seconds());
            if (velAgeSec < 0.0f)
            {
                velAgeSec = 0.0f;
            }
        }

        float leadDtSec = poseAgeSec;
        if (hasTargetVelocity)
        {
            leadDtSec = std::max(poseAgeSec, velAgeSec);
        }

        // Bù tuổi dữ liệu + chu kỳ update controller + phần lead thêm
        leadDtSec += std::max(dt_s, 0.0f);
        leadDtSec += std::max(_param_control_extra_lead_sec, 0.0f);
        leadDtSec = std::clamp(leadDtSec, 0.0f, _param_prediction_dt_max);

        // -------------------------------------------------
        // Dự đoán tương lai của target và drone
        // -------------------------------------------------
        Eigen::Vector3f targetFutureWorld = targetPositionWorld;
        Eigen::Vector3f droneFutureWorld = dronePosition;

        if (leadDtSec > 0.0f)
        {
            if (hasTargetVelocity)
            {
                targetFutureWorld += targetVelocityWorld * leadDtSec;
            }

            const float accGain = std::max(_param_predictive_acc_gain, 0.0f);

            droneFutureWorld.x() +=
                droneVelocity.x() * leadDtSec +
                0.5f * accGain * vehicleAccelerationXY.x() * leadDtSec * leadDtSec;

            droneFutureWorld.y() +=
                droneVelocity.y() * leadDtSec +
                0.5f * accGain * vehicleAccelerationXY.y() * leadDtSec * leadDtSec;
        }

        // Sai số điều khiển là sai số tương lai
        Eigen::Vector2f errorForControl(
            targetFutureWorld.x() - droneFutureWorld.x(),
            targetFutureWorld.y() - droneFutureWorld.y());

        _approach_altitude = std::abs(dronePosition.z());

        // Feedback chỉ xử lý phần sai số còn lại
        const Eigen::Vector2f velocityFeedback = pidVelocityXY(
            errorForControl.x(),
            errorForControl.y(),
            dt_s);

        // Feedforward vận tốc target
        Eigen::Vector2f velocityCommand = velocityFeedback;
        if (hasTargetVelocity)
        {
            velocityCommand.x() += targetVelocityWorld.x();
            velocityCommand.y() += targetVelocityWorld.y();
        }

        velocityCommand.x() = std::clamp(
            velocityCommand.x(),
            -_param_descent_max_velocity,
            _param_descent_max_velocity);

        velocityCommand.y() = std::clamp(
            velocityCommand.y(),
            -_param_descent_max_velocity,
            _param_descent_max_velocity);

        // Slew giữ nguyên, nhưng đặt ở lệnh cuối cùng
        _vxFilt = applySlew(velocityCommand.x(), _vxFilt, _param_slew_acc, dt_s);
        _vyFilt = applySlew(velocityCommand.y(), _vyFilt, _param_slew_acc, dt_s);

        Eigen::Vector2f velocityXY(
            std::clamp(_vxFilt, -_param_descent_max_velocity, _param_descent_max_velocity),
            std::clamp(_vyFilt, -_param_descent_max_velocity, _param_descent_max_velocity));

        float vz = computeDescentVelocity(errX, errY);

        if (!_yawSpInit)
        {
            _yaw_sp = px4_ros2::quaternionToYaw(_vehicle_attitude->attitude());
            _yawSpInit = true;
        }

        geometry_msgs::msg::PoseStamped debugPredMsg;
        debugPredMsg.header.stamp = ctrlStartNow;
        debugPredMsg.header.frame_id = "map";
        debugPredMsg.pose.position.x = targetFutureWorld.x();
        debugPredMsg.pose.position.y = targetFutureWorld.y();
        debugPredMsg.pose.position.z = targetFutureWorld.z();
        debugPredMsg.pose.orientation.w = 1.0;
        debugPredMsg.pose.orientation.x = 0.0;
        debugPredMsg.pose.orientation.y = 0.0;
        debugPredMsg.pose.orientation.z = 0.0;
        _debug_target_pred_pub->publish(debugPredMsg);

        const rclcpp::Time ctrlEndNow = _node.now();

        _trajectory_setpoint->update(
            Eigen::Vector3f(velocityXY.x(), velocityXY.y(), vz),
            std::nullopt,
            std::nullopt);

        const rclcpp::Time cmdPubNow = _node.now();

        publishPrecisionLandTiming(
            _debug_dt_pub,
            imageTimestamp.nanoseconds() != 0 ? imageTimestamp.seconds() : -1.0,
            _targetWorld.timestamp.seconds(),
            _targetWorld.validVelocity ? _targetWorld.velocityTimestamp.seconds() : -1.0,
            _targetPoseRxNow.nanoseconds() != 0 ? _targetPoseRxNow.seconds() : -1.0,
            _targetVelRxNow.nanoseconds() != 0 ? _targetVelRxNow.seconds() : -1.0,
            ctrlStartNow.seconds(),
            ctrlEndNow.seconds(),
            cmdPubNow.seconds());

        if (_land_detected)
        {
            switchToState(State::Finished);
        }
        break;
    }

    case State::Finished:
    {
        RCLCPP_WARN(_node.get_logger(), "[PL] Finished");

        std_msgs::msg::String msg;
        msg.data = "CENTER_LOOKUP_FOLLOW";
        _gimbal_seq_pub->publish(msg);

        ModeBase::completed(px4_ros2::Result::Success);
        return;
    }
    }
}

Eigen::Vector2f PrecisionLand::estimateVehicleAccelerationXY(float dt_s)
{
    const float dt = std::max(dt_s, 1e-3f);
    const Eigen::Vector3f vehicleVelocity = _vehicle_local_position->velocityNed();

    const float currentVelX = vehicleVelocity.x();
    const float currentVelY = vehicleVelocity.y();

    if (!_prevVehicleVelValid)
    {
        _prevVehicleVelX = currentVelX;
        _prevVehicleVelY = currentVelY;
        _prevVehicleVelValid = true;
        return Eigen::Vector2f(0.0f, 0.0f);
    }

    float accXRaw = (currentVelX - _prevVehicleVelX) / dt;
    float accYRaw = (currentVelY - _prevVehicleVelY) / dt;

    const float accMax = std::max(_param_predictive_acc_max, 0.0f);
    accXRaw = std::clamp(accXRaw, -accMax, accMax);
    accYRaw = std::clamp(accYRaw, -accMax, accMax);

    const float alpha = std::clamp(_param_predictive_acc_lpf_alpha, 0.0f, 1.0f);
    _vehicleAccXFilt = alpha * accXRaw + (1.0f - alpha) * _vehicleAccXFilt;
    _vehicleAccYFilt = alpha * accYRaw + (1.0f - alpha) * _vehicleAccYFilt;

    _prevVehicleVelX = currentVelX;
    _prevVehicleVelY = currentVelY;

    return Eigen::Vector2f(_vehicleAccXFilt, _vehicleAccYFilt);
}

Eigen::Vector2f PrecisionLand::predictFutureRelativeErrorQuadratic(
    float errX,
    float errY,
    float relativeVelX,
    float relativeVelY,
    float vehicleAccX,
    float vehicleAccY,
    float predictionDt) const
{
    const float dt = std::clamp(predictionDt, 0.0f, _param_prediction_dt_max);

    const float accGain = std::max(_param_predictive_acc_gain, 0.0f);

    const float errDDotX = -accGain * vehicleAccX;
    const float errDDotY = -accGain * vehicleAccY;

    const float errPredX = errX + relativeVelX * dt + 0.5f * errDDotX * dt * dt;
    const float errPredY = errY + relativeVelY * dt + 0.5f * errDDotY * dt * dt;

    return Eigen::Vector2f(errPredX, errPredY);
}

float PrecisionLand::applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dt_s) const
{
    const float dt = std::max(dt_s, 1e-3f);
    const float maxDeltaVelocity = accelLimit * dt;
    const float deltaVelocity = std::clamp(commandVelocity - previousVelocity, -maxDeltaVelocity, maxDeltaVelocity);
    return previousVelocity + deltaVelocity;
}

Eigen::Vector2f PrecisionLand::pidVelocityXY(
    float errPredX,
    float errPredY,
    float dt_s)
{
    const float dt = std::max(dt_s, 1e-3f);

    const float Xp = _param_descent_kp * errPredX;
    const float Yp = _param_descent_kp * errPredY;

    if (std::abs(errPredX) > _param_pid_deadband)
    {
        _velXIntegral += errPredX * dt;
    }

    if (std::abs(errPredY) > _param_pid_deadband)
    {
        _velYIntegral += errPredY * dt;
    }

    float Xi = 0.0f;
    float Yi = 0.0f;

    if (_param_descent_ki > 1e-6f)
    {
        const float maxIntegral = 0.15f * _param_descent_max_velocity / _param_descent_ki;
        _velXIntegral = std::clamp(_velXIntegral, -maxIntegral, maxIntegral);
        _velYIntegral = std::clamp(_velYIntegral, -maxIntegral, maxIntegral);

        Xi = _param_descent_ki * _velXIntegral;
        Yi = _param_descent_ki * _velYIntegral;
    }

    float Xd = 0.0f;
    float Yd = 0.0f;

    if (_param_descent_kd > 1e-6f && _prevErrPredValid)
    {
        const float errPredDotX = (errPredX - _prevErrPredX) / dt;
        const float errPredDotY = (errPredY - _prevErrPredY) / dt;

        Xd = _param_descent_kd * errPredDotX;
        Yd = _param_descent_kd * errPredDotY;
    }

    _prevErrPredX = errPredX;
    _prevErrPredY = errPredY;
    _prevErrPredValid = true;

    float vxFb = Xp + Xi + Xd;
    float vyFb = Yp + Yi + Yd;

    vxFb = std::clamp(vxFb, -_param_descent_max_velocity, _param_descent_max_velocity);
    vyFb = std::clamp(vyFb, -_param_descent_max_velocity, _param_descent_max_velocity);

    return Eigen::Vector2f(vxFb, vyFb);
}

float PrecisionLand::computeDescentVelocity(float errX, float errY)
{
    const float z = std::abs(_vehicle_local_position->positionNed().z());

    if (z < _param_land_zone_z)
    {
        return std::abs(_param_descent_vel);
    }

    const float lateralError = std::sqrt(errX * errX + errY * errY);

    if (lateralError >= _param_descent_gate_radius)
    {
        return 0.0f;
    }

    const float scale =
        1.0f - (lateralError / std::max(_param_descent_gate_radius, 1e-6f));

    const float scaleClamped = std::clamp(scale, 0.0f, 1.0f);

    return _param_vmin + (_param_vmax - _param_vmin) * scaleClamped;
}

bool PrecisionLand::checkTargetTimeout() const
{
    if (!_targetWorld.validPose)
    {
        return true;
    }

    return ((_node.now() - _targetWorld.timestamp).seconds() > _param_target_timeout);
}

std::string PrecisionLand::stateName(State state) const
{
    switch (state)
    {
    case State::Search:
        return "Search";
    case State::Descend:
        return "Descend";
    case State::Finished:
        return "Finished";
    default:
        return "Unknown";
    }
}

void PrecisionLand::switchToState(State state)
{
    _state = state;
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<PrecisionLand>>(kModeName, kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}