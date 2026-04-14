#include "PrecisionLand.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_command_ack.hpp>

namespace
{
const std::string kModeName = "PLHEOC";
constexpr bool kEnableDebugOutput = true;

/**
 * Publish debug timing cho node PrecisionLand.
 *
 * Input:
 *     pub: publisher debug timing
 *     imageStampSec: stamp ảnh gốc để gom end-to-end
 *     poseStampSec: stamp pose filtered sau Kalman
 *     velStampSec: stamp velocity filtered sau Kalman
 *     poseRxNowSec: thời điểm callback nhận pose
 *     velRxNowSec: thời điểm callback nhận velocity
 *     ctrlStartNowSec: thời điểm bắt đầu tính điều khiển
 *     ctrlEndNowSec: thời điểm tính điều khiển xong
 *     cmdPubNowSec: thời điểm gửi setpoint đi
 *
 * Logic:
 *     - imageStampSec dùng để cộng end-to-end toàn pipeline
 *     - poseStampSec/velStampSec dùng để tính tuổi dữ liệu còn lại sau Kalman
 *
 * Output:
 *     Publish JSON string lên /debug_dt/precision_land
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
} // namespace

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

    _vehicle_command_pub =
        _node.create_publisher<px4_msgs::msg::VehicleCommand>(
            "/fmu/in/vehicle_command",
            rclcpp::QoS(10).best_effort());

    _vehicle_command_ack_sub =
        _node.create_subscription<px4_msgs::msg::VehicleCommandAck>(
            "/fmu/out/vehicle_command_ack",
            rclcpp::QoS(10).best_effort(),
            std::bind(&PrecisionLand::vehicleCommandAckCallback, this, std::placeholders::_1));

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

    _node.declare_parameter<float>("descent_kp_pid", 0.9f);
    _node.declare_parameter<float>("descent_ki_pid", 0.03f);
    _node.declare_parameter<float>("descent_kd_pid", 0.0f);
    _node.declare_parameter<float>("descent_max_velocity", 10.0f);
    _node.declare_parameter<float>("slew_acc", 18.0f);

    _node.declare_parameter<float>("land_zone_z", 0.5f);
    _node.declare_parameter<float>("descent_vel", 0.4f);

    _node.declare_parameter<float>("descent_gate_radius", 0.3f);
    _node.declare_parameter<float>("vmin", 0.45f);
    _node.declare_parameter<float>("vmax", 0.8f);
    _node.declare_parameter<float>("disarm_height", 0.06f);

    _node.declare_parameter<bool>("use_predictive_error", true);
    _node.declare_parameter<float>("prediction_dt_max", 0.75f);
    _node.declare_parameter<float>("control_extra_lead_sec", 0.25f);

    _node.declare_parameter<float>("predictive_acc_gain", 0.0f);
    _node.declare_parameter<float>("predictive_acc_lpf_alpha", 0.4f);
    _node.declare_parameter<float>("predictive_acc_max", 4.0f);

    _node.get_parameter("topics.target_pose", _targetPoseTopic);
    _node.get_parameter("topics.target_velocity", _targetVelocityTopic);
    _node.get_parameter("topics.vehicle_land_detected", _vehicleLandDetectedTopic);
    _node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);
    _node.get_parameter("topics.gimbal_command", _gimbalCommandTopic);
    _node.get_parameter("topics.gimbal_attitude", _gimbalAttitudeTopic);

    _node.get_parameter("disarm_height", _param_disarm_height);

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

    precision_land::XYControllerParams xyParams;
    xyParams.kp = _param_descent_kp;
    xyParams.ki = _param_descent_ki;
    xyParams.kd = _param_descent_kd;
    xyParams.deadband = _param_pid_deadband;
    xyParams.maxVelocity = _param_descent_max_velocity;
    xyParams.slewAcc = _param_slew_acc;
    _xyVelocityController.configure(xyParams);

    precision_land::ZControllerParams zParams;
    zParams.landZoneZ = _param_land_zone_z;
    zParams.descentVel = _param_descent_vel;
    zParams.descentGateRadius = _param_descent_gate_radius;
    zParams.vmin = _param_vmin;
    zParams.vmax = _param_vmax;
    zParams.disarmHeight = _param_disarm_height;
    _descentZController.configure(zParams);
}

void PrecisionLand::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (std::isfinite(msg->dist_bottom) && msg->dist_bottom > 0.0f)
    {
        z_dist_bottom = msg->dist_bottom;
        _dist_bottom_valid = true;
    }
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
    _prevVehicleVelX = 0.0f;
    _prevVehicleVelY = 0.0f;
    _vehicleAccXFilt = 0.0f;
    _vehicleAccYFilt = 0.0f;
    _prevVehicleVelValid = false;

    _xyVelocityController.reset();
    _disarm_sent = false;
    _dist_bottom_valid = false;
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
    updateTargetLostStatus(targetLost);

    switch (_state)
    {
    case State::Search:
    {
        handleSearchState(targetLost);
        break;
    }

    case State::Descend:
    {
        handleDescendState(dt_s, targetLost);
        break;
    }

    case State::Finished:
    {
        handleFinishedState();
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

void PrecisionLand::updateTargetLostStatus(bool targetLost)
{
    if (targetLost && !_target_lost_prev)
    {
        RCLCPP_INFO(_node.get_logger(), "Target lost (state=%s)", stateName(_state).c_str());
    }
    else if (!targetLost && _target_lost_prev)
    {
        RCLCPP_INFO(_node.get_logger(), "Target acquired");
    }

    _target_lost_prev = targetLost;
}

void PrecisionLand::handleSearchState(bool targetLost)
{
    if (!targetLost && _targetWorld.validPose)
    {
        switchToState(State::Descend);
        return;
    }

    Hover();
}

void PrecisionLand::handleDescendState(float dt_s, bool targetLost)
{
    if (targetLost)
    {
        switchToState(State::Search);
        return;
    }

    const rclcpp::Time ctrlStartNow = _node.now();

    const precision_land::PredictionInput predictionInput = buildPredictionInput(dt_s, ctrlStartNow);
    const precision_land::PredictionOutput predictionOutput = _predictionModel.predict(predictionInput);

    _approach_altitude = std::abs(predictionInput.vehicle.positionWorld.z());

    precision_land::XYControllerInput xyInput;
    xyInput.futureErrorXY = predictionOutput.futureErrorXY;
    xyInput.targetVelocityXY = predictionInput.target.velocityWorld.head<2>();
    xyInput.useTargetFeedforward = predictionInput.target.hasVelocity;
    xyInput.dtSec = dt_s;

    const precision_land::XYControllerOutput xyOutput = _xyVelocityController.update(xyInput);

    precision_land::ZControllerOutput zOutput{};
    float vz = 0.0f;

    // ----- debug hien tai de tam nhu nay chua sua -----
    _dist_bottom_valid = true;
    z_dist_bottom = _approach_altitude;

    if (_dist_bottom_valid)
    {
    precision_land::ZControllerInput zInput;
    zInput.futureErrorXY = predictionOutput.futureErrorXY;
    zInput.vehicleAltitudeAbs = std::abs(z_dist_bottom);

        zOutput = _descentZController.computeCommand(zInput);
        vz = zOutput.vzCommand;
    }
    else
    {
        vz = 0.0f;
    }

    if (!_yawSpInit)
    {
        _yaw_sp = px4_ros2::quaternionToYaw(_vehicle_attitude->attitude());
        _yawSpInit = true;
    }

    publishPredictedTargetDebug(ctrlStartNow, predictionOutput.targetFutureWorld);

    const rclcpp::Time ctrlEndNow = _node.now();

    _trajectory_setpoint->update(
        Eigen::Vector3f(xyOutput.velocitySpXY.x(), xyOutput.velocitySpXY.y(), vz),
        std::nullopt,
        std::nullopt);

    const rclcpp::Time cmdPubNow = _node.now();
    publishTimingDebug(ctrlStartNow, ctrlEndNow, cmdPubNow);

    if (_dist_bottom_valid && zOutput.shouldDisarm )
    {
        sendDisarmCommand();
    }

    if (_land_detected)
    {
        switchToState(State::Finished);
    }
}

void PrecisionLand::handleFinishedState()
{
    RCLCPP_WARN(_node.get_logger(), "[PL] Finished");

    std_msgs::msg::String msg;
    msg.data = "CENTER_LOOKUP_FOLLOW";
    _gimbal_seq_pub->publish(msg);

    ModeBase::completed(px4_ros2::Result::Success);
}

float PrecisionLand::computeLeadTimeSec(float dt_s, const rclcpp::Time &ctrlStartNow) const
{
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
    if (_param_use_predictive_error && _targetWorld.validVelocity)
    {
        leadDtSec = std::max(poseAgeSec, velAgeSec);
    }

    leadDtSec += std::max(dt_s, 0.0f);
    leadDtSec += std::max(_param_control_extra_lead_sec, 0.0f);

    return std::clamp(leadDtSec, 0.0f, _param_prediction_dt_max);
}

precision_land::PredictionInput PrecisionLand::buildPredictionInput(float dt_s, const rclcpp::Time &ctrlStartNow)
{
    precision_land::PredictionInput input;

    input.leadDtSec = computeLeadTimeSec(dt_s, ctrlStartNow);
    input.predictiveAccGain = std::max(_param_predictive_acc_gain, 0.0f);

    input.vehicle.positionWorld = _vehicle_local_position->positionNed();
    input.vehicle.velocityWorld = _vehicle_local_position->velocityNed();
    input.vehicle.accelerationXY = estimateVehicleAccelerationXY(dt_s);

    input.target.positionWorld = Eigen::Vector3f(
        static_cast<float>(_targetWorld.position.x()),
        static_cast<float>(_targetWorld.position.y()),
        static_cast<float>(_targetWorld.position.z()));

    input.target.hasVelocity = _param_use_predictive_error && _targetWorld.validVelocity;
    if (input.target.hasVelocity)
    {
        input.target.velocityWorld = Eigen::Vector3f(
            static_cast<float>(_targetWorld.velocity.x()),
            static_cast<float>(_targetWorld.velocity.y()),
            static_cast<float>(_targetWorld.velocity.z()));
    }

    return input;
}

void PrecisionLand::publishPredictedTargetDebug(const rclcpp::Time &stamp, const Eigen::Vector3f &targetFutureWorld)
{
    geometry_msgs::msg::PoseStamped debugPredMsg;
    debugPredMsg.header.stamp = stamp;
    debugPredMsg.header.frame_id = "map";
    debugPredMsg.pose.position.x = targetFutureWorld.x();
    debugPredMsg.pose.position.y = targetFutureWorld.y();
    debugPredMsg.pose.position.z = targetFutureWorld.z();
    debugPredMsg.pose.orientation.w = 1.0;
    debugPredMsg.pose.orientation.x = 0.0;
    debugPredMsg.pose.orientation.y = 0.0;
    debugPredMsg.pose.orientation.z = 0.0;

    _debug_target_pred_pub->publish(debugPredMsg);
}

void PrecisionLand::publishTimingDebug(
    const rclcpp::Time &ctrlStartNow,
    const rclcpp::Time &ctrlEndNow,
    const rclcpp::Time &cmdPubNow)
{
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

/**
 * Publish vehicle command tới PX4.
 *
 * Input:
 *     command: mã lệnh PX4
 *     param1: tham số 1 của lệnh
 *     param2: tham số 2 của lệnh
 *
 * Logic:
 *     đóng gói VehicleCommand và publish qua DDS bridge
 *
 * Output:
 *     publish lên /fmu/in/vehicle_command
 */
void PrecisionLand::publishVehicleCommand(uint16_t command, float param1, float param2)
{
    px4_msgs::msg::VehicleCommand msg{};
    msg.timestamp = _node.now().nanoseconds() / 1000;
    msg.param1 = param1;
    msg.param2 = param2;
    msg.command = command;
    msg.target_system = 1;
    msg.target_component = 1;
    msg.source_system = 1;
    msg.source_component = 1;
    msg.from_external = true;

    _vehicle_command_pub->publish(msg);
}

/**
 * Gửi lệnh disarm đúng 1 lần.
 *
 * Input:
 *     không có
 *
 * Logic:
 *     nếu chưa gửi thì gửi lệnh disarm tới PX4
 *
 * Output:
 *     publish lệnh disarm
 */
void PrecisionLand::sendDisarmCommand()
{
    if (_disarm_sent)
    {
        return;
    }

    publishVehicleCommand(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM,0.0f,0.0f);

    _disarm_sent = true;
    RCLCPP_WARN(_node.get_logger(), "[PL] GUI LENH DISARM");
}

/**
 * Callback nhận phản hồi command từ PX4.
 *
 * Input:
 *     msg: phản hồi VehicleCommandAck
 *
 * Logic:
 *     log kết quả phản hồi cho lệnh arm/disarm
 *
 * Output:
 *     không có
 */
void PrecisionLand::vehicleCommandAckCallback(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg)
{
    if (msg->command == px4_msgs::msg::VehicleCommand::VEHICLE_CMD_COMPONENT_ARM_DISARM)
    {
        switchToState(State::Finished);
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<PrecisionLand>>(kModeName, kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}