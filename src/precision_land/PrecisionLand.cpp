#include "PrecisionLand.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/utils/geometry.hpp>

#include <px4_msgs/msg/vehicle_command.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
const std::string kModeName = "PLHEOC";
constexpr bool kEnableDebugOutput = true;

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

precision_land::DisarmMode parseDisarmMode(const std::string &value)
{
    if (value == "disabled")
    {
        return precision_land::DisarmMode::Disabled;
    }

    return precision_land::DisarmMode::Enabled;
}

precision_land::DisarmAltitudeSource parseDisarmAltitudeSource(const std::string &value)
{
    if (value == "local_position_z")
    {
        return precision_land::DisarmAltitudeSource::LocalPositionZ;
    }

    return precision_land::DisarmAltitudeSource::DistBottom;
}
} // namespace

using namespace px4_ros2::literals;

PrecisionLand::PrecisionLand(rclcpp::Node &node)
    : ModeBase(node, kModeName),
      _node(node)
{
    _trajectorySetpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
    _vehicleLocalPosition = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
    _vehicleAttitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);

    // Doc param truoc de su dung topic dung theo yaml
    loadParameters();

    _targetPoseRawSub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetPoseRawTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::targetPoseRawCallback, this, std::placeholders::_1));

    _targetPoseSub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetPoseTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::targetPoseCallback, this, std::placeholders::_1));

    _targetVelocitySub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetVelocityTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::targetVelocityCallback, this, std::placeholders::_1));

    _vehicleLandDetectedSub =
        _node.create_subscription<px4_msgs::msg::VehicleLandDetected>(
            _vehicleLandDetectedTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::vehicleLandDetectedCallback, this, std::placeholders::_1));

    _vehicleLocalPosSub =
        _node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            _vehicleLocalPositionTopic,
            rclcpp::QoS(1).best_effort(),
            std::bind(&PrecisionLand::vehicleLocalPositionCallback, this, std::placeholders::_1));

    _gimbalSub =
        _node.create_subscription<geometry_msgs::msg::Vector3>(
            _gimbalAttitudeTopic,
            rclcpp::QoS(10).best_effort(),
            std::bind(&PrecisionLand::gimbalAttCallback, this, std::placeholders::_1));

    _gimbalSeqPub =
        _node.create_publisher<std_msgs::msg::String>(
            _gimbalCommandTopic,
            rclcpp::QoS(1).best_effort());

    _debugTargetPredPub =
        _node.create_publisher<geometry_msgs::msg::PoseStamped>(
            "/debug/precision_land/target_pose_pred_world",
            rclcpp::QoS(1).best_effort());

    _debugDtPub =
        _node.create_publisher<std_msgs::msg::String>(
            "/debug_dt/precision_land",
            rclcpp::QoS(10).best_effort());

    _vehicleCommandPub =
        _node.create_publisher<px4_msgs::msg::VehicleCommand>(
            "/fmu/in/vehicle_command",
            rclcpp::QoS(10).best_effort());

    precision_land::DisarmControllerParams disarmParams;
    disarmParams.mode = parseDisarmMode(_paramDisarmMode);
    disarmParams.altitudeSource = parseDisarmAltitudeSource(_paramDisarmAltitudeSource);
    disarmParams.disarmHeight = _paramDisarmHeight;
    disarmParams.lateralErrorThreshold = _paramDisarmLateralErrorThreshold;
    disarmParams.verticalSpeedThreshold = _paramDisarmVerticalSpeedThreshold;
    disarmParams.allowLandedImmediateDisarm = _paramDisarmAllowLandedImmediate;

    _disarmController.configure(disarmParams, &_node, _vehicleCommandPub);

    _vehicleCommandAckSub =
        _node.create_subscription<px4_msgs::msg::VehicleCommandAck>(
            "/fmu/out/vehicle_command_ack",
            rclcpp::QoS(10).best_effort(),
            std::bind(&PrecisionLand::vehicleCommandAckCallback, this, std::placeholders::_1));

    modeRequirements().manual_control = false;
}

/**
 * Load toan bo parameter cho node PrecisionLand.
 *
 * Input:
 *     khong co
 *
 * Logic:
 *     - Khai bao va doc topics
 *     - Cau hinh XY controller, Z controller, DisarmController
 *     - Bat/tat debug logger
 *
 * Output:
 *     cap nhat bien noi bo
 */
void PrecisionLand::loadParameters()
{
    _node.declare_parameter<std::string>("topics.target_pose_raw", "/KalmanFilter/target_pose_NED");
    _node.declare_parameter<std::string>("topics.target_pose", "/KalmanFilter/target_pose_est_NED");
    _node.declare_parameter<std::string>("topics.target_velocity", "/KalmanFilter/target_velocity_est_NED");
    _node.declare_parameter<std::string>("topics.vehicle_land_detected", "/fmu/out/vehicle_land_detected");
    _node.declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position");
    _node.declare_parameter<std::string>("topics.gimbal_command", "/gimbal/cmd/sequence");
    _node.declare_parameter<std::string>("topics.gimbal_attitude", "/gimbal/state/attitude");

    _node.declare_parameter<float>("PID_deadband", 0.05f);
    _node.declare_parameter<float>("target_timeout", 3.0f);

    _node.declare_parameter<float>("descent_kp_pid", 0.9f);
    _node.declare_parameter<float>("descent_ki_pid", 0.01f);
    _node.declare_parameter<float>("descent_kd_pid", 0.0f);
    _node.declare_parameter<float>("descent_max_velocity", 10.0f);
    _node.declare_parameter<float>("slew_acc", 10.0f);

    _node.declare_parameter<float>("land_zone_z", 0.5f);
    _node.declare_parameter<float>("descent_vel", 0.5f);

    _node.declare_parameter<float>("descent_gate_radius", 0.3f);
    _node.declare_parameter<float>("vmin", 0.45f);
    _node.declare_parameter<float>("vmax", 0.8f);

    _node.declare_parameter<bool>("use_predictive_error", true);
    _node.declare_parameter<float>("prediction_dt_max", 0.75f);
    _node.declare_parameter<float>("control_extra_lead_sec", 0.25f);

    _node.declare_parameter<float>("predictive_acc_gain", 0.0f);
    _node.declare_parameter<float>("predictive_acc_lpf_alpha", 0.4f);
    _node.declare_parameter<float>("predictive_acc_max", 4.0f);

    _node.declare_parameter<std::string>("disarm.mode", "enabled");
    _node.declare_parameter<std::string>("disarm.altitude_source", "dist_bottom");
    _node.declare_parameter<float>("disarm.height", 0.06f);
    _node.declare_parameter<float>("disarm.lateral_error_threshold", 0.10f);
    _node.declare_parameter<float>("disarm.vertical_speed_threshold", 0.15f);
    _node.declare_parameter<bool>("disarm.allow_landed_immediate", true);

    _node.declare_parameter<bool>("debug_logger", true);

    _node.get_parameter("topics.target_pose_raw", _targetPoseRawTopic);
    _node.get_parameter("topics.target_pose", _targetPoseTopic);
    _node.get_parameter("topics.target_velocity", _targetVelocityTopic);
    _node.get_parameter("topics.vehicle_land_detected", _vehicleLandDetectedTopic);
    _node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);
    _node.get_parameter("topics.gimbal_command", _gimbalCommandTopic);
    _node.get_parameter("topics.gimbal_attitude", _gimbalAttitudeTopic);

    _node.get_parameter("PID_deadband", _paramPidDeadband);
    _node.get_parameter("target_timeout", _paramTargetTimeout);

    _node.get_parameter("descent_kp_pid", _paramDescentKp);
    _node.get_parameter("descent_ki_pid", _paramDescentKi);
    _node.get_parameter("descent_kd_pid", _paramDescentKd);
    _node.get_parameter("descent_max_velocity", _paramDescentMaxVelocity);
    _node.get_parameter("slew_acc", _paramSlewAcc);

    _node.get_parameter("land_zone_z", _paramLandZoneZ);
    _node.get_parameter("descent_vel", _paramDescentVel);

    _node.get_parameter("descent_gate_radius", _paramDescentGateRadius);
    _node.get_parameter("vmin", _paramVmin);
    _node.get_parameter("vmax", _paramVmax);

    _node.get_parameter("use_predictive_error", _paramUsePredictiveError);
    _node.get_parameter("prediction_dt_max", _paramPredictionDtMax);
    _node.get_parameter("control_extra_lead_sec", _paramControlExtraLeadSec);

    _node.get_parameter("predictive_acc_gain", _paramPredictiveAccGain);
    _node.get_parameter("predictive_acc_lpf_alpha", _paramPredictiveAccLpfAlpha);
    _node.get_parameter("predictive_acc_max", _paramPredictiveAccMax);

    _node.get_parameter("disarm.mode", _paramDisarmMode);
    _node.get_parameter("disarm.altitude_source", _paramDisarmAltitudeSource);
    _node.get_parameter("disarm.height", _paramDisarmHeight);
    _node.get_parameter("disarm.lateral_error_threshold", _paramDisarmLateralErrorThreshold);
    _node.get_parameter("disarm.vertical_speed_threshold", _paramDisarmVerticalSpeedThreshold);
    _node.get_parameter("disarm.allow_landed_immediate", _paramDisarmAllowLandedImmediate);

    _node.get_parameter("debug_logger", _paramDebugLogger);

    try
    {
        precision_land::XYControllerParams xyParams;
        xyParams.kp = _paramDescentKp;
        xyParams.ki = _paramDescentKi;
        xyParams.kd = _paramDescentKd;
        xyParams.deadband = _paramPidDeadband;
        xyParams.maxVelocity = _paramDescentMaxVelocity;
        xyParams.slewAcc = _paramSlewAcc;
        _xyVelocityController.configure(xyParams);

        precision_land::ZControllerParams zParams;
        zParams.landZoneZ = _paramLandZoneZ;
        zParams.descentVel = _paramDescentVel;
        zParams.descentGateRadius = _paramDescentGateRadius;
        zParams.vmin = _paramVmin;
        zParams.vmax = _paramVmax;
        zParams.disarmHeight = _paramDisarmHeight;
        _descentZController.configure(zParams);

        _debugLogger.setEnabled(_paramDebugLogger);
        _pipelineTimingCollector.setEnabled(_paramDebugLogger);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR(_node.get_logger(), "[PL] Loi loadParameters: %s", e.what());
        throw;
    }
    catch (...)
    {
        RCLCPP_ERROR(_node.get_logger(), "[PL] Loi loadParameters khong xac dinh");
        throw;
    }
}

void PrecisionLand::targetPoseRawCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    _latestTargetRawWorld.x() = static_cast<float>(msg->pose.position.x);
    _latestTargetRawWorld.y() = static_cast<float>(msg->pose.position.y);
    _latestTargetRawWorld.z() = static_cast<float>(msg->pose.position.z);
    _latestTargetRawValid = true;
}

void PrecisionLand::targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_searchStarted)
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
    _imageTimestamp = msgTimestamp;
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

void PrecisionLand::vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg)
{
    _landDetected = msg->landed;
}

void PrecisionLand::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (std::isfinite(msg->dist_bottom) && msg->dist_bottom > 0.0f)
    {
        _zDistBottom = msg->dist_bottom;
        _distBottomValid = true;
    }
    else
    {
        _distBottomValid = false;
    }
}

void PrecisionLand::gimbalAttCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
    _gimbalPitchDeg = static_cast<float>(msg->y);
    _gimbalReady = std::abs(_gimbalPitchDeg) > 80.0f;

    const double yaw = msg->x * M_PI / 180.0;
    const double pitch = msg->y * M_PI / 180.0;
    const double roll = msg->z * M_PI / 180.0;

    _qGimbal =
        Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

    _qGimbal.normalize();
    _gimbalValid = true;
}

void PrecisionLand::vehicleCommandAckCallback(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg)
{
    try
    {
        const precision_land::DisarmDecisionStatus disarmStatus = _disarmController.handleAck(msg);

        if (disarmStatus == precision_land::DisarmDecisionStatus::Accepted)
        {
            RCLCPP_WARN(_node.get_logger(), "[PL] PX4 chap nhan DISARM -> chuyen Finished");
            switchToState(State::Finished);
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi vehicleCommandAckCallback: %s",
            e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi vehicleCommandAckCallback khong xac dinh");
    }
}

void PrecisionLand::onActivate()
{
    _prevVehicleVelX = 0.0f;
    _prevVehicleVelY = 0.0f;
    _vehicleAccXFilt = 0.0f;
    _vehicleAccYFilt = 0.0f;
    _prevVehicleVelValid = false;

    _xyVelocityController.reset();
    _disarmController.reset();

    _searchStarted = true;
    _targetLostPrev = true;
    _distBottomValid = false;
    _landDetected = false;
    _yawSpInit = false;

    _latestTargetRawWorld.setZero();
    _latestTargetRawValid = false;

    try
    {
        _debugLogger.startSession();

        if (_debugLogger.isEnabled())
        {
            _pipelineTimingCollector.startSession(_node, _debugLogger.getSessionStamp());
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi khoi tao debug session: %s",
            e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi khoi tao debug session khong xac dinh");
    }

    switchToState(State::Search);
}

void PrecisionLand::onDeactivate()
{
    _searchStarted = false;

    try
    {
        _pipelineTimingCollector.close();
        _debugLogger.close();
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi dong debug session: %s",
            e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi dong debug session khong xac dinh");
    }
}

void PrecisionLand::Hover()
{
    _trajectorySetpoint->update(
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        std::nullopt,
        std::nullopt);
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
    const Eigen::Vector3f vehicleVelocity = _vehicleLocalPosition->velocityNed();

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

    const float accMax = std::max(_paramPredictiveAccMax, 0.0f);
    accXRaw = std::clamp(accXRaw, -accMax, accMax);
    accYRaw = std::clamp(accYRaw, -accMax, accMax);

    const float alpha = std::clamp(_paramPredictiveAccLpfAlpha, 0.0f, 1.0f);
    _vehicleAccXFilt = alpha * accXRaw + (1.0f - alpha) * _vehicleAccXFilt;
    _vehicleAccYFilt = alpha * accYRaw + (1.0f - alpha) * _vehicleAccYFilt;

    _prevVehicleVelX = currentVelX;
    _prevVehicleVelY = currentVelY;

    return Eigen::Vector2f(_vehicleAccXFilt, _vehicleAccYFilt);
}

void PrecisionLand::updateTargetLostStatus(bool targetLost)
{
    if (targetLost && !_targetLostPrev)
    {
        RCLCPP_INFO(_node.get_logger(), "Target lost (state=%s)", stateName(_state).c_str());
    }
    else if (!targetLost && _targetLostPrev)
    {
        RCLCPP_INFO(_node.get_logger(), "Target acquired");
    }

    _targetLostPrev = targetLost;
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

void PrecisionLand::fillDebugTimingSample(
    precision_land::PrecisionLandDebugSample &sample,
    const rclcpp::Time &ctrlStartNow,
    const rclcpp::Time &ctrlEndNow,
    const rclcpp::Time &cmdPubNow) const
{
    sample.timing.poseWaitDt =
        (_targetPoseRxNow.nanoseconds() != 0) ? (ctrlStartNow - _targetPoseRxNow).seconds() : -1.0;

    sample.timing.velWaitDt =
        (_targetVelRxNow.nanoseconds() != 0) ? (ctrlStartNow - _targetVelRxNow).seconds() : -1.0;

    sample.timing.controlProcessingDt = (ctrlEndNow - ctrlStartNow).seconds();
    sample.timing.sendCmdDt = (cmdPubNow - ctrlEndNow).seconds();
    sample.timing.totalImageToCmdDt =
        (_imageTimestamp.nanoseconds() != 0) ? (cmdPubNow - _imageTimestamp).seconds() : -1.0;
}

void PrecisionLand::logDebugSample(
    const rclcpp::Time &ctrlStartNow,
    const rclcpp::Time &ctrlEndNow,
    const rclcpp::Time &cmdPubNow,
    const precision_land::PredictionOutput &predictionOutput,
    const precision_land::XYControllerInput &xyInput,
    const precision_land::XYControllerOutput &xyOutput,
    const precision_land::DisarmControllerOutput &disarmOutput,
    float vz,
    float altitudeNow)
{
    if (!_debugLogger.isEnabled())
    {
        return;
    }

    try
    {
        precision_land::PrecisionLandDebugSample sample;
        sample.timeSec = _node.now().seconds();
        sample.state = stateName(_state);

        sample.dronePos = _vehicleLocalPosition->positionNed();
        sample.droneVel = _vehicleLocalPosition->velocityNed();

        sample.targetRaw = _latestTargetRawValid ? _latestTargetRawWorld : Eigen::Vector3f::Zero();

        sample.targetEst = Eigen::Vector3f(
            static_cast<float>(_targetWorld.position.x()),
            static_cast<float>(_targetWorld.position.y()),
            static_cast<float>(_targetWorld.position.z()));

        sample.targetPred = predictionOutput.targetFutureWorld;

        sample.targetVel = _targetWorld.validVelocity
            ? Eigen::Vector3f(
                static_cast<float>(_targetWorld.velocity.x()),
                static_cast<float>(_targetWorld.velocity.y()),
                static_cast<float>(_targetWorld.velocity.z()))
            : Eigen::Vector3f::Zero();

        sample.errorXY.x() = sample.targetEst.x() - sample.dronePos.x();
        sample.errorXY.y() = sample.targetEst.y() - sample.dronePos.y();
        sample.futureErrorXY = predictionOutput.futureErrorXY;

        sample.pidOutXY = xyOutput.feedbackXY;
        sample.ffXY = xyInput.useTargetFeedforward ? xyInput.targetVelocityXY : Eigen::Vector2f::Zero();

        sample.finalSp.x() = xyOutput.velocitySpXY.x();
        sample.finalSp.y() = xyOutput.velocitySpXY.y();
        sample.finalSp.z() = vz;

        sample.altitudeAbs = altitudeNow;
        sample.distBottom = _distBottomValid ? _zDistBottom : -1.0f;

        sample.shouldDisarm =
            (disarmOutput.status == precision_land::DisarmDecisionStatus::WaitingAck) ||
            (disarmOutput.status == precision_land::DisarmDecisionStatus::Accepted);

        sample.landDetected = _landDetected;

        fillDebugTimingSample(sample, ctrlStartNow, ctrlEndNow, cmdPubNow);

        _debugLogger.logSample(sample);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi debug logger: %s",
            e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            2000,
            "[PL] Loi debug logger khong xac dinh");
    }
}

void PrecisionLand::handleDescendState(float dt_s, bool targetLost)
{
    const float altitudeNow =
        _distBottomValid ? std::abs(_zDistBottom)
                         : std::abs(_vehicleLocalPosition->positionNed().z());

    const bool allowBlindFinalDescent =
        targetLost && _targetWorld.validPose && (altitudeNow <= _paramLandZoneZ);

    if (targetLost && !allowBlindFinalDescent)
    {
        switchToState(State::Search);
        return;
    }

    const rclcpp::Time ctrlStartNow = _node.now();

    precision_land::PredictionOutput predictionOutput{};
    precision_land::PredictionInput predictionInput{};
    precision_land::XYControllerInput xyInput{};
    precision_land::XYControllerOutput xyOutput{};
    precision_land::DisarmControllerOutput disarmOutput{};
    float lateralError = 0.0f;
    float vz = 0.0f;

    try
    {
        predictionInput = buildPredictionInput(dt_s, ctrlStartNow);
        predictionOutput = _predictionModel.predict(predictionInput);

        _approachAltitude = std::abs(predictionInput.vehicle.positionWorld.z());

        xyInput.futureErrorXY = predictionOutput.futureErrorXY;
        xyInput.dtSec = dt_s;

        const bool canUseTrackedVelocity =
            _targetWorld.validVelocity && (!targetLost || allowBlindFinalDescent);

        if (canUseTrackedVelocity)
        {
            xyInput.targetVelocityXY = predictionInput.target.velocityWorld.head<2>().eval();
        }
        else
        {
            xyInput.targetVelocityXY = Eigen::Vector2f::Zero();
        }

        xyInput.useTargetFeedforward = canUseTrackedVelocity;

        xyOutput = _xyVelocityController.update(xyInput);

        lateralError = std::sqrt(
            predictionOutput.futureErrorXY.x() * predictionOutput.futureErrorXY.x() +
            predictionOutput.futureErrorXY.y() * predictionOutput.futureErrorXY.y());

        float altitudeForZ = 0.0f;
        bool altitudeForZValid = false;

        if (_distBottomValid)
        {
            altitudeForZ = std::abs(_zDistBottom);
            altitudeForZValid = true;
        }
        else
        {
            altitudeForZ = _approachAltitude;
            altitudeForZValid = true;
        }

        if (altitudeForZValid)
        {
            if (allowBlindFinalDescent)
            {
                vz = std::abs(_paramDescentVel);
            }
            else
            {
                precision_land::ZControllerInput zInput;
                zInput.futureErrorXY = predictionOutput.futureErrorXY;
                zInput.vehicleAltitudeAbs = altitudeForZ;

                const precision_land::ZControllerOutput zOutput = _descentZController.computeCommand(zInput);
                vz = zOutput.vzCommand;
            }
        }
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            1000,
            "[PL] Loi handleDescendState: %s",
            e.what());

        _trajectorySetpoint->update(Eigen::Vector3f::Zero(), std::nullopt, std::nullopt);
        return;
    }
    catch (...)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            1000,
            "[PL] Loi handleDescendState khong xac dinh");

        _trajectorySetpoint->update(Eigen::Vector3f::Zero(), std::nullopt, std::nullopt);
        return;
    }

    if (!_yawSpInit)
    {
        _yawSp = px4_ros2::quaternionToYaw(_vehicleAttitude->attitude());
        _yawSpInit = true;
    }

    publishPredictedTargetDebug(ctrlStartNow, predictionOutput.targetFutureWorld);

    const rclcpp::Time ctrlEndNow = _node.now();

    _trajectorySetpoint->update(
        Eigen::Vector3f(xyOutput.velocitySpXY.x(), xyOutput.velocitySpXY.y(), vz),
        std::nullopt,
        std::nullopt);

    const rclcpp::Time cmdPubNow = _node.now();

    publishTimingDebug(ctrlStartNow, ctrlEndNow, cmdPubNow);

    try
    {
        precision_land::DisarmControllerInput disarmInput;
        disarmInput.distBottomValid = _distBottomValid;
        disarmInput.distBottom = _zDistBottom;
        disarmInput.localPositionZValid = true;
        disarmInput.localPositionZ = _vehicleLocalPosition->positionNed().z();
        disarmInput.lateralError = lateralError;
        disarmInput.verticalSpeedAbs = std::abs(_vehicleLocalPosition->velocityNed().z());
        disarmInput.landed = _landDetected;

        disarmOutput = _disarmController.update(disarmInput);

        logDebugSample(
            ctrlStartNow,
            ctrlEndNow,
            cmdPubNow,
            predictionOutput,
            xyInput,
            xyOutput,
            disarmOutput,
            vz,
            altitudeNow);

        RCLCPP_WARN_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            500,
            "[PL] Disarm input: distBottomValid=%d distBottom=%.3f localZ=%.3f lateral=%.3f vz=%.3f landed=%d",
            static_cast<int>(disarmInput.distBottomValid),
            static_cast<double>(disarmInput.distBottom),
            static_cast<double>(disarmInput.localPositionZ),
            static_cast<double>(disarmInput.lateralError),
            static_cast<double>(disarmInput.verticalSpeedAbs),
            static_cast<int>(disarmInput.landed));
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            1000,
            "[PL] Loi Disarm/Debug: %s",
            e.what());
    }
    catch (...)
    {
        RCLCPP_ERROR_THROTTLE(
            _node.get_logger(),
            *(_node.get_clock()),
            1000,
            "[PL] Loi Disarm/Debug khong xac dinh");
    }

    if (_landDetected)
    {
        switchToState(State::Finished);
    }
}

void PrecisionLand::handleFinishedState()
{
    RCLCPP_WARN(_node.get_logger(), "[PL] Finished");

    std_msgs::msg::String msg;
    msg.data = "CENTER_LOOKUP_FOLLOW";
    _gimbalSeqPub->publish(msg);

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
    if (_paramUsePredictiveError && _targetWorld.validVelocity)
    {
        leadDtSec = std::max(poseAgeSec, velAgeSec);
    }

    leadDtSec += std::max(dt_s, 0.0f);
    leadDtSec += std::max(_paramControlExtraLeadSec, 0.0f);

    return std::clamp(leadDtSec, 0.0f, _paramPredictionDtMax);
}

precision_land::PredictionInput PrecisionLand::buildPredictionInput(float dt_s, const rclcpp::Time &ctrlStartNow)
{
    precision_land::PredictionInput input;

    input.leadDtSec = computeLeadTimeSec(dt_s, ctrlStartNow);
    input.predictiveAccGain = std::max(_paramPredictiveAccGain, 0.0f);

    input.vehicle.positionWorld = _vehicleLocalPosition->positionNed();
    input.vehicle.velocityWorld = _vehicleLocalPosition->velocityNed();
    input.vehicle.accelerationXY = estimateVehicleAccelerationXY(dt_s);

    input.target.positionWorld = Eigen::Vector3f(
        static_cast<float>(_targetWorld.position.x()),
        static_cast<float>(_targetWorld.position.y()),
        static_cast<float>(_targetWorld.position.z()));

    input.target.hasVelocity = _paramUsePredictiveError && _targetWorld.validVelocity;
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

    _debugTargetPredPub->publish(debugPredMsg);
}

void PrecisionLand::publishTimingDebug(
    const rclcpp::Time &ctrlStartNow,
    const rclcpp::Time &ctrlEndNow,
    const rclcpp::Time &cmdPubNow)
{
    publishPrecisionLandTiming(
        _debugDtPub,
        _imageTimestamp.nanoseconds() != 0 ? _imageTimestamp.seconds() : -1.0,
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

    return ((_node.now() - _targetWorld.timestamp).seconds() > _paramTargetTimeout);
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