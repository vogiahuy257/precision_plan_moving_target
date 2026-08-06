#include "PrecisionLand.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
const std::string kModeName = "PLHEOC";
constexpr bool kEnableDebugOutput = false;
constexpr double kLandRetryCooldownSec = 0.1;
constexpr float kVelocityDataTimeoutSec = 0.35f;
constexpr float kPi = 3.14159265358979323846f;
} // namespace

using namespace px4_ros2::literals;

PrecisionLand::PrecisionLand(rclcpp::Node &node)
    : ModeBase(node, kModeName),
      _node(node)
{
    _trajectorySetpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
    _vehicleAttitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);
    _vehicleLocalPosition = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);

    loadParameters();

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

    _gimbalSeqPub =
        _node.create_publisher<std_msgs::msg::String>(
            _gimbalCommandTopic,
            rclcpp::QoS(1).best_effort());

    _vehicleCommandPub =
        _node.create_publisher<px4_msgs::msg::VehicleCommand>(
            "/fmu/in/vehicle_command",
            rclcpp::QoS(10).best_effort());

    _vehicleCommandAckSub =
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

    _node.declare_parameter<float>("PID_deadband", 0.05f);
    _node.declare_parameter<float>("target_timeout", 3.0f);

    _node.declare_parameter<float>("descent_kp_pid", 0.9f);
    _node.declare_parameter<float>("descent_ki_pid", 0.01f);
    _node.declare_parameter<float>("descent_kd_pid", 0.0f);
    _node.declare_parameter<float>("descent_max_velocity", 10.0f);
    _node.declare_parameter<float>("slew_acc", 10.0f);

    _node.declare_parameter<bool>("yaw.enabled", true);
    _node.declare_parameter<float>("yaw.kp", 1.5f);
    _node.declare_parameter<float>("yaw.max_rate_rad_s", 0.8f);
    _node.declare_parameter<float>("yaw.slew_acc_rad_s2", 1.2f);
    _node.declare_parameter<float>("yaw.deadband_rad", 0.03f);

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
    _node.declare_parameter<bool>("debug_logger", false);

    _node.get_parameter("topics.target_pose", _targetPoseTopic);
    _node.get_parameter("topics.target_velocity", _targetVelocityTopic);
    _node.get_parameter("topics.vehicle_land_detected", _vehicleLandDetectedTopic);
    _node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);
    _node.get_parameter("topics.gimbal_command", _gimbalCommandTopic);

    _node.get_parameter("PID_deadband", _paramPidDeadband);
    _node.get_parameter("target_timeout", _paramTargetTimeout);

    _node.get_parameter("descent_kp_pid", _paramDescentKp);
    _node.get_parameter("descent_ki_pid", _paramDescentKi);
    _node.get_parameter("descent_kd_pid", _paramDescentKd);
    _node.get_parameter("descent_max_velocity", _paramDescentMaxVelocity);
    _node.get_parameter("slew_acc", _paramSlewAcc);

    _node.get_parameter("yaw.enabled", _paramYawControlEnabled);
    _node.get_parameter("yaw.kp", _paramYawKp);
    _node.get_parameter("yaw.max_rate_rad_s", _paramYawMaxRateRadS);
    _node.get_parameter("yaw.slew_acc_rad_s2", _paramYawSlewAccRadS2);
    _node.get_parameter("yaw.deadband_rad", _paramYawDeadbandRad);

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

    _paramPidDeadband = std::max(_paramPidDeadband, 0.0f);
    _paramTargetTimeout = std::max(_paramTargetTimeout, 0.01f);

    _paramDescentMaxVelocity = std::max(_paramDescentMaxVelocity, 0.0f);
    _paramSlewAcc = std::max(_paramSlewAcc, 0.0f);

    _paramYawKp = std::max(_paramYawKp, 0.0f);
    _paramYawMaxRateRadS = std::max(_paramYawMaxRateRadS, 0.0f);
    _paramYawSlewAccRadS2 = std::max(_paramYawSlewAccRadS2, 0.0f);
    _paramYawDeadbandRad = std::max(_paramYawDeadbandRad, 0.0f);

    _paramLandZoneZ = std::max(_paramLandZoneZ, 0.0f);
    _paramDescentVel = std::max(_paramDescentVel, 0.0f);
    _paramDescentGateRadius = std::max(_paramDescentGateRadius, 0.0f);
    _paramVmin = std::max(_paramVmin, 0.0f);
    _paramVmax = std::max(_paramVmax, _paramVmin);

    _paramPredictionDtMax = std::max(_paramPredictionDtMax, 0.0f);
    _paramControlExtraLeadSec = std::max(_paramControlExtraLeadSec, 0.0f);
    _paramPredictiveAccGain = std::max(_paramPredictiveAccGain, 0.0f);
    _paramPredictiveAccLpfAlpha = std::clamp(_paramPredictiveAccLpfAlpha, 0.0f, 1.0f);
    _paramPredictiveAccMax = std::max(_paramPredictiveAccMax, 0.0f);

    _paramDisarmHeight = std::max(_paramDisarmHeight, 0.0f);
    _paramDisarmLateralErrorThreshold = std::max(_paramDisarmLateralErrorThreshold, 0.0f);
    _paramDisarmVerticalSpeedThreshold = std::max(_paramDisarmVerticalSpeedThreshold, 0.0f);

    _disarmMode = parseDisarmMode(_paramDisarmMode);
    _disarmAltitudeSource = parseDisarmAltitudeSource(_paramDisarmAltitudeSource);
    _debugLogEnabled = _paramDebugLogger;
}

void PrecisionLand::targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_searchStarted || msg == nullptr)
    {
        return;
    }

    _targetWorld.position = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);
    _targetWorld.yawRad = yawFromPose(msg->pose);
    _targetWorld.validYaw = std::isfinite(_targetWorld.yawRad);

    rclcpp::Time msgTimestamp = msg->header.stamp;
    if (msgTimestamp.nanoseconds() == 0)
    {
        msgTimestamp = _node.now();
    }

    _targetWorld.timestamp = msgTimestamp;
    _targetWorld.validPose = true;
    _targetPoseRxNow = _node.now();
}

void PrecisionLand::targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

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
    if (msg == nullptr)
    {
        return;
    }

    _landDetected = msg->landed;
}

void PrecisionLand::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

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

void PrecisionLand::vehicleCommandAckCallback(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

    if (msg->command != px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND)
    {
        return;
    }

    if (msg->result == px4_msgs::msg::VehicleCommandAck::VEHICLE_CMD_RESULT_ACCEPTED)
    {
        _waitingLandAck = false;
        _disarmStatus = DisarmDecisionStatus::Accepted;
        switchToState(State::Finished);
        return;
    }

    _waitingLandAck = false;
    _disarmStatus = DisarmDecisionStatus::Rejected;
    _landRequestTime = _node.now();
}

void PrecisionLand::onActivate()
{
    _prevVehicleVelX = 0.0f;
    _prevVehicleVelY = 0.0f;
    _vehicleAccXFilt = 0.0f;
    _vehicleAccYFilt = 0.0f;
    _prevVehicleVelValid = false;

    resetXyController();
    resetYawController();
    resetDisarmLogic();

    _searchStarted = true;
    _targetLostPrev = true;
    _distBottomValid = false;
    _landDetected = false;
    _approachAltitude = 0.0f;
    _targetPoseRxNow = rclcpp::Time(0, 0, _node.get_clock()->get_clock_type());
    _targetVelRxNow = rclcpp::Time(0, 0, _node.get_clock()->get_clock_type());

    startDebugLogSession();

    switchToState(State::Search);
}

void PrecisionLand::onDeactivate()
{
    _searchStarted = false;
    resetXyController();
    resetYawController();
    resetDisarmLogic();
    closeDebugLogSession();
}

void PrecisionLand::hover()
{
    _yawRateSpRadS = 0.0f;

    _trajectorySetpoint->update(
        Eigen::Vector3f(0.0f, 0.0f, 0.0f),
        std::nullopt,
        std::nullopt,
        0.0f);
}

void PrecisionLand::updateSetpoint(float dt_s)
{
    const bool targetLost = checkTargetTimeout();
    _targetLostPrev = targetLost;

    switch (_state)
    {
    case State::Search:
        handleSearchState(targetLost);
        break;

    case State::Descend:
        handleDescendState(dt_s, targetLost);
        break;

    case State::Finished:
        handleFinishedState();
        break;
    }
}

void PrecisionLand::handleSearchState(bool targetLost)
{
    if (!targetLost && _targetWorld.validPose)
    {
        switchToState(State::Descend);
        return;
    }

    hover();
}

void PrecisionLand::handleDescendState(float dt_s, bool targetLost)
{
    const float altitudeNow =
        _distBottomValid ? std::abs(_zDistBottom) : std::abs(_vehicleLocalPosition->positionNed().z());

    const bool allowBlindFinalDescent =
        targetLost && _targetWorld.validPose && altitudeNow <= _paramLandZoneZ;

    if (targetLost && !allowBlindFinalDescent)
    {
        switchToState(State::Search);
        hover();
        return;
    }

    const rclcpp::Time ctrlStartNow = _node.now();

    PredictionOutput predictionOutput{};
    XYControllerInput xyInput{};
    XYControllerOutput xyOutput{};
    YawControllerOutput yawOutput{};
    float lateralError = 0.0f;
    float vz = 0.0f;

    try
    {
        const PredictionInput predictionInput = buildPredictionInput(dt_s, ctrlStartNow);
        predictionOutput = predictTarget(predictionInput);

        _approachAltitude = std::abs(predictionInput.vehicle.positionWorld.z());

        xyInput.futureErrorXY = predictionOutput.futureErrorXY;
        xyInput.dtSec = dt_s;
        xyInput.targetValid = !targetLost || allowBlindFinalDescent;

        const bool velocityTimestampValid = _targetWorld.velocityTimestamp.nanoseconds() != 0;
        const float velocityAgeSec =
            (_targetWorld.validVelocity && velocityTimestampValid)
                ? static_cast<float>((ctrlStartNow - _targetWorld.velocityTimestamp).seconds())
                : kVelocityDataTimeoutSec + 1.0f;

        const bool velocityFresh =
            _targetWorld.validVelocity &&
            velocityTimestampValid &&
            velocityAgeSec >= 0.0f &&
            velocityAgeSec <= kVelocityDataTimeoutSec;

        const bool canUseTrackedVelocity =
            velocityFresh && (!targetLost || allowBlindFinalDescent);

        xyInput.targetVelocityXY =
            canUseTrackedVelocity ? predictionInput.target.velocityWorld.head<2>().eval()
                                  : Eigen::Vector2f::Zero();
        xyInput.useTargetFeedforward = canUseTrackedVelocity;

        xyOutput = updateXyController(xyInput);

        yawOutput = updateYawController(
            dt_s,
            _targetWorld.yawRad,
            xyInput.targetValid && _targetWorld.validYaw);

        lateralError = predictionOutput.futureErrorXY.norm();

        const float altitudeForZ = _distBottomValid ? std::abs(_zDistBottom) : _approachAltitude;
        vz = allowBlindFinalDescent
                 ? std::abs(_paramDescentVel)
                 : computeZVelocityCommand(altitudeForZ, predictionOutput.futureErrorXY);
    }
    catch (...)
    {
        hover();
        return;
    }

    const rclcpp::Time ctrlEndNow = _node.now();

    std::optional<float> yawRateSp = std::nullopt;
    if (_paramYawControlEnabled)
    {
        yawRateSp = yawOutput.yawRateSpRadS;
    }

    _trajectorySetpoint->update(
        Eigen::Vector3f(xyOutput.velocitySpXY.x(), xyOutput.velocitySpXY.y(), vz),
        std::nullopt,
        std::nullopt,
        yawRateSp);

    const rclcpp::Time cmdPubNow = _node.now();

    DisarmInput disarmInput{};
    disarmInput.distBottomValid = _distBottomValid;
    disarmInput.distBottom = _zDistBottom;
    disarmInput.localPositionZValid = true;
    disarmInput.localPositionZ = _vehicleLocalPosition->positionNed().z();
    disarmInput.lateralError = lateralError;
    disarmInput.verticalSpeedAbs = std::abs(_vehicleLocalPosition->velocityNed().z());
    disarmInput.landed = _landDetected;

    const DisarmOutput disarmOutput = updateDisarmLogic(disarmInput);

    logDebugSample(
        ctrlStartNow,
        ctrlEndNow,
        cmdPubNow,
        predictionOutput,
        xyInput,
        xyOutput,
        yawOutput,
        disarmOutput,
        vz,
        altitudeNow);

    if (_landDetected)
    {
        switchToState(State::Finished);
    }
}

void PrecisionLand::handleFinishedState()
{
    closeDebugLogSession();

    if (_gimbalSeqPub)
    {
        std_msgs::msg::String msg;
        msg.data = "CENTER_LOOKUP_FOLLOW";
        _gimbalSeqPub->publish(msg);
    }

    ModeBase::completed(px4_ros2::Result::Success);
}

void PrecisionLand::resetXyController()
{
    _velXIntegral = 0.0f;
    _velYIntegral = 0.0f;
    _prevErrX = 0.0f;
    _prevErrY = 0.0f;
    _prevErrValid = false;
    _vxFilt = 0.0f;
    _vyFilt = 0.0f;
}

void PrecisionLand::resetYawController()
{
    _yawRateSpRadS = 0.0f;
}

void PrecisionLand::resetDisarmLogic()
{
    _disarmSent = false;
    _waitingLandAck = false;
    _disarmStatus = DisarmDecisionStatus::Idle;
    _landRequestTime = rclcpp::Time(0, 0, _node.get_clock()->get_clock_type());
}

bool PrecisionLand::checkTargetTimeout() const
{
    if (!_targetWorld.validPose)
    {
        return true;
    }

    return (_node.now() - _targetWorld.timestamp).seconds() > _paramTargetTimeout;
}

void PrecisionLand::switchToState(State state)
{
    _state = state;
}

float PrecisionLand::computeLeadTimeSec(float dt_s, const rclcpp::Time &ctrlStartNow) const
{
    float poseAgeSec = static_cast<float>((ctrlStartNow - _targetWorld.timestamp).seconds());
    poseAgeSec = std::max(poseAgeSec, 0.0f);

    float velAgeSec = poseAgeSec;
    if (_targetWorld.validVelocity)
    {
        velAgeSec = static_cast<float>((ctrlStartNow - _targetWorld.velocityTimestamp).seconds());
        velAgeSec = std::max(velAgeSec, 0.0f);
    }

    float leadDtSec = poseAgeSec;
    if (_paramUsePredictiveError && _targetWorld.validVelocity)
    {
        leadDtSec = std::max(poseAgeSec, velAgeSec);
    }

    leadDtSec += std::max(dt_s, 0.0f);
    leadDtSec += _paramControlExtraLeadSec;

    return std::clamp(leadDtSec, 0.0f, _paramPredictionDtMax);
}

PrecisionLand::PredictionInput PrecisionLand::buildPredictionInput(
    float dt_s,
    const rclcpp::Time &ctrlStartNow)
{
    PredictionInput input{};

    input.leadDtSec = computeLeadTimeSec(dt_s, ctrlStartNow);
    input.predictiveAccGain = _paramPredictiveAccGain;

    input.vehicle.positionWorld = _vehicleLocalPosition->positionNed();
    input.vehicle.velocityWorld = _vehicleLocalPosition->velocityNed();
    input.vehicle.accelerationXY = estimateVehicleAccelerationXY(dt_s);

    input.target.positionWorld = Eigen::Vector3f(
        static_cast<float>(_targetWorld.position.x()),
        static_cast<float>(_targetWorld.position.y()),
        static_cast<float>(_targetWorld.position.z()));

    const bool velocityTimestampValid = _targetWorld.velocityTimestamp.nanoseconds() != 0;
    const float velocityAgeSec =
        (_targetWorld.validVelocity && velocityTimestampValid)
            ? static_cast<float>((ctrlStartNow - _targetWorld.velocityTimestamp).seconds())
            : kVelocityDataTimeoutSec + 1.0f;

    const bool velocityFresh =
        _targetWorld.validVelocity &&
        velocityTimestampValid &&
        velocityAgeSec >= 0.0f &&
        velocityAgeSec <= kVelocityDataTimeoutSec;

    input.target.hasVelocity = _paramUsePredictiveError && velocityFresh;
    if (input.target.hasVelocity)
    {
        input.target.velocityWorld = Eigen::Vector3f(
            static_cast<float>(_targetWorld.velocity.x()),
            static_cast<float>(_targetWorld.velocity.y()),
            static_cast<float>(_targetWorld.velocity.z()));
    }

    return input;
}

PrecisionLand::PredictionOutput PrecisionLand::predictTarget(const PredictionInput &input) const
{
    PredictionOutput output{};

    output.targetFutureWorld = input.target.positionWorld;
    if (input.leadDtSec > 0.0f && input.target.hasVelocity)
    {
        output.targetFutureWorld += input.target.velocityWorld * input.leadDtSec;
    }

    output.vehicleFutureWorld = input.vehicle.positionWorld;
    if (input.leadDtSec > 0.0f)
    {
        output.vehicleFutureWorld.x() +=
            input.vehicle.velocityWorld.x() * input.leadDtSec +
            0.5f * input.predictiveAccGain * input.vehicle.accelerationXY.x() * input.leadDtSec * input.leadDtSec;

        output.vehicleFutureWorld.y() +=
            input.vehicle.velocityWorld.y() * input.leadDtSec +
            0.5f * input.predictiveAccGain * input.vehicle.accelerationXY.y() * input.leadDtSec * input.leadDtSec;

        output.vehicleFutureWorld.z() += input.vehicle.velocityWorld.z() * input.leadDtSec;
    }

    output.futureErrorXY.x() = output.targetFutureWorld.x() - output.vehicleFutureWorld.x();
    output.futureErrorXY.y() = output.targetFutureWorld.y() - output.vehicleFutureWorld.y();

    if (!std::isfinite(output.futureErrorXY.x()) || !std::isfinite(output.futureErrorXY.y()))
    {
        throw std::runtime_error("prediction output is not finite");
    }

    return output;
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

    const float accMax = _paramPredictiveAccMax;
    accXRaw = std::clamp(accXRaw, -accMax, accMax);
    accYRaw = std::clamp(accYRaw, -accMax, accMax);

    _vehicleAccXFilt = _paramPredictiveAccLpfAlpha * accXRaw +
                        (1.0f - _paramPredictiveAccLpfAlpha) * _vehicleAccXFilt;
    _vehicleAccYFilt = _paramPredictiveAccLpfAlpha * accYRaw +
                        (1.0f - _paramPredictiveAccLpfAlpha) * _vehicleAccYFilt;

    _prevVehicleVelX = currentVelX;
    _prevVehicleVelY = currentVelY;

    return Eigen::Vector2f(_vehicleAccXFilt, _vehicleAccYFilt);
}

Eigen::Vector2f PrecisionLand::clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const
{
    if (maxNorm <= 0.0f)
    {
        return Eigen::Vector2f::Zero();
    }

    const float norm = value.norm();
    if (norm <= maxNorm || norm < 1e-6f)
    {
        return value;
    }

    return value * (maxNorm / norm);
}

PrecisionLand::XYControllerOutput PrecisionLand::updateXyController(const XYControllerInput &input)
{
    XYControllerOutput output{};

    const float dt = std::max(input.dtSec, 1e-3f);
    const float errX = input.futureErrorXY.x();
    const float errY = input.futureErrorXY.y();

    const float xp = _paramDescentKp * errX;
    const float yp = _paramDescentKp * errY;

    if (input.targetValid && std::abs(errX) > _paramPidDeadband)
    {
        _velXIntegral += errX * dt;
    }
    else
    {
        _velXIntegral *= 0.9f;
    }

    if (input.targetValid && std::abs(errY) > _paramPidDeadband)
    {
        _velYIntegral += errY * dt;
    }
    else
    {
        _velYIntegral *= 0.9f;
    }

    float xi = 0.0f;
    float yi = 0.0f;
    if (_paramDescentKi > 1e-6f)
    {
        const float maxIntegral = 0.15f * _paramDescentMaxVelocity / _paramDescentKi;
        _velXIntegral = std::clamp(_velXIntegral, -maxIntegral, maxIntegral);
        _velYIntegral = std::clamp(_velYIntegral, -maxIntegral, maxIntegral);
        xi = _paramDescentKi * _velXIntegral;
        yi = _paramDescentKi * _velYIntegral;
    }

    float xd = 0.0f;
    float yd = 0.0f;
    if (input.targetValid && _paramDescentKd > 1e-6f && _prevErrValid)
    {
        xd = _paramDescentKd * (errX - _prevErrX) / dt;
        yd = _paramDescentKd * (errY - _prevErrY) / dt;
    }

    _prevErrX = errX;
    _prevErrY = errY;
    _prevErrValid = input.targetValid;

    output.feedbackXY.x() = xp + xi + xd;
    output.feedbackXY.y() = yp + yi + yd;
    output.feedbackXY = clampVectorNorm(output.feedbackXY, _paramDescentMaxVelocity);

    output.commandRawXY = output.feedbackXY;
    if (input.useTargetFeedforward)
    {
        output.commandRawXY += input.targetVelocityXY;
    }

    output.commandRawXY = clampVectorNorm(output.commandRawXY, _paramDescentMaxVelocity);


    _vxFilt = applySlew(output.commandRawXY.x(), _vxFilt, _paramSlewAcc, dt);
    _vyFilt = applySlew(output.commandRawXY.y(), _vyFilt, _paramSlewAcc, dt);

    output.velocitySpXY.x() = _vxFilt;
    output.velocitySpXY.y() = _vyFilt;
    output.velocitySpXY = clampVectorNorm(output.velocitySpXY, _paramDescentMaxVelocity);


    if (!std::isfinite(output.velocitySpXY.x()) || !std::isfinite(output.velocitySpXY.y()))
    {
        throw std::runtime_error("xy controller output is not finite");
    }

    return output;
}

float PrecisionLand::applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const
{
    const float dt = std::max(dtSec, 1e-3f);
    const float maxDeltaVelocity = std::max(accelLimit, 0.0f) * dt;
    const float deltaVelocity = std::clamp(commandVelocity - previousVelocity, -maxDeltaVelocity, maxDeltaVelocity);

    return previousVelocity + deltaVelocity;
}

PrecisionLand::YawControllerOutput PrecisionLand::updateYawController(
    float dtSec,
    float targetYawRad,
    bool targetYawValid)
{
    YawControllerOutput output{};
    output.yawRateSpRadS = _yawRateSpRadS;

    if (!_paramYawControlEnabled || !targetYawValid)
    {
        _yawRateSpRadS = applyYawSlew(0.0f, _yawRateSpRadS, _paramYawSlewAccRadS2, dtSec);
        output.yawRateSpRadS = _yawRateSpRadS;
        return output;
    }

    const float currentYawRad = px4_ros2::quaternionToYaw(_vehicleAttitude->attitude());
    if (!std::isfinite(currentYawRad) || !std::isfinite(targetYawRad))
    {
        _yawRateSpRadS = applyYawSlew(0.0f, _yawRateSpRadS, _paramYawSlewAccRadS2, dtSec);
        output.yawRateSpRadS = _yawRateSpRadS;
        return output;
    }

    float yawErrorRad = normalizeAnglePi(targetYawRad - currentYawRad);
    if (std::abs(yawErrorRad) < _paramYawDeadbandRad)
    {
        yawErrorRad = 0.0f;
    }

    const float yawRateRawRadS = std::clamp(
        _paramYawKp * yawErrorRad,
        -_paramYawMaxRateRadS,
        _paramYawMaxRateRadS);

    _yawRateSpRadS = applyYawSlew(yawRateRawRadS, _yawRateSpRadS, _paramYawSlewAccRadS2, dtSec);

    output.valid = true;
    output.currentYawRad = currentYawRad;
    output.targetYawRad = targetYawRad;
    output.errorYawRad = yawErrorRad;
    output.yawRateRawRadS = yawRateRawRadS;
    output.yawRateSpRadS = _yawRateSpRadS;
    output.yawTurnDirection = (yawErrorRad > 0.0f) ? 1 : ((yawErrorRad < 0.0f) ? -1 : 0);

    return output;
}
float PrecisionLand::applyYawSlew(
    float commandYawRate,
    float previousYawRate,
    float slewLimit,
    float dtSec) const
{
    const float dt = std::max(dtSec, 1e-3f);
    const float maxDeltaYawRate = std::max(slewLimit, 0.0f) * dt;
    const float deltaYawRate = std::clamp(
        commandYawRate - previousYawRate,
        -maxDeltaYawRate,
        maxDeltaYawRate);

    return previousYawRate + deltaYawRate;
}

float PrecisionLand::normalizeAnglePi(float angleRad) const
{
    return std::atan2(std::sin(angleRad), std::cos(angleRad));
}

float PrecisionLand::yawFromPose(const geometry_msgs::msg::Pose &pose) const
{
    const auto &q = pose.orientation;
    const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);

    if (norm <= 1e-9)
    {
        return NAN;
    }

    const double w = q.w / norm;
    const double x = q.x / norm;
    const double y = q.y / norm;
    const double z = q.z / norm;

    return static_cast<float>(std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)));
}

float PrecisionLand::computeZVelocityCommand(float vehicleAltitudeAbs, const Eigen::Vector2f &futureErrorXY) const
{
    if (!std::isfinite(vehicleAltitudeAbs) || vehicleAltitudeAbs < 0.0f)
    {
        return 0.0f;
    }

    const float lateralError = futureErrorXY.norm();

    if (vehicleAltitudeAbs < _paramLandZoneZ)
    {
        return std::abs(_paramDescentVel);
    }

    if (lateralError >= _paramDescentGateRadius)
    {
        return 0.0f;
    }

    const float denominator = std::max(_paramDescentGateRadius, 1e-6f);
    const float scale = 1.0f - lateralError / denominator;
    const float scaleClamped = std::clamp(scale, 0.0f, 1.0f);

    return _paramVmin + (_paramVmax - _paramVmin) * scaleClamped;
}

PrecisionLand::DisarmMode PrecisionLand::parseDisarmMode(const std::string &value) const
{
    if (value == "disabled")
    {
        return DisarmMode::Disabled;
    }

    return DisarmMode::Enabled;
}

PrecisionLand::DisarmAltitudeSource PrecisionLand::parseDisarmAltitudeSource(const std::string &value) const
{
    if (value == "local_position_z")
    {
        return DisarmAltitudeSource::LocalPositionZ;
    }

    return DisarmAltitudeSource::DistBottom;
}

float PrecisionLand::selectDisarmAltitude(const DisarmInput &input, bool &isValid) const
{
    switch (_disarmAltitudeSource)
    {
    case DisarmAltitudeSource::DistBottom:
        isValid = input.distBottomValid;
        return input.distBottom;

    case DisarmAltitudeSource::LocalPositionZ:
        isValid = input.localPositionZValid;
        return std::abs(input.localPositionZ);
    }

    isValid = false;
    return 0.0f;
}

bool PrecisionLand::shouldRequestLand(
    const DisarmInput &input,
    float &selectedAltitude,
    bool &selectedAltitudeValid) const
{
    if (_disarmMode == DisarmMode::Disabled)
    {
        selectedAltitude = 0.0f;
        selectedAltitudeValid = false;
        return false;
    }

    if (input.landed && _paramDisarmAllowLandedImmediate)
    {
        selectedAltitude = 0.0f;
        selectedAltitudeValid = true;
        return true;
    }

    selectedAltitude = selectDisarmAltitude(input, selectedAltitudeValid);
    if (!selectedAltitudeValid)
    {
        return false;
    }

    const bool heightOk = selectedAltitude <= _paramDisarmHeight;
    return heightOk;
}

PrecisionLand::DisarmOutput PrecisionLand::updateDisarmLogic(const DisarmInput &input)
{
    DisarmOutput output{};
    output.status = _disarmStatus;

    float selectedAltitude = 0.0f;
    bool selectedAltitudeValid = false;
    const bool allowLandNow = shouldRequestLand(input, selectedAltitude, selectedAltitudeValid);

    output.selectedAltitude = selectedAltitude;
    output.selectedAltitudeValid = selectedAltitudeValid;

    if (_disarmMode == DisarmMode::Disabled)
    {
        _disarmStatus = DisarmDecisionStatus::Disabled;
        output.status = _disarmStatus;
        return output;
    }

    if (_disarmStatus == DisarmDecisionStatus::Accepted)
    {
        output.status = _disarmStatus;
        return output;
    }

    if (!allowLandNow)
    {
        _waitingLandAck = false;
        _disarmStatus = DisarmDecisionStatus::Blocked;
        output.status = _disarmStatus;
        return output;
    }

    const double dtFromLastRequest = (_node.now() - _landRequestTime).seconds();
    if (!_disarmSent || dtFromLastRequest >= kLandRetryCooldownSec)
    {
        output.shouldSendLand = sendLandCommand();
        output.status = _disarmStatus;
        return output;
    }

    _disarmStatus = _waitingLandAck ? DisarmDecisionStatus::WaitingAck : _disarmStatus;
    output.status = _disarmStatus;
    return output;
}

bool PrecisionLand::sendLandCommand()
{
    if (!_vehicleCommandPub)
    {
        _disarmStatus = DisarmDecisionStatus::Rejected;
        return false;
    }

    publishVehicleCommand(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_NAV_LAND, 0.0f, 0.0f);

    _disarmSent = true;
    _waitingLandAck = true;
    _landRequestTime = _node.now();
    _disarmStatus = DisarmDecisionStatus::WaitingAck;

    return true;
}

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
    msg.confirmation = 0;
    msg.from_external = true;

    _vehicleCommandPub->publish(msg);
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
    }

    return "Unknown";
}

void PrecisionLand::startDebugLogSession()
{
    if (!_debugLogEnabled)
    {
        return;
    }

    try
    {
        closeDebugLogSession();
        _debugLogSessionStamp = makeCurrentTimeString();
        _debugLogPath = buildDebugCsvPath();
        _debugLogSessionStarted = true;
        openDebugLogFileIfNeeded();
        writeDebugLogHeaderIfNeeded();
    }
    catch (...)
    {
        disableDebugLog();
    }
}

void PrecisionLand::closeDebugLogSession()
{
    try
    {
        flushDebugLog();
    }
    catch (...)
    {
    }

    _debugLogBuffer.clear();

    if (_debugLogFile.is_open())
    {
        _debugLogFile.close();
    }

    _debugLogFileOpened = false;
    _debugLogHeaderWritten = false;
    _debugLogSessionStarted = false;
    _debugLogSessionStamp.clear();
    _debugLogPath.clear();
}

void PrecisionLand::flushDebugLog()
{
    if (!_debugLogEnabled || !_debugLogFileOpened || _debugLogBuffer.empty())
    {
        return;
    }

    for (const std::string &line : _debugLogBuffer)
    {
        _debugLogFile << line << '\n';
    }

    _debugLogFile.flush();
    _debugLogBuffer.clear();
}

void PrecisionLand::logDebugSample(
    const rclcpp::Time &ctrlStartNow,
    const rclcpp::Time &ctrlEndNow,
    const rclcpp::Time &cmdPubNow,
    const PredictionOutput &predictionOutput,
    const XYControllerInput &xyInput,
    const XYControllerOutput &xyOutput,
    const YawControllerOutput &yawOutput,
    const DisarmOutput &disarmOutput,
    float vz,
    float altitudeNow)
{
    if (!_debugLogEnabled || !_debugLogSessionStarted)
    {
        return;
    }

    try
    {
        DebugSample sample{};
        sample.timeSec = _node.now().seconds();
        sample.state = stateName(_state);

        sample.dronePos = _vehicleLocalPosition->positionNed();
        sample.droneVel = _vehicleLocalPosition->velocityNed();

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

        sample.currentYawRad = yawOutput.currentYawRad;
        sample.targetYawRad = yawOutput.targetYawRad;
        sample.yawErrorRad = yawOutput.errorYawRad;
        sample.yawRateRawRadS = yawOutput.yawRateRawRadS;
        sample.yawRateSpRadS = yawOutput.yawRateSpRadS;
        sample.yawTurnDirection = yawOutput.yawTurnDirection;
        sample.yawControlValid = yawOutput.valid;

        sample.altitudeAbs = altitudeNow;
        sample.distBottom = _distBottomValid ? _zDistBottom : -1.0f;
        sample.shouldLand =
            (disarmOutput.status == DisarmDecisionStatus::WaitingAck) ||
            (disarmOutput.status == DisarmDecisionStatus::Accepted);
        sample.landDetected = _landDetected;

        fillDebugTimingSample(sample, ctrlStartNow, ctrlEndNow, cmdPubNow);

        openDebugLogFileIfNeeded();
        writeDebugLogHeaderIfNeeded();
        _debugLogBuffer.push_back(debugSampleToCsvLine(sample));

        if (_debugLogBuffer.size() >= kDebugLogFlushBatchSize)
        {
            flushDebugLog();
        }
    }
    catch (...)
    {
        disableDebugLog();
    }
}

void PrecisionLand::fillDebugTimingSample(
    DebugSample &sample,
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
        (_targetWorld.timestamp.nanoseconds() != 0) ? (cmdPubNow - _targetWorld.timestamp).seconds() : -1.0;
}

void PrecisionLand::openDebugLogFileIfNeeded()
{
    if (_debugLogFileOpened)
    {
        return;
    }

    if (_debugLogPath.empty())
    {
        return;
    }

    _debugLogFile.open(_debugLogPath, std::ios::out | std::ios::trunc);
    if (!_debugLogFile.is_open())
    {
        disableDebugLog();
        return;
    }

    _debugLogFileOpened = true;
}

void PrecisionLand::writeDebugLogHeaderIfNeeded()
{
    if (!_debugLogFileOpened || _debugLogHeaderWritten)
    {
        return;
    }

    _debugLogFile
        << "time,state,"
        << "drone_pos_x,drone_pos_y,drone_pos_z,"
        << "drone_vel_x,drone_vel_y,drone_vel_z,"
        << "target_est_x,target_est_y,target_est_z,"
        << "target_pred_x,target_pred_y,target_pred_z,"
        << "target_vel_x,target_vel_y,target_vel_z,"
        << "error_x,error_y,"
        << "future_error_x,future_error_y,"
        << "error_xy_norm,future_error_xy_norm,"
        << "pid_out_x,pid_out_y,"
        << "ff_x,ff_y,"
        << "final_sp_x,final_sp_y,final_sp_z,"
        << "current_yaw_rad,target_yaw_rad,yaw_error_rad,"
        << "yaw_rate_raw_rad_s,yaw_rate_sp_rad_s,yaw_turn_direction,yaw_control_valid,"
        << "altitude_abs,dist_bottom,"
        << "should_land,land_detected,"
        << "pose_wait_dt,vel_wait_dt,control_processing_dt,send_cmd_dt,total_image_to_cmd_dt"
        << '\n';

    _debugLogFile.flush();
    _debugLogHeaderWritten = true;
}

void PrecisionLand::disableDebugLog()
{
    _debugLogEnabled = false;
    _paramDebugLogger = false;
    closeDebugLogSession();
}

std::string PrecisionLand::makeCurrentTimeString() const
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTimeT = std::chrono::system_clock::to_time_t(now);

    std::tm localTm{};
#if defined(_WIN32)
    localtime_s(&localTm, &nowTimeT);
#else
    localtime_r(&nowTimeT, &localTm);
#endif

    std::ostringstream ss;
    ss << std::put_time(&localTm, "%H%M_%d_%m_%y");
    return ss.str();
}

std::string PrecisionLand::buildDebugCsvPath() const
{
    namespace fs = std::filesystem;

    const fs::path logDir(kDebugLogDirectory);
    fs::create_directories(logDir);

    return (logDir / (_debugLogSessionStamp + "_controller.csv")).string();
}

std::string PrecisionLand::debugSampleToCsvLine(const DebugSample &sample) const
{
    const float errorNorm = sample.errorXY.norm();
    const float futureErrorNorm = sample.futureErrorXY.norm();

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);

    ss
        << sample.timeSec << ','
        << sample.state << ','
        << sample.dronePos.x() << ',' << sample.dronePos.y() << ',' << sample.dronePos.z() << ','
        << sample.droneVel.x() << ',' << sample.droneVel.y() << ',' << sample.droneVel.z() << ','
        << sample.targetEst.x() << ',' << sample.targetEst.y() << ',' << sample.targetEst.z() << ','
        << sample.targetPred.x() << ',' << sample.targetPred.y() << ',' << sample.targetPred.z() << ','
        << sample.targetVel.x() << ',' << sample.targetVel.y() << ',' << sample.targetVel.z() << ','
        << sample.errorXY.x() << ',' << sample.errorXY.y() << ','
        << sample.futureErrorXY.x() << ',' << sample.futureErrorXY.y() << ','
        << errorNorm << ',' << futureErrorNorm << ','
        << sample.pidOutXY.x() << ',' << sample.pidOutXY.y() << ','
        << sample.ffXY.x() << ',' << sample.ffXY.y() << ','
        << sample.finalSp.x() << ',' << sample.finalSp.y() << ',' << sample.finalSp.z() << ','
        << sample.currentYawRad << ',' << sample.targetYawRad << ',' << sample.yawErrorRad << ','
        << sample.yawRateRawRadS << ',' << sample.yawRateSpRadS << ','
        << sample.yawTurnDirection << ','
        << static_cast<int>(sample.yawControlValid) << ','
        << sample.altitudeAbs << ',' << sample.distBottom << ','
        << static_cast<int>(sample.shouldLand) << ','
        << static_cast<int>(sample.landDetected) << ','
        << sample.timing.poseWaitDt << ','
        << sample.timing.velWaitDt << ','
        << sample.timing.controlProcessingDt << ','
        << sample.timing.sendCmdDt << ','
        << sample.timing.totalImageToCmdDt;

    return ss.str();
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<PrecisionLand>>(kModeName, kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}
