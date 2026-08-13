#include "TargetDrop.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
const std::string kModeName = "TGT_DROP";
constexpr bool kEnableDebugOutput = false;
constexpr float kVelocityDataTimeoutSec = 0.35f;
} // namespace

TargetDrop::TargetDrop(rclcpp::Node &node)
    : ModeBase(node, kModeName),
      _node(node)
{
    _trajectorySetpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
    _vehicleAttitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);
    _vehicleLocalPosition = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);

    loadParameters();

    const auto qos = rclcpp::QoS(1).best_effort();

    _targetPoseSub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetPoseTopic,
            qos,
            std::bind(&TargetDrop::targetPoseCallback, this, std::placeholders::_1));

    _targetVelocitySub =
        _node.create_subscription<geometry_msgs::msg::PoseStamped>(
            _targetVelocityTopic,
            qos,
            std::bind(&TargetDrop::targetVelocityCallback, this, std::placeholders::_1));

    if (_targetModel == TargetModel::Ctra)
    {
        _targetMotionSub =
            _node.create_subscription<std_msgs::msg::Float64MultiArray>(
                _targetMotionTopic,
                qos,
                std::bind(&TargetDrop::targetMotionCallback, this, std::placeholders::_1));
    }

    _targetCovarianceSub =
        _node.create_subscription<std_msgs::msg::Float64MultiArray>(
            _targetCovarianceTopic,
            qos,
            std::bind(&TargetDrop::targetCovarianceCallback, this, std::placeholders::_1));

    const auto configQos = rclcpp::QoS(1).reliable().transient_local();
    _targetProcessNoiseSub =
        _node.create_subscription<std_msgs::msg::Float64MultiArray>(
            _targetProcessNoiseTopic,
            configQos,
            std::bind(&TargetDrop::targetProcessNoiseCallback, this, std::placeholders::_1));

    _vehicleLocalPositionSub =
        _node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
            _vehicleLocalPositionTopic,
            qos,
            std::bind(&TargetDrop::vehicleLocalPositionCallback, this, std::placeholders::_1));

    modeRequirements().manual_control = false;
}

void TargetDrop::loadParameters()
{
    _node.declare_parameter<std::string>("estimator.model", "kf");
    _node.declare_parameter<float>("estimator.motion_timeout_s", 0.20f);
    _node.declare_parameter<float>("estimator.prediction_step_s", 0.02f);

    _node.declare_parameter<std::string>("topics.target_pose", "/KF/target_pose_est_NED");
    _node.declare_parameter<std::string>("topics.target_velocity", "/KF/target_velocity_est_NED");
    _node.declare_parameter<std::string>("topics.target_motion", "/EKF/target_motion");
    _node.declare_parameter<std::string>("topics.target_covariance", "/KF/target_covariance_NE");
    _node.declare_parameter<std::string>("topics.target_process_noise", "/KF/process_noise");
    _node.declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position_v1");

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

    _node.declare_parameter<float>("tracking_height", 3.0f);
    _node.declare_parameter<float>("height_tolerance", 0.15f);
    _node.declare_parameter<float>("height_kp", 0.6f);
    _node.declare_parameter<float>("vertical_slew_acc", 0.6f);
    _node.declare_parameter<float>("descent_gate_radius", 0.3f);
    _node.declare_parameter<float>("vmin", 0.3f);
    _node.declare_parameter<float>("vmax", 0.45f);

    _node.declare_parameter<bool>("use_predictive_error", true);
    _node.declare_parameter<float>("prediction_dt_max", 0.75f);
    _node.declare_parameter<float>("control_extra_lead_sec", 0.25f);
    _node.declare_parameter<float>("predictive_acc_gain", 0.0f);
    _node.declare_parameter<float>("predictive_acc_lpf_alpha", 0.4f);
    _node.declare_parameter<float>("predictive_acc_max", 4.0f);

    _node.declare_parameter<float>("v_wind_n", 0.0f);
    _node.declare_parameter<float>("v_wind_e", 0.0f);
    _node.declare_parameter<float>("v_wind_d", 0.0f);
    _node.declare_parameter<float>("m_payload", 0.5f);
    _node.declare_parameter<float>("c_d", 1.0f);
    _node.declare_parameter<float>("area_m2", 0.01f);
    _node.declare_parameter<float>("rho_air", 1.225f);
    _node.declare_parameter<float>("drop_dt", 0.005f);
    _node.declare_parameter<float>("drop_t_max", 3.0f);

    _node.declare_parameter<float>("release.delay_sec", 0.20f);
    _node.declare_parameter<float>("release.max_error_m", 0.30f);
    _node.declare_parameter<float>("release.max_sigma_m", 0.20f);
    _node.declare_parameter<float>("release.max_relative_velocity_m_s", 0.50f);
    _node.declare_parameter<float>("release.max_target_age_s", 0.20f);
    _node.declare_parameter<int>("release.confirm_cycles", 5);
    _node.declare_parameter<float>("release.cov_timeout_s", 0.20f);
    _node.declare_parameter<float>("release.payload_sigma_n_m", 0.0f);
    _node.declare_parameter<float>("release.payload_sigma_e_m", 0.0f);

    _node.get_parameter("estimator.model", _paramTargetModel);
    _node.get_parameter("estimator.motion_timeout_s", _paramEstimatorMotionTimeoutSec);
    _node.get_parameter("estimator.prediction_step_s", _paramEstimatorPredictionStepSec);

    _node.get_parameter("topics.target_pose", _targetPoseTopic);
    _node.get_parameter("topics.target_velocity", _targetVelocityTopic);
    _node.get_parameter("topics.target_motion", _targetMotionTopic);
    _node.get_parameter("topics.target_covariance", _targetCovarianceTopic);
    _node.get_parameter("topics.target_process_noise", _targetProcessNoiseTopic);
    _node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);

    _node.get_parameter("PID_deadband", _paramPidDeadband);
    _node.get_parameter("target_timeout", _paramTargetTimeout);

    _node.get_parameter("descent_kp_pid", _paramTrackingKp);
    _node.get_parameter("descent_ki_pid", _paramTrackingKi);
    _node.get_parameter("descent_kd_pid", _paramTrackingKd);
    _node.get_parameter("descent_max_velocity", _paramTrackingMaxVelocity);
    _node.get_parameter("slew_acc", _paramSlewAcc);

    _node.get_parameter("yaw.enabled", _paramYawControlEnabled);
    _node.get_parameter("yaw.kp", _paramYawKp);
    _node.get_parameter("yaw.max_rate_rad_s", _paramYawMaxRateRadS);
    _node.get_parameter("yaw.slew_acc_rad_s2", _paramYawSlewAccRadS2);
    _node.get_parameter("yaw.deadband_rad", _paramYawDeadbandRad);

    _node.get_parameter("tracking_height", _paramTrackingHeight);
    _node.get_parameter("height_tolerance", _paramHeightTolerance);
    _node.get_parameter("height_kp", _paramHeightKp);
    _node.get_parameter("vertical_slew_acc", _paramVerticalSlewAcc);
    _node.get_parameter("descent_gate_radius", _paramDescentGateRadius);
    _node.get_parameter("vmin", _paramVmin);
    _node.get_parameter("vmax", _paramVmax);

    _node.get_parameter("use_predictive_error", _paramUsePredictiveError);
    _node.get_parameter("prediction_dt_max", _paramPredictionDtMax);
    _node.get_parameter("control_extra_lead_sec", _paramControlExtraLeadSec);
    _node.get_parameter("predictive_acc_gain", _paramPredictiveAccGain);
    _node.get_parameter("predictive_acc_lpf_alpha", _paramPredictiveAccLpfAlpha);
    _node.get_parameter("predictive_acc_max", _paramPredictiveAccMax);

    _node.get_parameter("v_wind_n", _paramVWindN);
    _node.get_parameter("v_wind_e", _paramVWindE);
    _node.get_parameter("v_wind_d", _paramVWindD);
    _node.get_parameter("m_payload", _paramPayloadMassKg);
    _node.get_parameter("c_d", _paramCd);
    _node.get_parameter("area_m2", _paramAreaM2);
    _node.get_parameter("rho_air", _paramRhoAir);
    _node.get_parameter("drop_dt", _paramDropDtSec);
    _node.get_parameter("drop_t_max", _paramDropMaxTimeSec);

    _node.get_parameter("release.delay_sec", _paramReleaseDelaySec);
    _node.get_parameter("release.max_error_m", _paramReleaseMaxErrorM);
    _node.get_parameter("release.max_sigma_m", _paramReleaseMaxSigmaM);
    _node.get_parameter("release.max_relative_velocity_m_s", _paramReleaseMaxRelativeVelocityMps);
    _node.get_parameter("release.max_target_age_s", _paramReleaseMaxTargetAgeSec);
    _node.get_parameter("release.confirm_cycles", _paramReleaseConfirmCycles);
    _node.get_parameter("release.cov_timeout_s", _paramReleaseCovTimeoutSec);
    _node.get_parameter("release.payload_sigma_n_m", _paramPayloadSigmaN);
    _node.get_parameter("release.payload_sigma_e_m", _paramPayloadSigmaE);

    std::transform(
        _paramTargetModel.begin(),
        _paramTargetModel.end(),
        _paramTargetModel.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (_paramTargetModel == "cv" || _paramTargetModel == "kf")
    {
        _targetModel = TargetModel::Cv;
    }
    else if (_paramTargetModel == "ctra" || _paramTargetModel == "ekf")
    {
        _targetModel = TargetModel::Ctra;
    }
    else
    {
        throw std::runtime_error("estimator.model must be kf/cv or ekf/ctra");
    }

    _paramEstimatorMotionTimeoutSec = std::max(_paramEstimatorMotionTimeoutSec, 0.01f);
    _paramEstimatorPredictionStepSec = std::max(_paramEstimatorPredictionStepSec, 0.001f);

    _paramPidDeadband = std::max(_paramPidDeadband, 0.0f);
    _paramTargetTimeout = std::max(_paramTargetTimeout, 0.01f);
    _paramTrackingMaxVelocity = std::max(_paramTrackingMaxVelocity, 0.0f);
    _paramSlewAcc = std::max(_paramSlewAcc, 0.0f);

    _paramYawKp = std::max(_paramYawKp, 0.0f);
    _paramYawMaxRateRadS = std::max(_paramYawMaxRateRadS, 0.0f);
    _paramYawSlewAccRadS2 = std::max(_paramYawSlewAccRadS2, 0.0f);
    _paramYawDeadbandRad = std::max(_paramYawDeadbandRad, 0.0f);

    _paramTrackingHeight = std::max(_paramTrackingHeight, 0.1f);
    _paramHeightTolerance = std::max(_paramHeightTolerance, 0.0f);
    _paramHeightKp = std::max(_paramHeightKp, 0.0f);
    _paramVerticalSlewAcc = std::max(_paramVerticalSlewAcc, 0.0f);
    _paramDescentGateRadius = std::max(_paramDescentGateRadius, 0.0f);
    _paramVmin = std::max(_paramVmin, 0.0f);
    _paramVmax = std::max(_paramVmax, _paramVmin);

    _paramPredictionDtMax = std::max(_paramPredictionDtMax, 0.0f);
    _paramControlExtraLeadSec = std::max(_paramControlExtraLeadSec, 0.0f);
    _paramPredictiveAccGain = std::max(_paramPredictiveAccGain, 0.0f);
    _paramPredictiveAccLpfAlpha = std::clamp(_paramPredictiveAccLpfAlpha, 0.0f, 1.0f);
    _paramPredictiveAccMax = std::max(_paramPredictiveAccMax, 0.0f);

    _paramReleaseDelaySec = std::max(_paramReleaseDelaySec, 0.0f);
    _paramReleaseMaxErrorM = std::max(_paramReleaseMaxErrorM, 0.0f);
    _paramReleaseMaxSigmaM = std::max(_paramReleaseMaxSigmaM, 0.0f);
    _paramReleaseMaxRelativeVelocityMps = std::max(_paramReleaseMaxRelativeVelocityMps, 0.0f);
    _paramReleaseMaxTargetAgeSec = std::max(_paramReleaseMaxTargetAgeSec, 0.01f);
    _paramReleaseConfirmCycles = std::max(_paramReleaseConfirmCycles, 1);
    _paramReleaseCovTimeoutSec = std::max(_paramReleaseCovTimeoutSec, 0.01f);
    _paramPayloadSigmaN = std::max(_paramPayloadSigmaN, 0.0f);
    _paramPayloadSigmaE = std::max(_paramPayloadSigmaE, 0.0f);

    if (!std::isfinite(_paramVWindN) ||
        !std::isfinite(_paramVWindE) ||
        !std::isfinite(_paramVWindD))
    {
        throw std::runtime_error("v_wind_n/e/d must be finite");
    }

    if (!std::isfinite(_paramPayloadMassKg) || _paramPayloadMassKg <= 0.0f ||
        !std::isfinite(_paramCd) || _paramCd < 0.0f ||
        !std::isfinite(_paramAreaM2) || _paramAreaM2 < 0.0f ||
        !std::isfinite(_paramRhoAir) || _paramRhoAir <= 0.0f ||
        !std::isfinite(_paramDropDtSec) || _paramDropDtSec <= 0.0f ||
        !std::isfinite(_paramDropMaxTimeSec) || _paramDropMaxTimeSec <= 0.0f)
    {
        throw std::runtime_error("invalid payload drop model parameters");
    }
}

void TargetDrop::targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_active || msg == nullptr)
    {
        return;
    }

    _targetWorld.position = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    _targetWorld.yawRad = yawFromPose(msg->pose);
    _targetWorld.validYaw = std::isfinite(_targetWorld.yawRad);

    rclcpp::Time timestamp = msg->header.stamp;
    if (timestamp.nanoseconds() == 0)
    {
        timestamp = _node.now();
    }

    _targetWorld.timestamp = timestamp;
    _targetWorld.validPose = _targetWorld.position.allFinite();
}

void TargetDrop::targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_active || msg == nullptr)
    {
        return;
    }

    _targetWorld.velocity = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    rclcpp::Time timestamp = msg->header.stamp;
    if (timestamp.nanoseconds() == 0)
    {
        timestamp = _node.now();
    }

    _targetWorld.velocityTimestamp = timestamp;
    _targetWorld.validVelocity = _targetWorld.velocity.allFinite();
}

void TargetDrop::targetMotionCallback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    _targetWorld.validMotion = false;

    if (!_active || msg == nullptr || msg->data.size() != 2)
    {
        return;
    }

    const float tangentialAcc = static_cast<float>(msg->data[0]);
    const float turnRate = static_cast<float>(msg->data[1]);

    if (!std::isfinite(tangentialAcc) || !std::isfinite(turnRate))
    {
        return;
    }

    _targetWorld.tangentialAccMps2 = tangentialAcc;
    _targetWorld.turnRateRadS = turnRate;
    _targetWorld.motionTimestamp = _node.now();
    _targetWorld.validMotion = true;
}

void TargetDrop::targetCovarianceCallback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    _targetCov.valid = false;

    if (!_active || msg == nullptr)
    {
        return;
    }

    if (_targetModel == TargetModel::Cv)
    {
        if (msg->data.size() != 16)
        {
            return;
        }

        Eigen::Matrix4f covariance;
        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                covariance(row, col) = static_cast<float>(
                    msg->data[static_cast<std::size_t>(row * 4 + col)]);
            }
        }

        if (!covariance.allFinite())
        {
            return;
        }

        _targetCov.cv = 0.5f * (covariance + covariance.transpose());
    }
    else
    {
        if (msg->data.size() != 36)
        {
            return;
        }

        DropPred::Matrix6f covariance;
        for (int row = 0; row < 6; ++row)
        {
            for (int col = 0; col < 6; ++col)
            {
                covariance(row, col) = static_cast<float>(
                    msg->data[static_cast<std::size_t>(row * 6 + col)]);
            }
        }

        if (!covariance.allFinite())
        {
            return;
        }

        _targetCov.ctra = 0.5f * (covariance + covariance.transpose());
    }

    _targetCov.timestamp = _node.now();
    _targetCov.valid = true;
}

void TargetDrop::targetProcessNoiseCallback(
    const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg == nullptr || msg->data.size() != 2)
    {
        return;
    }

    const float primary = static_cast<float>(msg->data[0]);
    const float secondary = static_cast<float>(msg->data[1]);

    if (!std::isfinite(primary) || !std::isfinite(secondary) ||
        primary < 0.0f || secondary < 0.0f)
    {
        return;
    }

    _targetNoise.primary = primary;
    _targetNoise.secondary = secondary;
    _targetNoise.valid = true;
}

void TargetDrop::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

    _distBottomValid =
        std::isfinite(msg->dist_bottom) && msg->dist_bottom > 0.0f;

    if (_distBottomValid)
    {
        _distBottom = msg->dist_bottom;
    }
}

void TargetDrop::onActivate()
{
    _prevVehicleVelX = 0.0f;
    _prevVehicleVelY = 0.0f;
    _vehicleAccXFilt = 0.0f;
    _vehicleAccYFilt = 0.0f;
    _prevVehicleVelValid = false;

    resetXyController();
    resetYawController();
    resetZController();

    _active = true;
    _distBottomValid = false;
    _dropOutput = {};
    _targetWorld = {};
    _targetCov = {};
    resetReleaseGate();
    switchToState(State::Search);
}

void TargetDrop::onDeactivate()
{
    _active = false;
    _dropOutput = {};
    _targetWorld = {};
    _targetCov = {};
    resetReleaseGate();
    resetXyController();
    resetYawController();
    resetZController();
}

void TargetDrop::hover()
{
    _yawRateSpRadS = 0.0f;

    _trajectorySetpoint->update(
        Eigen::Vector3f::Zero(),
        std::nullopt,
        std::nullopt,
        0.0f);
}

void TargetDrop::updateSetpoint(float dtSec)
{
    const bool targetLost = checkTargetTimeout();

    switch (_state)
    {
    case State::Search:
        handleSearchState(targetLost);
        break;

    case State::Track:
        handleTrackState(dtSec, targetLost);
        break;
    }
}

void TargetDrop::handleSearchState(bool targetLost)
{
    if (!targetLost && _targetWorld.validPose)
    {
        switchToState(State::Track);
        return;
    }

    resetReleaseGate();
    hover();
}

void TargetDrop::handleTrackState(float dtSec, bool targetLost)
{
    if (targetLost)
    {
        resetReleaseGate();
        switchToState(State::Search);
        hover();
        return;
    }

    const rclcpp::Time controlTime = _node.now();

    try
    {
        const PredictionInput predictionInput = buildPredictionInput(dtSec, controlTime);
        const PredictionOutput predictionOutput = predictTarget(predictionInput);

        XYControllerInput xyInput{};
        xyInput.futureErrorXY = predictionOutput.futureErrorXY;
        xyInput.dtSec = dtSec;
        xyInput.targetValid = true;

        const bool velocityTimestampValid = _targetWorld.velocityTimestamp.nanoseconds() != 0;
        const float velocityAgeSec =
            (_targetWorld.validVelocity && velocityTimestampValid)
                ? static_cast<float>((controlTime - _targetWorld.velocityTimestamp).seconds())
                : kVelocityDataTimeoutSec + 1.0f;

        const bool velocityFresh =
            _targetWorld.validVelocity &&
            velocityTimestampValid &&
            velocityAgeSec >= 0.0f &&
            velocityAgeSec <= kVelocityDataTimeoutSec;

        xyInput.targetVelocityXY =
            velocityFresh ? predictionInput.target.velocityWorld.head<2>().eval()
                          : Eigen::Vector2f::Zero();
        xyInput.useTargetFeedforward = velocityFresh;

        const XYControllerOutput xyOutput = updateXyController(xyInput);
        const YawControllerOutput yawOutput = updateYawController(
            dtSec,
            _targetWorld.yawRad,
            _targetWorld.validYaw);

        const float verticalVelocity = computeZVelocityCommand(
            _distBottom,
            predictionOutput.futureErrorXY,
            dtSec);

        std::optional<float> yawRateSp = std::nullopt;
        if (_paramYawControlEnabled)
        {
            yawRateSp = yawOutput.yawRateSpRadS;
        }

        _trajectorySetpoint->update(
            Eigen::Vector3f(
                xyOutput.velocitySpXY.x(),
                xyOutput.velocitySpXY.y(),
                verticalVelocity),
            std::nullopt,
            std::nullopt,
            yawRateSp);

        updateDropPrediction();
        updateReleaseGate(controlTime);
    }
    catch (...)
    {
        resetReleaseGate();
        hover();
    }
}

void TargetDrop::resetXyController()
{
    _velXIntegral = 0.0f;
    _velYIntegral = 0.0f;
    _prevErrX = 0.0f;
    _prevErrY = 0.0f;
    _prevErrValid = false;
    _vxFilt = 0.0f;
    _vyFilt = 0.0f;
}

void TargetDrop::resetYawController()
{
    _yawRateSpRadS = 0.0f;
}

void TargetDrop::resetZController()
{
    _vzFilt = 0.0f;
}

bool TargetDrop::checkTargetTimeout() const
{
    if (!_targetWorld.validPose)
    {
        return true;
    }

    return (_node.now() - _targetWorld.timestamp).seconds() > _paramTargetTimeout;
}

void TargetDrop::switchToState(State state)
{
    _state = state;
}

float TargetDrop::computeLeadTimeSec(float dtSec, const rclcpp::Time &controlTime) const
{
    float poseAgeSec = static_cast<float>((controlTime - _targetWorld.timestamp).seconds());
    poseAgeSec = std::max(poseAgeSec, 0.0f);

    float velocityAgeSec = poseAgeSec;
    if (_targetWorld.validVelocity)
    {
        velocityAgeSec = static_cast<float>(
            (controlTime - _targetWorld.velocityTimestamp).seconds());
        velocityAgeSec = std::max(velocityAgeSec, 0.0f);
    }

    float leadDtSec = poseAgeSec;
    if (_paramUsePredictiveError && _targetWorld.validVelocity)
    {
        leadDtSec = std::max(poseAgeSec, velocityAgeSec);
    }

    leadDtSec += std::max(dtSec, 0.0f);
    leadDtSec += _paramControlExtraLeadSec;

    return std::clamp(leadDtSec, 0.0f, _paramPredictionDtMax);
}

TargetDrop::PredictionInput TargetDrop::buildPredictionInput(
    float dtSec,
    const rclcpp::Time &controlTime)
{
    PredictionInput input{};

    input.leadDtSec = computeLeadTimeSec(dtSec, controlTime);
    input.predictiveAccGain = _paramPredictiveAccGain;

    input.vehicle.positionWorld = _vehicleLocalPosition->positionNed();
    input.vehicle.velocityWorld = _vehicleLocalPosition->velocityNed();
    input.vehicle.accelerationXY = estimateVehicleAccelerationXY(dtSec);

    input.target.positionWorld = Eigen::Vector3f(
        static_cast<float>(_targetWorld.position.x()),
        static_cast<float>(_targetWorld.position.y()),
        static_cast<float>(_targetWorld.position.z()));

    const bool velocityTimestampValid = _targetWorld.velocityTimestamp.nanoseconds() != 0;
    const float velocityAgeSec =
        (_targetWorld.validVelocity && velocityTimestampValid)
            ? static_cast<float>((controlTime - _targetWorld.velocityTimestamp).seconds())
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

TargetDrop::PredictionOutput TargetDrop::predictTarget(const PredictionInput &input) const
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
            0.5f * input.predictiveAccGain * input.vehicle.accelerationXY.x() *
                input.leadDtSec * input.leadDtSec;

        output.vehicleFutureWorld.y() +=
            input.vehicle.velocityWorld.y() * input.leadDtSec +
            0.5f * input.predictiveAccGain * input.vehicle.accelerationXY.y() *
                input.leadDtSec * input.leadDtSec;

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

Eigen::Vector2f TargetDrop::estimateVehicleAccelerationXY(float dtSec)
{
    const float dt = std::max(dtSec, 1e-3f);
    const Eigen::Vector3f vehicleVelocity = _vehicleLocalPosition->velocityNed();

    const float currentVelX = vehicleVelocity.x();
    const float currentVelY = vehicleVelocity.y();

    if (!_prevVehicleVelValid)
    {
        _prevVehicleVelX = currentVelX;
        _prevVehicleVelY = currentVelY;
        _prevVehicleVelValid = true;
        return Eigen::Vector2f::Zero();
    }

    float accXRaw = (currentVelX - _prevVehicleVelX) / dt;
    float accYRaw = (currentVelY - _prevVehicleVelY) / dt;

    accXRaw = std::clamp(accXRaw, -_paramPredictiveAccMax, _paramPredictiveAccMax);
    accYRaw = std::clamp(accYRaw, -_paramPredictiveAccMax, _paramPredictiveAccMax);

    _vehicleAccXFilt =
        _paramPredictiveAccLpfAlpha * accXRaw +
        (1.0f - _paramPredictiveAccLpfAlpha) * _vehicleAccXFilt;
    _vehicleAccYFilt =
        _paramPredictiveAccLpfAlpha * accYRaw +
        (1.0f - _paramPredictiveAccLpfAlpha) * _vehicleAccYFilt;

    _prevVehicleVelX = currentVelX;
    _prevVehicleVelY = currentVelY;

    return Eigen::Vector2f(_vehicleAccXFilt, _vehicleAccYFilt);
}

Eigen::Vector2f TargetDrop::clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const
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

TargetDrop::XYControllerOutput TargetDrop::updateXyController(const XYControllerInput &input)
{
    XYControllerOutput output{};

    const float dt = std::max(input.dtSec, 1e-3f);
    const float errX = input.futureErrorXY.x();
    const float errY = input.futureErrorXY.y();

    const float xp = _paramTrackingKp * errX;
    const float yp = _paramTrackingKp * errY;

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
    if (_paramTrackingKi > 1e-6f)
    {
        const float maxIntegral =
            0.15f * _paramTrackingMaxVelocity / _paramTrackingKi;
        _velXIntegral = std::clamp(_velXIntegral, -maxIntegral, maxIntegral);
        _velYIntegral = std::clamp(_velYIntegral, -maxIntegral, maxIntegral);
        xi = _paramTrackingKi * _velXIntegral;
        yi = _paramTrackingKi * _velYIntegral;
    }

    float xd = 0.0f;
    float yd = 0.0f;
    if (input.targetValid && _paramTrackingKd > 1e-6f && _prevErrValid)
    {
        xd = _paramTrackingKd * (errX - _prevErrX) / dt;
        yd = _paramTrackingKd * (errY - _prevErrY) / dt;
    }

    _prevErrX = errX;
    _prevErrY = errY;
    _prevErrValid = input.targetValid;

    output.feedbackXY.x() = xp + xi + xd;
    output.feedbackXY.y() = yp + yi + yd;
    output.feedbackXY = clampVectorNorm(output.feedbackXY, _paramTrackingMaxVelocity);

    output.commandRawXY = output.feedbackXY;
    if (input.useTargetFeedforward)
    {
        output.commandRawXY += input.targetVelocityXY;
    }

    output.commandRawXY = clampVectorNorm(
        output.commandRawXY,
        _paramTrackingMaxVelocity);

    _vxFilt = applySlew(output.commandRawXY.x(), _vxFilt, _paramSlewAcc, dt);
    _vyFilt = applySlew(output.commandRawXY.y(), _vyFilt, _paramSlewAcc, dt);

    output.velocitySpXY.x() = _vxFilt;
    output.velocitySpXY.y() = _vyFilt;
    output.velocitySpXY = clampVectorNorm(
        output.velocitySpXY,
        _paramTrackingMaxVelocity);

    if (!std::isfinite(output.velocitySpXY.x()) || !std::isfinite(output.velocitySpXY.y()))
    {
        throw std::runtime_error("xy controller output is not finite");
    }

    return output;
}

TargetDrop::YawControllerOutput TargetDrop::updateYawController(
    float dtSec,
    float targetYawRad,
    bool targetYawValid)
{
    YawControllerOutput output{};
    output.yawRateSpRadS = _yawRateSpRadS;

    if (!_paramYawControlEnabled || !targetYawValid)
    {
        _yawRateSpRadS = applyYawSlew(
            0.0f,
            _yawRateSpRadS,
            _paramYawSlewAccRadS2,
            dtSec);
        output.yawRateSpRadS = _yawRateSpRadS;
        return output;
    }

    const float currentYawRad = px4_ros2::quaternionToYaw(_vehicleAttitude->attitude());
    if (!std::isfinite(currentYawRad) || !std::isfinite(targetYawRad))
    {
        _yawRateSpRadS = applyYawSlew(
            0.0f,
            _yawRateSpRadS,
            _paramYawSlewAccRadS2,
            dtSec);
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

    _yawRateSpRadS = applyYawSlew(
        yawRateRawRadS,
        _yawRateSpRadS,
        _paramYawSlewAccRadS2,
        dtSec);

    output.valid = true;
    output.currentYawRad = currentYawRad;
    output.targetYawRad = targetYawRad;
    output.errorYawRad = yawErrorRad;
    output.yawRateRawRadS = yawRateRawRadS;
    output.yawRateSpRadS = _yawRateSpRadS;
    output.yawTurnDirection =
        (yawErrorRad > 0.0f) ? 1 : ((yawErrorRad < 0.0f) ? -1 : 0);

    return output;
}

float TargetDrop::computeZVelocityCommand(
    float distanceBottom,
    const Eigen::Vector2f &futureErrorXY,
    float dtSec)
{
    float commandVelocity = 0.0f;

    if (_distBottomValid && std::isfinite(distanceBottom) && distanceBottom > 0.0f)
    {
        const float heightError = distanceBottom - _paramTrackingHeight;

        if (std::abs(heightError) > _paramHeightTolerance)
        {
            if (heightError > 0.0f)
            {
                const float lateralError = futureErrorXY.norm();
                if (lateralError < _paramDescentGateRadius)
                {
                    const float denominator = std::max(_paramDescentGateRadius, 1e-6f);
                    const float centeringScale = std::clamp(
                        1.0f - lateralError / denominator,
                        0.0f,
                        1.0f);

                    const float descentLimit =
                        _paramVmin + (_paramVmax - _paramVmin) * centeringScale;

                    commandVelocity = std::min(
                        _paramHeightKp * heightError,
                        descentLimit);
                }
            }
            else
            {
                commandVelocity = std::max(
                    _paramHeightKp * heightError,
                    -_paramVmax);
            }
        }
    }

    _vzFilt = applySlew(
        commandVelocity,
        _vzFilt,
        _paramVerticalSlewAcc,
        dtSec);

    return _vzFilt;
}

bool TargetDrop::dropHeightReady() const
{
    return _distBottomValid &&
           std::isfinite(_distBottom) &&
           std::abs(_distBottom - _paramTrackingHeight) <= _paramHeightTolerance;
}

void TargetDrop::updateDropPrediction()
{
    _dropOutput = {};

    if (!dropHeightReady())
    {
        return;
    }

    DropPred::DropInput input{};
    input.velocityNed = _vehicleLocalPosition->velocityNed();
    input.vWindNed = Eigen::Vector3f(
        _paramVWindN,
        _paramVWindE,
        _paramVWindD);
    input.heightM = _distBottom;
    input.massKg = _paramPayloadMassKg;
    input.cd = _paramCd;
    input.areaM2 = _paramAreaM2;
    input.rhoAir = _paramRhoAir;
    input.dtSec = _paramDropDtSec;
    input.maxTimeSec = _paramDropMaxTimeSec;
    input.valid = input.velocityNed.allFinite();

    _dropOutput = _dropPred.predictDrop(input);
}

DropPred::TargetOutput TargetDrop::predictReleaseTarget(
    float predictionTimeSec) const
{
    DropPred::TargetOutput output{};

    if (!_targetWorld.validPose ||
        !_targetWorld.validVelocity ||
        !_targetCov.valid ||
        !_targetNoise.valid ||
        !std::isfinite(predictionTimeSec))
    {
        return output;
    }

    const Eigen::Vector2f positionNE =
        _targetWorld.position.head<2>().cast<float>();
    const Eigen::Vector2f velocityNE =
        _targetWorld.velocity.head<2>().cast<float>();

    if (_targetModel == TargetModel::Cv)
    {
        DropPred::CvInput input{};
        input.positionNE = positionNE;
        input.velocityNE = velocityNE;
        input.covariance = _targetCov.cv;
        input.predictionTimeSec = predictionTimeSec;
        input.qAccN = _targetNoise.primary;
        input.qAccE = _targetNoise.secondary;
        input.valid = true;
        return _dropPred.predictCv(input);
    }

    if (!_targetWorld.validYaw || !_targetWorld.validMotion)
    {
        return output;
    }

    DropPred::CtraInput input{};
    input.positionNE = positionNE;
    input.speedMps = velocityNE.norm();
    input.headingRad = _targetWorld.yawRad;
    input.tangentialAccMps2 = _targetWorld.tangentialAccMps2;
    input.turnRateRadS = _targetWorld.turnRateRadS;
    input.covariance = _targetCov.ctra;
    input.predictionTimeSec = predictionTimeSec;
    input.stepSec = _paramEstimatorPredictionStepSec;
    input.qAcc = _targetNoise.primary;
    input.qTurnRate = _targetNoise.secondary;
    input.valid = true;
    return _dropPred.predictCtra(input);
}

void TargetDrop::updateReleaseGate(const rclcpp::Time &controlTime)
{
    if (!dropHeightReady() || !_dropOutput.valid)
    {
        resetReleaseGate();
        return;
    }

    const bool poseTimestampValid = _targetWorld.timestamp.nanoseconds() != 0;
    const bool velocityTimestampValid =
        _targetWorld.velocityTimestamp.nanoseconds() != 0;
    const bool covarianceTimestampValid = _targetCov.timestamp.nanoseconds() != 0;

    const float targetAgeSec =
        (_targetWorld.validPose && poseTimestampValid)
            ? static_cast<float>((controlTime - _targetWorld.timestamp).seconds())
            : _paramReleaseMaxTargetAgeSec + 1.0f;

    const float velocityAgeSec =
        (_targetWorld.validVelocity && velocityTimestampValid)
            ? static_cast<float>((controlTime - _targetWorld.velocityTimestamp).seconds())
            : kVelocityDataTimeoutSec + 1.0f;

    const float covarianceAgeSec =
        (_targetCov.valid && covarianceTimestampValid)
            ? static_cast<float>((controlTime - _targetCov.timestamp).seconds())
            : _paramReleaseCovTimeoutSec + 1.0f;

    const bool velocityFresh =
        _targetWorld.validVelocity &&
        velocityTimestampValid &&
        velocityAgeSec >= 0.0f &&
        velocityAgeSec <= kVelocityDataTimeoutSec;

    const bool covarianceFresh =
        _targetCov.valid &&
        covarianceTimestampValid &&
        covarianceAgeSec >= 0.0f &&
        covarianceAgeSec <= _paramReleaseCovTimeoutSec;

    bool motionFresh = true;
    if (_targetModel == TargetModel::Ctra)
    {
        const bool motionTimestampValid =
            _targetWorld.motionTimestamp.nanoseconds() != 0;
        const float motionAgeSec =
            (_targetWorld.validMotion && motionTimestampValid)
                ? static_cast<float>(
                      (controlTime - _targetWorld.motionTimestamp).seconds())
                : _paramEstimatorMotionTimeoutSec + 1.0f;

        motionFresh =
            _targetWorld.validMotion &&
            _targetWorld.validYaw &&
            motionTimestampValid &&
            motionAgeSec >= 0.0f &&
            motionAgeSec <= _paramEstimatorMotionTimeoutSec;
    }

    if (!velocityFresh || !covarianceFresh || !motionFresh)
    {
        resetReleaseGate();
        return;
    }

    const float impactPredictionTimeSec =
        _paramReleaseDelaySec + _dropOutput.impactTimeSec;

    const DropPred::TargetOutput targetAtSeparation =
        predictReleaseTarget(_paramReleaseDelaySec);
    const DropPred::TargetOutput targetAtImpact =
        predictReleaseTarget(impactPredictionTimeSec);

    if (!targetAtSeparation.valid || !targetAtImpact.valid)
    {
        resetReleaseGate();
        return;
    }

    const Eigen::Vector3f vehiclePosition =
        _vehicleLocalPosition->positionNed();
    const Eigen::Vector3f vehicleVelocity =
        _vehicleLocalPosition->velocityNed();

    // Short-horizon UAV state at physical payload separation.
    // The tracking controller continues running; over the measured actuator
    // delay we use a constant-velocity approximation for the release point.
    const Eigen::Vector3f vehicleSeparationPosition =
        vehiclePosition + vehicleVelocity * _paramReleaseDelaySec;
    const Eigen::Vector3f payloadImpactPosition =
        vehicleSeparationPosition + _dropOutput.impactOffsetNed;

    if (!vehiclePosition.allFinite() ||
        !vehicleVelocity.allFinite() ||
        !vehicleSeparationPosition.allFinite() ||
        !payloadImpactPosition.allFinite())
    {
        resetReleaseGate();
        return;
    }

    const Eigen::Vector2f errorNE =
        payloadImpactPosition.head<2>() - targetAtImpact.positionNE;

    Eigen::Matrix2f combinedCovariance = targetAtImpact.covarianceNE;
    combinedCovariance(0, 0) +=
        _paramPayloadSigmaN * _paramPayloadSigmaN;
    combinedCovariance(1, 1) +=
        _paramPayloadSigmaE * _paramPayloadSigmaE;

    DropGate::Input gateInput{};
    gateInput.errorNE = errorNE;
    gateInput.covarianceNE = combinedCovariance;
    gateInput.relativeVelocityNE =
        vehicleVelocity.head<2>() - targetAtSeparation.velocityNE;
    gateInput.heightErrorM = _distBottom - _paramTrackingHeight;
    gateInput.targetAgeSec = targetAgeSec;
    gateInput.vehicleReady =
        _active &&
        vehiclePosition.allFinite() &&
        vehicleVelocity.allFinite();
    gateInput.valid =
        errorNE.allFinite() &&
        combinedCovariance.allFinite() &&
        gateInput.relativeVelocityNE.allFinite() &&
        std::isfinite(targetAgeSec);

    DropGate::Limits limits{};
    limits.maxErrorM = _paramReleaseMaxErrorM;
    limits.maxSigmaM = _paramReleaseMaxSigmaM;
    limits.maxRelativeVelocityMps = _paramReleaseMaxRelativeVelocityMps;
    limits.maxHeightErrorM = _paramHeightTolerance;
    limits.maxTargetAgeSec = _paramReleaseMaxTargetAgeSec;
    limits.confirmCycles = _paramReleaseConfirmCycles;

    _gateOutput = _dropGate.update(gateInput, limits);

    // _gateOutput.release is intentionally not connected to the servo yet.
}

void TargetDrop::resetReleaseGate()
{
    _dropGate.reset();
    _gateOutput = {};
}

float TargetDrop::applySlew(
    float commandVelocity,
    float previousVelocity,
    float accelLimit,
    float dtSec) const
{
    const float dt = std::max(dtSec, 1e-3f);
    const float maxDeltaVelocity = std::max(accelLimit, 0.0f) * dt;
    const float deltaVelocity = std::clamp(
        commandVelocity - previousVelocity,
        -maxDeltaVelocity,
        maxDeltaVelocity);

    return previousVelocity + deltaVelocity;
}

float TargetDrop::applyYawSlew(
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

float TargetDrop::normalizeAnglePi(float angleRad) const
{
    return std::atan2(std::sin(angleRad), std::cos(angleRad));
}

float TargetDrop::yawFromPose(const geometry_msgs::msg::Pose &pose) const
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

    return static_cast<float>(
        std::atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)));
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(
        std::make_shared<px4_ros2::NodeWithMode<TargetDrop>>(
            kModeName,
            kEnableDebugOutput));
    rclcpp::shutdown();
    return 0;
}
