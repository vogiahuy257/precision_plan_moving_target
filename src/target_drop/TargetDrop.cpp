#include "TargetDrop.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

#include <px4_ros2/components/node_with_mode.hpp>

namespace
{
const std::string kModeName = "TGT_DROP";
constexpr bool kEnableDebugOutput = false;
} // namespace

TargetDrop::TargetDrop(rclcpp::Node &node)
    : ModeBase(node, kModeName),
      _node(node)
{
    _trajectorySetpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
    _vehicleLocalPosition = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);

    loadParameters();

    const auto qos = rclcpp::QoS(1).best_effort();
    _targetPoseSub = _node.create_subscription<geometry_msgs::msg::PoseStamped>(
        _targetPoseTopic, qos,
        std::bind(&TargetDrop::targetPoseCallback, this, std::placeholders::_1));

    _targetVelocitySub = _node.create_subscription<geometry_msgs::msg::PoseStamped>(
        _targetVelocityTopic, qos,
        std::bind(&TargetDrop::targetVelocityCallback, this, std::placeholders::_1));

    if (_targetModel == TargetModel::Ctra)
    {
        _targetMotionSub = _node.create_subscription<std_msgs::msg::Float64MultiArray>(
            _targetMotionTopic, qos,
            std::bind(&TargetDrop::targetMotionCallback, this, std::placeholders::_1));
    }

    _targetCovarianceSub = _node.create_subscription<std_msgs::msg::Float64MultiArray>(
        _targetCovarianceTopic, qos,
        std::bind(&TargetDrop::targetCovarianceCallback, this, std::placeholders::_1));

    const auto configQos = rclcpp::QoS(1).reliable().transient_local();
    _targetProcessNoiseSub = _node.create_subscription<std_msgs::msg::Float64MultiArray>(
        _targetProcessNoiseTopic, configQos,
        std::bind(&TargetDrop::targetProcessNoiseCallback, this, std::placeholders::_1));

    _targetStateSub = _node.create_subscription<std_msgs::msg::String>(
        _targetStateTopic, qos,
        std::bind(&TargetDrop::targetStateCallback, this, std::placeholders::_1));

    _vehicleLocalPositionSub = _node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
        _vehicleLocalPositionTopic, qos,
        std::bind(&TargetDrop::vehicleLocalPositionCallback, this, std::placeholders::_1));
    
    const auto loggerQos = rclcpp::QoS(1).reliable().transient_local();

    _loggerEnablePub = _node.create_publisher<std_msgs::msg::Bool>("/logger/enable",loggerQos);

    modeRequirements().manual_control = false;
}

void TargetDrop::loadParameters()
{
    _node.declare_parameter<std::string>("estimator.model", "kf");
    _node.declare_parameter<std::string>("topics.target_pose", "/KF/target_pose_est_NED");
    _node.declare_parameter<std::string>("topics.target_velocity", "/KF/target_velocity_est_NED");
    _node.declare_parameter<std::string>("topics.target_motion", "/EKF/target_motion");
    _node.declare_parameter<std::string>("topics.target_covariance", "/KF/target_covariance_NE");
    _node.declare_parameter<std::string>("topics.target_process_noise", "/KF/process_noise");
    _node.declare_parameter<std::string>("topics.target_state", "/Aruco/target_state");
    _node.declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position_v1");

    _node.declare_parameter<float>("controller.kp", 1.0f);
    _node.declare_parameter<float>("controller.ki", 0.0f);
    _node.declare_parameter<float>("controller.kd", 0.0f);
    _node.declare_parameter<float>("controller.deadband_m", 0.08f);
    _node.declare_parameter<float>("controller.max_velocity_m_s", 10.0f);
    _node.declare_parameter<float>("controller.slew_acc_m_s2", 0.88f);

    _node.declare_parameter<float>("altitude.release_height_m", 3.0f);
    _node.declare_parameter<float>("altitude.tolerance_m", 0.15f);
    _node.declare_parameter<float>("altitude.kp", 0.6f);
    _node.declare_parameter<float>("altitude.slew_acc_m_s2", 0.6f);
    _node.declare_parameter<float>("altitude.descent_gate_radius_m", 0.30f);
    _node.declare_parameter<float>("altitude.descent_min_m_s", 0.30f);
    _node.declare_parameter<float>("altitude.descent_max_m_s", 0.45f);

    _node.declare_parameter<float>("payload.wind_x_m_s", 0.0f);
    _node.declare_parameter<float>("payload.wind_y_m_s", 0.0f);
    _node.declare_parameter<float>("payload.wind_z_m_s", 0.0f);
    _node.declare_parameter<float>("payload.mass_kg", 0.5f);
    _node.declare_parameter<float>("payload.cd", 1.0f);
    _node.declare_parameter<float>("payload.area_m2", 0.01f);
    _node.declare_parameter<float>("payload.rho_air", 1.225f);
    _node.declare_parameter<float>("payload.integration_step_s", 0.005f);

    _node.declare_parameter<float>("release.max_error_m", 0.30f);
    _node.declare_parameter<float>("release.max_sigma_m", 0.20f);
    _node.declare_parameter<float>("release.payload_sigma_x_m", 0.0f);
    _node.declare_parameter<float>("release.payload_sigma_y_m", 0.0f);

    _node.get_parameter("estimator.model", _paramTargetModel);
    _node.get_parameter("topics.target_pose", _targetPoseTopic);
    _node.get_parameter("topics.target_velocity", _targetVelocityTopic);
    _node.get_parameter("topics.target_motion", _targetMotionTopic);
    _node.get_parameter("topics.target_covariance", _targetCovarianceTopic);
    _node.get_parameter("topics.target_process_noise", _targetProcessNoiseTopic);
    _node.get_parameter("topics.target_state", _targetStateTopic);
    _node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);

    _node.get_parameter("controller.kp", _paramKp);
    _node.get_parameter("controller.ki", _paramKi);
    _node.get_parameter("controller.kd", _paramKd);
    _node.get_parameter("controller.deadband_m", _paramDeadbandM);
    _node.get_parameter("controller.max_velocity_m_s", _paramMaxVelocityMps);
    _node.get_parameter("controller.slew_acc_m_s2", _paramSlewAccMps2);

    _node.get_parameter("altitude.release_height_m", _paramReleaseHeightM);
    _node.get_parameter("altitude.tolerance_m", _paramHeightToleranceM);
    _node.get_parameter("altitude.kp", _paramHeightKp);
    _node.get_parameter("altitude.slew_acc_m_s2", _paramVerticalSlewAccMps2);
    _node.get_parameter("altitude.descent_gate_radius_m", _paramDescentGateRadiusM);
    _node.get_parameter("altitude.descent_min_m_s", _paramDescentMinMps);
    _node.get_parameter("altitude.descent_max_m_s", _paramDescentMaxMps);

    float windX = 0.0f;
    float windY = 0.0f;
    float windZ = 0.0f;
    _node.get_parameter("payload.wind_x_m_s", windX);
    _node.get_parameter("payload.wind_y_m_s", windY);
    _node.get_parameter("payload.wind_z_m_s", windZ);
    _paramWindXyz = Eigen::Vector3f(windX, windY, windZ);
    _node.get_parameter("payload.mass_kg", _paramPayloadMassKg);
    _node.get_parameter("payload.cd", _paramCd);
    _node.get_parameter("payload.area_m2", _paramAreaM2);
    _node.get_parameter("payload.rho_air", _paramRhoAir);
    _node.get_parameter("payload.integration_step_s", _paramDropIntegrationStepSec);

    _node.get_parameter("release.max_error_m", _paramReleaseMaxErrorM);
    _node.get_parameter("release.max_sigma_m", _paramReleaseMaxSigmaM);
    _node.get_parameter("release.payload_sigma_x_m", _paramPayloadSigmaX);
    _node.get_parameter("release.payload_sigma_y_m", _paramPayloadSigmaY);

    std::transform(
        _paramTargetModel.begin(), _paramTargetModel.end(), _paramTargetModel.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (_paramTargetModel == "kf" || _paramTargetModel == "cv")
    {
        _targetModel = TargetModel::Cv;
    }
    else if (_paramTargetModel == "ekf" || _paramTargetModel == "ctra")
    {
        _targetModel = TargetModel::Ctra;
    }
    else
    {
        throw std::runtime_error("estimator.model must be 'kf' or 'ekf'");
    }

    _paramKp = std::max(_paramKp, 0.0f);
    _paramKi = std::max(_paramKi, 0.0f);
    _paramKd = std::max(_paramKd, 0.0f);
    _paramDeadbandM = std::max(_paramDeadbandM, 0.0f);
    _paramMaxVelocityMps = std::max(_paramMaxVelocityMps, 0.0f);
    _paramSlewAccMps2 = std::max(_paramSlewAccMps2, 0.0f);

    _paramReleaseHeightM = std::max(_paramReleaseHeightM, 0.01f);
    _paramHeightToleranceM = std::max(_paramHeightToleranceM, 0.0f);
    _paramHeightKp = std::max(_paramHeightKp, 0.0f);
    _paramVerticalSlewAccMps2 = std::max(_paramVerticalSlewAccMps2, 0.0f);
    _paramDescentGateRadiusM = std::max(_paramDescentGateRadiusM, 0.0f);
    _paramDescentMinMps = std::max(_paramDescentMinMps, 0.0f);
    _paramDescentMaxMps = std::max(_paramDescentMaxMps, _paramDescentMinMps);

    _paramReleaseMaxErrorM = std::max(_paramReleaseMaxErrorM, 0.0f);
    _paramReleaseMaxSigmaM = std::max(_paramReleaseMaxSigmaM, 0.0f);
    _paramPayloadSigmaX = std::max(_paramPayloadSigmaX, 0.0f);
    _paramPayloadSigmaY = std::max(_paramPayloadSigmaY, 0.0f);

    if (!_paramWindXyz.allFinite() ||
        !std::isfinite(_paramPayloadMassKg) || _paramPayloadMassKg <= 0.0f ||
        !std::isfinite(_paramCd) || _paramCd < 0.0f ||
        !std::isfinite(_paramAreaM2) || _paramAreaM2 < 0.0f ||
        !std::isfinite(_paramRhoAir) || _paramRhoAir <= 0.0f ||
        !std::isfinite(_paramDropIntegrationStepSec) ||
        _paramDropIntegrationStepSec <= 0.0f)
    {
        throw std::runtime_error("invalid payload model parameters");
    }
}

void TargetDrop::targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_active || msg == nullptr)
    {
        return;
    }

    _target.position = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    _target.headingRad = headingFromPose(msg->pose);
    _target.headingValid = std::isfinite(_target.headingRad);

    _target.poseTime = msg->header.stamp;
    _target.poseValid =
        _target.position.allFinite() &&
        _target.poseTime.nanoseconds() > 0;

    if (_target.poseValid)
    {
        _target.active = true;
    }
}

void TargetDrop::targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (!_active || msg == nullptr)
    {
        return;
    }

    _target.velocity = Eigen::Vector3d(
        msg->pose.position.x,
        msg->pose.position.y,
        msg->pose.position.z);

    _target.velocityTime = msg->header.stamp;
    _target.velocityValid =
        _target.velocity.allFinite() &&
        _target.velocityTime.nanoseconds() > 0;
}

void TargetDrop::targetMotionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    _target.motionValid = false;

    if (!_active || msg == nullptr || msg->data.size() != 2)
    {
        return;
    }

    const float acceleration = static_cast<float>(msg->data[0]);
    const float turnRate = static_cast<float>(msg->data[1]);

    if (!std::isfinite(acceleration) || !std::isfinite(turnRate))
    {
        return;
    }

    _target.tangentialAccMps2 = acceleration;
    _target.turnRateRadS = turnRate;
    _target.motionValid = true;
}

void TargetDrop::targetCovarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
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
                covariance(row, col) = static_cast<float>(msg->data[row * 4 + col]);
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
                covariance(row, col) = static_cast<float>(msg->data[row * 6 + col]);
            }
        }

        if (!covariance.allFinite())
        {
            return;
        }

        _targetCov.ctra = 0.5f * (covariance + covariance.transpose());
    }

    _targetCov.valid = true;
}

void TargetDrop::targetProcessNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
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

void TargetDrop::targetStateCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (!_active || msg == nullptr)
    {
        return;
    }

    if (msg->data == "ACTIVE")
    {
        _target.active = true;
        return;
    }

    if (msg->data == "RESET")
    {
        _target = {};
        _targetCov = {};
        resetControllers();
        resetReleaseGate();
        switchState(State::Search);
        hover();
    }
}

void TargetDrop::vehicleLocalPositionCallback(
    const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

    _distBottomValid = std::isfinite(msg->dist_bottom) && msg->dist_bottom > 0.0f;
    if (_distBottomValid)
    {
        _distBottom = msg->dist_bottom;
    }
}

void TargetDrop::onActivate()
{
    _active = true;
    std_msgs::msg::Bool msg;
    msg.data = true;
    _loggerEnablePub->publish(msg);
    _distBottomValid = false;
    _target = {};
    _targetCov = {};
    _targetNoise = {};
    resetControllers();
    resetReleaseGate();
    switchState(State::Search);
}

void TargetDrop::onDeactivate()
{
    std_msgs::msg::Bool msg;
    msg.data = false;
    _loggerEnablePub->publish(msg);
    _active = false;
    _target = {};
    _targetCov = {};
    resetControllers();
    resetReleaseGate();
}

void TargetDrop::hover()
{
    _trajectorySetpoint->update(
        Eigen::Vector3f::Zero(),
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

void TargetDrop::updateSetpoint(float dtSec)
{
    if (_state == State::Search)
    {
        handleSearch();
    }
    else
    {
        handleTrack(dtSec);
    }
}

void TargetDrop::switchState(State state)
{
    if (_state == state)
    {
        return;
    }

    _state = state;
    resetControllers();
    resetReleaseGate();
}

void TargetDrop::handleSearch()
{
    if (_target.active && _target.poseValid && _target.velocityValid)
    {
        switchState(State::Track);
        return;
    }

    resetReleaseGate();
    hover();
}

void TargetDrop::handleTrack(float dtSec)
{
    if (!_target.active)
    {
        resetControllers();
        resetReleaseGate();
        switchState(State::Search);
        hover();
        return;
    }

    try
    {
        const ReleasePlan plan = buildReleasePlan(_node.now());
        if (!plan.valid)
        {
            resetControllers();
            resetReleaseGate();
            hover();
            return;
        }

        const Eigen::Vector2f velocityXY = updateXyController(
            plan.errorXY,
            plan.feedforwardVelocityXY,
            dtSec);

        const float velocityD = updateZController(
            _distBottom,
            plan.errorXY,
            dtSec);

        _trajectorySetpoint->update(
            Eigen::Vector3f(velocityXY.x(), velocityXY.y(), velocityD),
            std::nullopt,
            std::nullopt,
            std::nullopt);

        updateReleaseGate(plan);
    }
    catch (...)
    {
        resetControllers();
        resetReleaseGate();
        hover();
    }
}

TargetDrop::ReleasePlan TargetDrop::buildReleasePlan(
    const rclcpp::Time &controlTime) const
{
    ReleasePlan plan{};

    const Eigen::Vector3f vehiclePosition = _vehicleLocalPosition->positionNed();
    const Eigen::Vector3f vehicleVelocity = _vehicleLocalPosition->velocityNed();

    if (!vehiclePosition.allFinite() || !vehicleVelocity.allFinite() ||
        !_target.poseValid || !_target.velocityValid || !_targetNoise.valid)
    {
        return plan;
    }

    if (_target.poseTime.nanoseconds() <= 0 ||
        _target.velocityTime.nanoseconds() <= 0)
    {
        return plan;
    }

    // KF/EKF pose and velocity are published from the same estimator state.
    // Require the same measurement timestamp instead of accepting asynchronous states.
    if (_target.poseTime.nanoseconds() != _target.velocityTime.nanoseconds())
    {
        return plan;
    }

    plan.measurementDtSec =
        static_cast<float>((controlTime - _target.poseTime).seconds());

    if (!std::isfinite(plan.measurementDtSec) || plan.measurementDtSec < 0.0f)
    {
        return plan;
    }

    if (_targetModel == TargetModel::Ctra &&
        (!_target.headingValid || !_target.motionValid))
    {
        return plan;
    }

    const DropPred::DropOutput drop = predictPayload(_paramReleaseHeightM);
    if (!drop.valid)
    {
        return plan;
    }

    const float impactHorizonSec =
        plan.measurementDtSec + drop.impactTimeSec;

    const DropPred::TargetOutput targetAtImpact = predictTarget(impactHorizonSec);
    if (!targetAtImpact.valid)
    {
        return plan;
    }

    // Payload impact = separation position + drop offset.
    // Therefore the optimal separation point is target impact - drop offset.
    plan.desiredReleaseXY =
        targetAtImpact.positionXY - drop.impactOffsetNed.head<2>();

    // RELEASE means physical separation now: no modeled command delay.
    // The controller therefore drives the current UAV XY position to the
    // optimal release point.
    plan.errorXY =
        plan.desiredReleaseXY - vehiclePosition.head<2>();
    plan.feedforwardVelocityXY = targetAtImpact.velocityXY;

    plan.covarianceXY = targetAtImpact.covarianceXY;
    plan.covarianceXY(0, 0) += _paramPayloadSigmaX * _paramPayloadSigmaX;
    plan.covarianceXY(1, 1) += _paramPayloadSigmaY * _paramPayloadSigmaY;
    plan.valid =
        plan.desiredReleaseXY.allFinite() &&
        plan.errorXY.allFinite() &&
        plan.feedforwardVelocityXY.allFinite() &&
        plan.covarianceXY.allFinite();

    return plan;
}

DropPred::DropOutput TargetDrop::predictPayload(float releaseHeightM) const
{
    DropPred::DropInput input{};
    input.velocityNed = _vehicleLocalPosition->velocityNed();
    input.vWindXyz = _paramWindXyz;
    input.heightM = releaseHeightM;
    input.massKg = _paramPayloadMassKg;
    input.cd = _paramCd;
    input.areaM2 = _paramAreaM2;
    input.rhoAir = _paramRhoAir;
    input.integrationStepSec = _paramDropIntegrationStepSec;
    input.valid = input.velocityNed.allFinite();
    return _dropPred.predictDrop(input);
}

DropPred::TargetOutput TargetDrop::predictTarget(float predictionTimeSec) const
{
    DropPred::TargetOutput output{};

    if (!_target.poseValid || !_target.velocityValid ||
        !_targetCov.valid || !_targetNoise.valid ||
        !std::isfinite(predictionTimeSec))
    {
        return output;
    }

    const Eigen::Vector2f positionXY = _target.position.head<2>().cast<float>();
    const Eigen::Vector2f velocityXY = _target.velocity.head<2>().cast<float>();

    if (_targetModel == TargetModel::Cv)
    {
        DropPred::CvInput input{};
        input.positionXY = positionXY;
        input.velocityXY = velocityXY;
        input.covariance = _targetCov.cv;
        input.predictionTimeSec = predictionTimeSec;
        input.qAccX = _targetNoise.primary;
        input.qAccY = _targetNoise.secondary;
        input.valid = true;
        return _dropPred.predictCv(input);
    }

    if (!_target.headingValid || !_target.motionValid)
    {
        return output;
    }

    DropPred::CtraInput input{};
    input.positionXY = positionXY;
    input.speedMps = velocityXY.norm();
    input.headingRad = _target.headingRad;
    input.tangentialAccMps2 = _target.tangentialAccMps2;
    input.turnRateRadS = _target.turnRateRadS;
    input.covariance = _targetCov.ctra;
    input.predictionTimeSec = predictionTimeSec;
    input.qAcc = _targetNoise.primary;
    input.qTurnRate = _targetNoise.secondary;
    input.valid = true;
    return _dropPred.predictCtra(input);
}

Eigen::Vector2f TargetDrop::updateXyController(
    const Eigen::Vector2f &releaseErrorXY,
    const Eigen::Vector2f &feedforwardVelocityXY,
    float dtSec)
{
    if (!std::isfinite(dtSec) || dtSec <= 0.0f)
    {
        return _velocitySetpointXY;
    }

    const float dt = dtSec;
    Eigen::Vector2f error = releaseErrorXY;

    for (int axis = 0; axis < 2; ++axis)
    {
        if (std::abs(error(axis)) <= _paramDeadbandM)
        {
            error(axis) = 0.0f;
            _integralXY(axis) *= 0.9f;
        }
        else
        {
            _integralXY(axis) += error(axis) * dt;
        }
    }

    if (_paramKi > 1e-6f)
    {
        const float maxIntegral = 0.15f * _paramMaxVelocityMps / _paramKi;
        _integralXY = _integralXY.cwiseMax(-maxIntegral).cwiseMin(maxIntegral);
    }
    else
    {
        _integralXY.setZero();
    }

    Eigen::Vector2f derivative = Eigen::Vector2f::Zero();
    if (_previousErrorValid)
    {
        derivative = (error - _previousErrorXY) / dt;
    }

    _previousErrorXY = error;
    _previousErrorValid = true;

    const Eigen::Vector2f feedback =
        _paramKp * error +
        _paramKi * _integralXY +
        _paramKd * derivative;

    const Eigen::Vector2f command = clampNorm(
        feedforwardVelocityXY + feedback,
        _paramMaxVelocityMps);

    for (int axis = 0; axis < 2; ++axis)
    {
        _velocitySetpointXY(axis) = applySlew(
            command(axis),
            _velocitySetpointXY(axis),
            _paramSlewAccMps2,
            dt);
    }

    _velocitySetpointXY = clampNorm(_velocitySetpointXY, _paramMaxVelocityMps);

    if (!_velocitySetpointXY.allFinite())
    {
        throw std::runtime_error("XY controller output is not finite");
    }

    return _velocitySetpointXY;
}

float TargetDrop::updateZController(
    float distanceBottom,
    const Eigen::Vector2f &releaseErrorXY,
    float dtSec)
{
    float command = 0.0f;

    if (_distBottomValid && std::isfinite(distanceBottom) && distanceBottom > 0.0f)
    {
        const float heightError = distanceBottom - _paramReleaseHeightM;

        if (std::abs(heightError) > _paramHeightToleranceM)
        {
            if (heightError < 0.0f)
            {
                command = std::max(
                    _paramHeightKp * heightError,
                    -_paramDescentMaxMps);
            }
            else if (releaseErrorXY.norm() < _paramDescentGateRadiusM)
            {
                const float centered = std::clamp(
                    1.0f - releaseErrorXY.norm() /
                               std::max(_paramDescentGateRadiusM, 1e-6f),
                    0.0f,
                    1.0f);

                const float descentLimit =
                    _paramDescentMinMps +
                    (_paramDescentMaxMps - _paramDescentMinMps) * centered;

                command = std::min(_paramHeightKp * heightError, descentLimit);
            }
        }
    }

    _verticalVelocitySetpoint = applySlew(
        command,
        _verticalVelocitySetpoint,
        _paramVerticalSlewAccMps2,
        dtSec);

    return _verticalVelocitySetpoint;
}

void TargetDrop::updateReleaseGate(const ReleasePlan &plan)
{
    DropGate::Input input{};
    input.releaseErrorXY = plan.errorXY;
    input.covarianceXY = plan.covarianceXY;
    input.heightErrorM = _distBottom - _paramReleaseHeightM;
    input.vehicleReady =
        _active &&
        _target.active &&
        _distBottomValid &&
        _vehicleLocalPosition->positionNed().allFinite() &&
        _vehicleLocalPosition->velocityNed().allFinite();
    input.valid = plan.valid;

    DropGate::Limits limits{};
    limits.maxReleaseErrorM = _paramReleaseMaxErrorM;
    limits.maxSigmaM = _paramReleaseMaxSigmaM;
    limits.maxHeightErrorM = _paramHeightToleranceM;

    _gateOutput = _dropGate.update(input, limits);

    // Connect _gateOutput.release to the payload actuator when the servo layer is ready.
}

void TargetDrop::resetControllers()
{
    _integralXY.setZero();
    _previousErrorXY.setZero();
    _velocitySetpointXY.setZero();
    _verticalVelocitySetpoint = 0.0f;
    _previousErrorValid = false;
}

void TargetDrop::resetReleaseGate()
{
    _gateOutput = {};
}

Eigen::Vector2f TargetDrop::clampNorm(
    const Eigen::Vector2f &value,
    float maxNorm) const
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

float TargetDrop::applySlew(
    float command,
    float previous,
    float accelLimit,
    float dtSec) const
{
    if (!std::isfinite(dtSec) || dtSec <= 0.0f)
    {
        return previous;
    }

    const float maxDelta = std::max(accelLimit, 0.0f) * dtSec;
    return previous + std::clamp(command - previous, -maxDelta, maxDelta);
}

float TargetDrop::headingFromPose(const geometry_msgs::msg::Pose &pose) const
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
