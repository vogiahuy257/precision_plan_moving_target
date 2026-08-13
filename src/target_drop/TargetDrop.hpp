#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "DropGate.hpp"
#include "DropPred.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>

class TargetDrop : public px4_ros2::ModeBase
{
public:
    explicit TargetDrop(rclcpp::Node &node);

    void onActivate() override;
    void onDeactivate() override;
    void updateSetpoint(float dtSec) override;

private:
    enum class State
    {
        Search,
        Track
    };

    enum class TargetModel
    {
        Cv,
        Ctra
    };

    struct TargetWorldData
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        Eigen::Vector3d velocity{0.0, 0.0, 0.0};

        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        rclcpp::Time velocityTimestamp{0, 0, RCL_ROS_TIME};
        rclcpp::Time motionTimestamp{0, 0, RCL_ROS_TIME};

        float yawRad{0.0f};
        float tangentialAccMps2{0.0f};
        float turnRateRadS{0.0f};

        bool validPose{false};
        bool validVelocity{false};
        bool validYaw{false};
        bool validMotion{false};
    };

    struct TargetState
    {
        Eigen::Vector3f positionWorld{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f velocityWorld{0.0f, 0.0f, 0.0f};
        bool hasVelocity{false};
    };

    struct TargetCovData
    {
        Eigen::Matrix4f cv{Eigen::Matrix4f::Zero()};
        DropPred::Matrix6f ctra{DropPred::Matrix6f::Zero()};
        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        bool valid{false};
    };

    struct TargetNoiseData
    {
        float primary{0.0f};
        float secondary{0.0f};
        bool valid{false};
    };

    struct VehicleState
    {
        Eigen::Vector3f positionWorld{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f velocityWorld{0.0f, 0.0f, 0.0f};
        Eigen::Vector2f accelerationXY{0.0f, 0.0f};
    };

    struct PredictionInput
    {
        TargetState target{};
        VehicleState vehicle{};
        float leadDtSec{0.0f};
        float predictiveAccGain{0.0f};
    };

    struct PredictionOutput
    {
        Eigen::Vector3f targetFutureWorld{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f vehicleFutureWorld{0.0f, 0.0f, 0.0f};
        Eigen::Vector2f futureErrorXY{0.0f, 0.0f};
    };

    struct XYControllerInput
    {
        Eigen::Vector2f futureErrorXY{0.0f, 0.0f};
        Eigen::Vector2f targetVelocityXY{0.0f, 0.0f};
        bool useTargetFeedforward{false};
        bool targetValid{false};
        float dtSec{0.0f};
    };

    struct XYControllerOutput
    {
        Eigen::Vector2f velocitySpXY{0.0f, 0.0f};
        Eigen::Vector2f feedbackXY{0.0f, 0.0f};
        Eigen::Vector2f commandRawXY{0.0f, 0.0f};
    };

    struct YawControllerOutput
    {
        bool valid{false};
        float currentYawRad{0.0f};
        float targetYawRad{0.0f};
        float errorYawRad{0.0f};
        float yawRateRawRadS{0.0f};
        float yawRateSpRadS{0.0f};
        int yawTurnDirection{0};
    };

    void loadParameters();
    void hover();

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetMotionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void targetCovarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void targetProcessNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    void handleSearchState(bool targetLost);
    void handleTrackState(float dtSec, bool targetLost);

    void resetXyController();
    void resetYawController();
    void resetZController();

    bool checkTargetTimeout() const;
    void switchToState(State state);

    float computeLeadTimeSec(float dtSec, const rclcpp::Time &controlTime) const;
    PredictionInput buildPredictionInput(float dtSec, const rclcpp::Time &controlTime);
    PredictionOutput predictTarget(const PredictionInput &input) const;

    Eigen::Vector2f estimateVehicleAccelerationXY(float dtSec);
    Eigen::Vector2f clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const;

    XYControllerOutput updateXyController(const XYControllerInput &input);
    YawControllerOutput updateYawController(float dtSec, float targetYawRad, bool targetYawValid);

    float computeZVelocityCommand(
        float distanceBottom,
        const Eigen::Vector2f &futureErrorXY,
        float dtSec);

    bool dropHeightReady() const;
    void updateDropPrediction();
    DropPred::TargetOutput predictReleaseTarget(float predictionTimeSec) const;
    void updateReleaseGate(const rclcpp::Time &controlTime);
    void resetReleaseGate();

    float applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const;
    float applyYawSlew(float commandYawRate, float previousYawRate, float slewLimit, float dtSec) const;
    float normalizeAnglePi(float angleRad) const;
    float yawFromPose(const geometry_msgs::msg::Pose &pose) const;

private:
    rclcpp::Node &_node;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectorySetpoint;
    std::shared_ptr<px4_ros2::OdometryAttitude> _vehicleAttitude;
    std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicleLocalPosition;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetPoseSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetVelocitySub;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _targetMotionSub;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _targetCovarianceSub;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _targetProcessNoiseSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicleLocalPositionSub;

    std::string _targetPoseTopic;
    std::string _targetVelocityTopic;
    std::string _targetMotionTopic;
    std::string _targetCovarianceTopic;
    std::string _targetProcessNoiseTopic;
    std::string _vehicleLocalPositionTopic;

    TargetModel _targetModel{TargetModel::Cv};
    std::string _paramTargetModel{"cv"};
    float _paramEstimatorMotionTimeoutSec{0.20f};
    float _paramEstimatorPredictionStepSec{0.02f};

    float _paramPidDeadband{0.05f};
    float _paramTargetTimeout{3.0f};

    float _paramTrackingKp{0.9f};
    float _paramTrackingKi{0.01f};
    float _paramTrackingKd{0.0f};
    float _paramTrackingMaxVelocity{10.0f};
    float _paramSlewAcc{10.0f};

    bool _paramYawControlEnabled{true};
    float _paramYawKp{1.5f};
    float _paramYawMaxRateRadS{0.8f};
    float _paramYawSlewAccRadS2{1.2f};
    float _paramYawDeadbandRad{0.03f};

    float _paramTrackingHeight{3.0f};
    float _paramHeightTolerance{0.15f};
    float _paramHeightKp{0.6f};
    float _paramVerticalSlewAcc{0.6f};
    float _paramDescentGateRadius{0.3f};
    float _paramVmin{0.3f};
    float _paramVmax{0.45f};

    bool _paramUsePredictiveError{true};
    float _paramPredictionDtMax{0.75f};
    float _paramControlExtraLeadSec{0.25f};
    float _paramPredictiveAccGain{0.0f};
    float _paramPredictiveAccLpfAlpha{0.4f};
    float _paramPredictiveAccMax{4.0f};

    float _paramVWindN{0.0f};
    float _paramVWindE{0.0f};
    float _paramVWindD{0.0f};

    float _paramPayloadMassKg{0.5f};
    float _paramCd{1.0f};
    float _paramAreaM2{0.01f};
    float _paramRhoAir{1.225f};
    float _paramDropDtSec{0.005f};
    float _paramDropMaxTimeSec{3.0f};

    float _paramReleaseDelaySec{0.20f};
    float _paramReleaseMaxErrorM{0.30f};
    float _paramReleaseMaxSigmaM{0.20f};
    float _paramReleaseMaxRelativeVelocityMps{0.50f};
    float _paramReleaseMaxTargetAgeSec{0.20f};
    float _paramReleaseCovTimeoutSec{0.20f};
    float _paramPayloadSigmaN{0.0f};
    float _paramPayloadSigmaE{0.0f};
    int _paramReleaseConfirmCycles{5};

    DropPred _dropPred{};
    DropPred::DropOutput _dropOutput{};
    DropGate _dropGate{};
    DropGate::Output _gateOutput{};

    State _state{State::Search};
    TargetWorldData _targetWorld{};
    TargetCovData _targetCov{};
    TargetNoiseData _targetNoise{};

    bool _active{false};
    bool _distBottomValid{false};
    float _distBottom{0.0f};

    float _prevVehicleVelX{0.0f};
    float _prevVehicleVelY{0.0f};
    float _vehicleAccXFilt{0.0f};
    float _vehicleAccYFilt{0.0f};
    bool _prevVehicleVelValid{false};

    float _velXIntegral{0.0f};
    float _velYIntegral{0.0f};
    float _prevErrX{0.0f};
    float _prevErrY{0.0f};
    bool _prevErrValid{false};
    float _vxFilt{0.0f};
    float _vyFilt{0.0f};
    float _vzFilt{0.0f};

    float _yawRateSpRadS{0.0f};
};
