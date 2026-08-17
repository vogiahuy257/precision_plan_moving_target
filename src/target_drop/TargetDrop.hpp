#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>

#include "DropGate.hpp"
#include "DropPred.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <std_msgs/msg/bool.hpp>

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

    struct TargetData
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        Eigen::Vector3d velocity{0.0, 0.0, 0.0};

        rclcpp::Time poseTime{0, 0, RCL_ROS_TIME};
        rclcpp::Time velocityTime{0, 0, RCL_ROS_TIME};

        float headingRad{0.0f};
        float tangentialAccMps2{0.0f};
        float turnRateRadS{0.0f};

        bool active{false};
        bool poseValid{false};
        bool velocityValid{false};
        bool headingValid{false};
        bool motionValid{false};
    };

    struct TargetCovariance
    {
        Eigen::Matrix4f cv{Eigen::Matrix4f::Zero()};
        DropPred::Matrix6f ctra{DropPred::Matrix6f::Zero()};
        bool valid{false};
    };

    struct TargetNoise
    {
        float primary{0.0f};
        float secondary{0.0f};
        bool valid{false};
    };

    struct ReleasePlan
    {
        Eigen::Vector2f desiredReleaseXY{0.0f, 0.0f};
        Eigen::Vector2f errorXY{0.0f, 0.0f};
        Eigen::Vector2f feedforwardVelocityXY{0.0f, 0.0f};
        Eigen::Matrix2f covarianceXY{Eigen::Matrix2f::Zero()};

        // Exact elapsed time from the estimator measurement timestamp
        // to the current control calculation. This is measured, never clamped.
        float measurementDtSec{0.0f};
        bool valid{false};
    };

    void loadParameters();
    void hover();

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetMotionCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void targetCovarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void targetProcessNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void targetStateCallback(const std_msgs::msg::String::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    void handleSearch();
    void handleTrack(float dtSec);
    void switchState(State state);

    ReleasePlan buildReleasePlan(const rclcpp::Time &controlTime) const;
    DropPred::TargetOutput predictTarget(float predictionTimeSec) const;
    DropPred::DropOutput predictPayload(float releaseHeightM) const;

    Eigen::Vector2f updateXyController(
        const Eigen::Vector2f &releaseErrorXY,
        const Eigen::Vector2f &feedforwardVelocityXY,
        float dtSec);

    float updateZController(
        float distanceBottom,
        const Eigen::Vector2f &releaseErrorXY,
        float dtSec);

    void updateReleaseGate(const ReleasePlan &plan);
    void resetControllers();
    void resetReleaseGate();

    Eigen::Vector2f clampNorm(const Eigen::Vector2f &value, float maxNorm) const;
    float applySlew(float command, float previous, float accelLimit, float dtSec) const;
    float headingFromPose(const geometry_msgs::msg::Pose &pose) const;

private:
    rclcpp::Node &_node;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectorySetpoint;
    std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicleLocalPosition;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetPoseSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetVelocitySub;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _targetMotionSub;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _targetCovarianceSub;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr _targetProcessNoiseSub;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _targetStateSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicleLocalPositionSub;

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr _loggerEnablePub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr _controlErrorPub;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr _controlOutputPub;

    std::string _targetPoseTopic;
    std::string _targetVelocityTopic;
    std::string _targetMotionTopic;
    std::string _targetCovarianceTopic;
    std::string _targetProcessNoiseTopic;
    std::string _targetStateTopic;
    std::string _vehicleLocalPositionTopic;

    TargetModel _targetModel{TargetModel::Cv};
    std::string _paramTargetModel{"kf"};

    float _paramKp{1.0f};
    float _paramKi{0.0f};
    float _paramKd{0.0f};
    float _paramDeadbandM{0.08f};
    float _paramMaxVelocityMps{10.0f};
    float _paramSlewAccMps2{0.88f};

    float _paramReleaseHeightM{3.0f};
    float _paramHeightToleranceM{0.15f};
    float _paramHeightKp{0.6f};
    float _paramVerticalSlewAccMps2{0.6f};
    float _paramDescentGateRadiusM{0.30f};
    float _paramDescentMinMps{0.30f};
    float _paramDescentMaxMps{0.45f};

    Eigen::Vector3f _paramWindXyz{0.0f, 0.0f, 0.0f};
    float _paramPayloadMassKg{0.5f};
    float _paramCd{1.0f};
    float _paramAreaM2{0.01f};
    float _paramRhoAir{1.225f};

    // Numerical integration resolution for the nonlinear drag ODE.
    // This is not a runtime delay or timeout.
    float _paramDropIntegrationStepSec{0.005f};

    float _paramReleaseMaxErrorM{0.30f};
    float _paramReleaseMaxSigmaM{0.20f};
    float _paramPayloadSigmaX{0.0f};
    float _paramPayloadSigmaY{0.0f};

    DropPred _dropPred{};
    DropGate _dropGate{};
    DropGate::Output _gateOutput{};

    State _state{State::Search};
    TargetData _target{};
    TargetCovariance _targetCov{};
    TargetNoise _targetNoise{};

    bool _active{false};
    bool _distBottomValid{false};
    float _distBottom{0.0f};

    Eigen::Vector2f _integralXY{0.0f, 0.0f};
    Eigen::Vector2f _previousErrorXY{0.0f, 0.0f};
    Eigen::Vector2f _velocitySetpointXY{0.0f, 0.0f};
    float _verticalVelocitySetpoint{0.0f};
    bool _previousErrorValid{false};
};
