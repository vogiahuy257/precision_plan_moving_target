#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_command_ack.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <std_msgs/msg/string.hpp>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>

class PrecisionLand : public px4_ros2::ModeBase
{
public:
    explicit PrecisionLand(rclcpp::Node &node);

    void onActivate() override;
    void onDeactivate() override;
    void updateSetpoint(float dt_s) override;

private:
    enum class State
    {
        Search,
        Descend,
        Finished
    };

    enum class DisarmAltitudeSource : uint8_t
    {
        DistBottom,
        LocalPositionZ
    };

    enum class DisarmMode : uint8_t
    {
        Disabled,
        Enabled
    };

    enum class DisarmDecisionStatus : uint8_t
    {
        Idle,
        Disabled,
        Blocked,
        WaitingAck,
        Accepted,
        Rejected
    };

    struct TargetWorldData
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        Eigen::Vector3d velocity{0.0, 0.0, 0.0};

        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        rclcpp::Time velocityTimestamp{0, 0, RCL_ROS_TIME};

        float yawRad{0.0f};
        bool validPose{false};
        bool validVelocity{false};
        bool validYaw{false};
    };

    struct TargetState
    {
        Eigen::Vector3f positionWorld{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f velocityWorld{0.0f, 0.0f, 0.0f};
        bool hasVelocity{false};
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

    struct DisarmInput
    {
        bool distBottomValid{false};
        float distBottom{0.0f};

        bool localPositionZValid{false};
        float localPositionZ{0.0f};

        float lateralError{0.0f};
        float verticalSpeedAbs{0.0f};
        bool landed{false};
    };

    struct DisarmOutput
    {
        bool shouldSendLand{false};
        bool selectedAltitudeValid{false};
        float selectedAltitude{0.0f};
        DisarmDecisionStatus status{DisarmDecisionStatus::Idle};
    };

    struct DebugTiming
    {
        double poseWaitDt{-1.0};
        double velWaitDt{-1.0};
        double controlProcessingDt{-1.0};
        double sendCmdDt{-1.0};
        double totalImageToCmdDt{-1.0};
    };

    struct DebugSample
    {
        double timeSec{0.0};
        std::string state{"Unknown"};

        Eigen::Vector3f dronePos{Eigen::Vector3f::Zero()};
        Eigen::Vector3f droneVel{Eigen::Vector3f::Zero()};

        Eigen::Vector3f targetEst{Eigen::Vector3f::Zero()};
        Eigen::Vector3f targetPred{Eigen::Vector3f::Zero()};
        Eigen::Vector3f targetVel{Eigen::Vector3f::Zero()};

        Eigen::Vector2f errorXY{Eigen::Vector2f::Zero()};
        Eigen::Vector2f futureErrorXY{Eigen::Vector2f::Zero()};
        Eigen::Vector2f pidOutXY{Eigen::Vector2f::Zero()};
        Eigen::Vector2f ffXY{Eigen::Vector2f::Zero()};
        Eigen::Vector3f finalSp{Eigen::Vector3f::Zero()};

        float currentYawRad{0.0f};
        float targetYawRad{0.0f};
        float yawErrorRad{0.0f};
        float yawRateRawRadS{0.0f};
        float yawRateSpRadS{0.0f};
        int yawTurnDirection{0};
        bool yawControlValid{false};

        float altitudeAbs{0.0f};
        float distBottom{-1.0f};

        bool shouldLand{false};
        bool landDetected{false};

        DebugTiming timing{};
    };

private:
    void loadParameters();
    void hover();

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void vehicleCommandAckCallback(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg);

    void handleSearchState(bool targetLost);
    void handleDescendState(float dt_s, bool targetLost);
    void handleFinishedState();

    void resetXyController();
    void resetYawController();
    void resetDisarmLogic();

    bool checkTargetTimeout() const;
    void switchToState(State state);

    float computeLeadTimeSec(float dt_s, const rclcpp::Time &ctrlStartNow) const;
    PredictionInput buildPredictionInput(float dt_s, const rclcpp::Time &ctrlStartNow);
    PredictionOutput predictTarget(const PredictionInput &input) const;

    Eigen::Vector2f estimateVehicleAccelerationXY(float dt_s);
    Eigen::Vector2f clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const;

    XYControllerOutput updateXyController(const XYControllerInput &input);
    YawControllerOutput updateYawController(float dtSec, float targetYawRad, bool targetYawValid);
    float applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const;
    float applyYawSlew(float commandYawRate, float previousYawRate, float slewLimit, float dtSec) const;
    float normalizeAnglePi(float angleRad) const;
    float yawFromPose(const geometry_msgs::msg::Pose &pose) const;

    float computeZVelocityCommand(float vehicleAltitudeAbs, const Eigen::Vector2f &futureErrorXY) const;

    DisarmMode parseDisarmMode(const std::string &value) const;
    DisarmAltitudeSource parseDisarmAltitudeSource(const std::string &value) const;
    float selectDisarmAltitude(const DisarmInput &input, bool &isValid) const;
    bool shouldRequestLand(const DisarmInput &input, float &selectedAltitude, bool &selectedAltitudeValid) const;
    DisarmOutput updateDisarmLogic(const DisarmInput &input);
    bool sendLandCommand();
    void publishVehicleCommand(uint16_t command, float param1, float param2);

    std::string stateName(State state) const;

    void startDebugLogSession();
    void closeDebugLogSession();
    void flushDebugLog();
    void logDebugSample(
        const rclcpp::Time &ctrlStartNow,
        const rclcpp::Time &ctrlEndNow,
        const rclcpp::Time &cmdPubNow,
        const PredictionOutput &predictionOutput,
        const XYControllerInput &xyInput,
        const XYControllerOutput &xyOutput,
        const YawControllerOutput &yawOutput,
        const DisarmOutput &disarmOutput,
        float vz,
        float altitudeNow);
    void fillDebugTimingSample(
        DebugSample &sample,
        const rclcpp::Time &ctrlStartNow,
        const rclcpp::Time &ctrlEndNow,
        const rclcpp::Time &cmdPubNow) const;
    void openDebugLogFileIfNeeded();
    void writeDebugLogHeaderIfNeeded();
    void disableDebugLog();
    std::string makeCurrentTimeString() const;
    std::string buildDebugCsvPath() const;
    std::string debugSampleToCsvLine(const DebugSample &sample) const;

private:
    rclcpp::Node &_node;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectorySetpoint;
    std::shared_ptr<px4_ros2::OdometryAttitude> _vehicleAttitude;
    std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicleLocalPosition;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetPoseSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetVelocitySub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr _vehicleLandDetectedSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicleLocalPosSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleCommandAck>::SharedPtr _vehicleCommandAckSub;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _gimbalSeqPub;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr _vehicleCommandPub;

    std::string _targetPoseTopic;
    std::string _targetVelocityTopic;
    std::string _vehicleLandDetectedTopic;
    std::string _vehicleLocalPositionTopic;
    std::string _gimbalCommandTopic;

    float _paramPidDeadband{0.05f};
    float _paramTargetTimeout{3.0f};

    float _paramDescentKp{0.9f};
    float _paramDescentKi{0.01f};
    float _paramDescentKd{0.0f};
    float _paramDescentMaxVelocity{10.0f};
    float _paramSlewAcc{10.0f};

    bool _paramYawControlEnabled{true};
    float _paramYawKp{1.5f};
    float _paramYawMaxRateRadS{0.8f};
    float _paramYawSlewAccRadS2{1.2f};
    float _paramYawDeadbandRad{0.03f};

    float _paramLandZoneZ{0.5f};
    float _paramDescentVel{0.5f};
    float _paramDescentGateRadius{0.3f};
    float _paramVmin{0.45f};
    float _paramVmax{0.8f};

    bool _paramUsePredictiveError{true};
    float _paramPredictionDtMax{0.75f};
    float _paramControlExtraLeadSec{0.25f};
    float _paramPredictiveAccGain{0.0f};
    float _paramPredictiveAccLpfAlpha{0.4f};
    float _paramPredictiveAccMax{4.0f};

    std::string _paramDisarmMode{"enabled"};
    std::string _paramDisarmAltitudeSource{"dist_bottom"};
    float _paramDisarmHeight{0.06f};
    float _paramDisarmLateralErrorThreshold{0.10f};
    float _paramDisarmVerticalSpeedThreshold{0.15f};
    bool _paramDisarmAllowLandedImmediate{true};

    bool _paramDebugLogger{false};

    DisarmMode _disarmMode{DisarmMode::Enabled};
    DisarmAltitudeSource _disarmAltitudeSource{DisarmAltitudeSource::DistBottom};
    DisarmDecisionStatus _disarmStatus{DisarmDecisionStatus::Idle};
    bool _disarmSent{false};
    bool _waitingLandAck{false};
    rclcpp::Time _landRequestTime{0, 0, RCL_ROS_TIME};

    State _state{State::Search};
    TargetWorldData _targetWorld{};
    rclcpp::Time _targetPoseRxNow{0, 0, RCL_ROS_TIME};
    rclcpp::Time _targetVelRxNow{0, 0, RCL_ROS_TIME};

    bool _searchStarted{false};
    bool _targetLostPrev{true};
    bool _distBottomValid{false};
    float _zDistBottom{0.0f};
    bool _landDetected{false};
    float _approachAltitude{0.0f};

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

    float _yawRateSpRadS{0.0f};

    static constexpr std::size_t kDebugLogFlushBatchSize{100};
    static constexpr const char *kDebugLogDirectory{"precisionland_logs/controller"};

    bool _debugLogEnabled{false};
    bool _debugLogFileOpened{false};
    bool _debugLogHeaderWritten{false};
    bool _debugLogSessionStarted{false};
    std::string _debugLogSessionStamp;
    std::string _debugLogPath;
    std::ofstream _debugLogFile;
    std::vector<std::string> _debugLogBuffer;
};
