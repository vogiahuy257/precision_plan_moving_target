#pragma once

#include <Eigen/Core>

namespace precision_land
{
//==== Common types ====
struct TargetState
{
    Eigen::Vector3f positionWorld{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f velocityWorld{0.0f, 0.0f, 0.0f};
    bool hasVelocity{false};
};
//==== Prediction related types ====
struct VehicleState
{
    Eigen::Vector3f positionWorld{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f velocityWorld{0.0f, 0.0f, 0.0f};
    Eigen::Vector2f accelerationXY{0.0f, 0.0f};
};
// ==== Prediction related types ====
struct PredictionInput
{
    TargetState target;
    VehicleState vehicle;
    float leadDtSec{0.0f};
    float predictiveAccGain{0.0f};
};

struct PredictionOutput
{
    Eigen::Vector3f targetFutureWorld{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f vehicleFutureWorld{0.0f, 0.0f, 0.0f};
    Eigen::Vector2f futureErrorXY{0.0f, 0.0f};
};

// ==== XY control related types ====
struct XYControllerParams
{
    float kp{0.0f};
    float ki{0.0f};
    float kd{0.0f};
    float deadband{0.0f};
    float maxVelocity{0.0f};
    float slewAcc{0.0f};
};

struct XYControllerInput
{
    Eigen::Vector2f futureErrorXY{0.0f, 0.0f};
    Eigen::Vector2f targetVelocityXY{0.0f, 0.0f};
    bool useTargetFeedforward{false};
    float dtSec{0.0f};
};

struct XYControllerOutput
{
    Eigen::Vector2f velocitySpXY{0.0f, 0.0f};
    Eigen::Vector2f feedbackXY{0.0f, 0.0f};
    Eigen::Vector2f commandRawXY{0.0f, 0.0f};
};

// ==== Z control and disarm related types ====
struct ZControllerParams
{
    float landZoneZ{0.10f};
    float descentGateRadius{0.30f};
    float vmin{0.08f};
    float vmax{0.25f};
    float descentVel{0.12f};

    // Ngưỡng quyết định disarm an toàn
    float disarmHeight{0.06f};
};

struct ZControllerInput
{
    float vehicleAltitudeAbs{0.0f};
    Eigen::Vector2f futureErrorXY{0.0f, 0.0f};
};

struct ZControllerOutput
{
    float vzCommand{0.0f};
    bool shouldDisarm{false};
};

// ==== Disarm logic related types ====
enum class DisarmAltitudeSource : uint8_t
{
    DistBottom,
    LocalPositionZ
};

enum class DisarmMode : uint8_t
{
    Disabled,   // không dùng disarm chủ động
    Enabled     // cho phép disarm chủ động
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

struct DisarmControllerParams
{
    DisarmMode mode = DisarmMode::Enabled;
    DisarmAltitudeSource altitudeSource = DisarmAltitudeSource::DistBottom;
    float disarmHeight = 0.06f;
    float lateralErrorThreshold = 0.10f;
    float verticalSpeedThreshold = 0.15f;
    bool allowLandedImmediateDisarm = true;
};

struct DisarmControllerInput
{
    bool distBottomValid = false;
    float distBottom = 0.0f;

    bool localPositionZValid = false;
    float localPositionZ = 0.0f;

    float lateralError = 0.0f;
    float verticalSpeedAbs = 0.0f;
    bool landed = false;
};

struct DisarmControllerOutput
{
    bool shouldSendDisarm = false;
    bool selectedAltitudeValid = false;
    float selectedAltitude = 0.0f;
    DisarmDecisionStatus status = DisarmDecisionStatus::Idle;
};

// ==== Debug logging related types ====
struct PrecisionLandTimingDebug
{
    double poseWaitDt = -1.0;
    double velWaitDt = -1.0;
    double controlProcessingDt = -1.0;
    double sendCmdDt = -1.0;
    double totalImageToCmdDt = -1.0;
};

struct PrecisionLandDebugSample
{
    double timeSec = 0.0;
    std::string state = "Unknown";

    Eigen::Vector3f dronePos = Eigen::Vector3f::Zero();
    Eigen::Vector3f droneVel = Eigen::Vector3f::Zero();

    Eigen::Vector3f targetRaw = Eigen::Vector3f::Zero();
    Eigen::Vector3f targetEst = Eigen::Vector3f::Zero();
    Eigen::Vector3f targetPred = Eigen::Vector3f::Zero();
    Eigen::Vector3f targetVel = Eigen::Vector3f::Zero();

    Eigen::Vector2f errorXY = Eigen::Vector2f::Zero();
    Eigen::Vector2f futureErrorXY = Eigen::Vector2f::Zero();

    Eigen::Vector2f pidOutXY = Eigen::Vector2f::Zero();
    Eigen::Vector2f ffXY = Eigen::Vector2f::Zero();

    Eigen::Vector3f finalSp = Eigen::Vector3f::Zero();

    float altitudeAbs = 0.0f;
    float distBottom = -1.0f;

    bool shouldDisarm = false;
    bool landDetected = false;

    PrecisionLandTimingDebug timing{};
};

} // namespace precision_land
