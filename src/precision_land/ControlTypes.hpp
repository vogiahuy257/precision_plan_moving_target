#pragma once

#include <Eigen/Core>

namespace precision_land
{
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
} // namespace precision_land
