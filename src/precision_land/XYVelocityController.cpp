#include "XYVelocityController.hpp"

#include <algorithm>
#include <cmath>

namespace precision_land
{
void XYVelocityController::configure(const XYControllerParams &params)
{
    params_ = params;
}

void XYVelocityController::reset()
{
    velXIntegral_ = 0.0f;
    velYIntegral_ = 0.0f;
    prevErrX_ = 0.0f;
    prevErrY_ = 0.0f;
    prevErrValid_ = false;
    vxFilt_ = 0.0f;
    vyFilt_ = 0.0f;
}

XYControllerOutput XYVelocityController::update(const XYControllerInput &input)
{
    XYControllerOutput output;

    const float dt = std::max(input.dtSec, 1e-3f);
    const float errX = input.futureErrorXY.x();
    const float errY = input.futureErrorXY.y();

    const float xp = params_.kp * errX;
    const float yp = params_.kp * errY;

    if (std::abs(errX) > params_.deadband)
    {
        velXIntegral_ += errX * dt;
    }

    if (std::abs(errY) > params_.deadband)
    {
        velYIntegral_ += errY * dt;
    }

    float xi = 0.0f;
    float yi = 0.0f;
    if (params_.ki > 1e-6f)
    {
        const float maxIntegral = 0.15f * params_.maxVelocity / params_.ki;
        velXIntegral_ = std::clamp(velXIntegral_, -maxIntegral, maxIntegral);
        velYIntegral_ = std::clamp(velYIntegral_, -maxIntegral, maxIntegral);
        xi = params_.ki * velXIntegral_;
        yi = params_.ki * velYIntegral_;
    }

    float xd = 0.0f;
    float yd = 0.0f;
    if (params_.kd > 1e-6f && prevErrValid_)
    {
        xd = params_.kd * (errX - prevErrX_) / dt;
        yd = params_.kd * (errY - prevErrY_) / dt;
    }

    prevErrX_ = errX;
    prevErrY_ = errY;
    prevErrValid_ = true;

    output.feedbackXY.x() = std::clamp(xp + xi + xd, -params_.maxVelocity, params_.maxVelocity);
    output.feedbackXY.y() = std::clamp(yp + yi + yd, -params_.maxVelocity, params_.maxVelocity);

    output.commandRawXY = output.feedbackXY;
    if (input.useTargetFeedforward)
    {
        output.commandRawXY += input.targetVelocityXY;
    }

    output.commandRawXY.x() = std::clamp(output.commandRawXY.x(), -params_.maxVelocity, params_.maxVelocity);
    output.commandRawXY.y() = std::clamp(output.commandRawXY.y(), -params_.maxVelocity, params_.maxVelocity);

    vxFilt_ = applySlew(output.commandRawXY.x(), vxFilt_, params_.slewAcc, dt);
    vyFilt_ = applySlew(output.commandRawXY.y(), vyFilt_, params_.slewAcc, dt);

    output.velocitySpXY.x() = std::clamp(vxFilt_, -params_.maxVelocity, params_.maxVelocity);
    output.velocitySpXY.y() = std::clamp(vyFilt_, -params_.maxVelocity, params_.maxVelocity);

    return output;
}

float XYVelocityController::applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const
{
    const float dt = std::max(dtSec, 1e-3f);
    const float maxDeltaVelocity = accelLimit * dt;
    const float deltaVelocity = std::clamp(commandVelocity - previousVelocity, -maxDeltaVelocity, maxDeltaVelocity);
    return previousVelocity + deltaVelocity;
}
} // namespace precision_land
