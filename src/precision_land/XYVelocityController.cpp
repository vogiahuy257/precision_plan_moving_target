#include "XYVelocityController.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace precision_land
{
void XYVelocityController::configure(const XYControllerParams &params)
{
    try
    {
        if (!std::isfinite(params.kp) ||
            !std::isfinite(params.ki) ||
            !std::isfinite(params.kd) ||
            !std::isfinite(params.deadband) ||
            !std::isfinite(params.maxVelocity) ||
            !std::isfinite(params.slewAcc))
        {
            throw std::runtime_error("XYControllerParams chua NaN/Inf");
        }

        if (params.deadband < 0.0f)
        {
            throw std::runtime_error("deadband phai >= 0");
        }

        if (params.maxVelocity < 0.0f)
        {
            throw std::runtime_error("maxVelocity phai >= 0");
        }

        if (params.slewAcc < 0.0f)
        {
            throw std::runtime_error("slewAcc phai >= 0");
        }

        params_ = params;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("XYVelocityController::configure loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("XYVelocityController::configure gap loi khong xac dinh");
    }
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
    try
    {
        if (!std::isfinite(input.dtSec) ||
            !std::isfinite(input.futureErrorXY.x()) ||
            !std::isfinite(input.futureErrorXY.y()) ||
            !std::isfinite(input.targetVelocityXY.x()) ||
            !std::isfinite(input.targetVelocityXY.y()))
        {
            throw std::runtime_error("XYControllerInput chua NaN/Inf");
        }

        XYControllerOutput output{};

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

        if (!std::isfinite(output.feedbackXY.x()) ||
            !std::isfinite(output.feedbackXY.y()) ||
            !std::isfinite(output.commandRawXY.x()) ||
            !std::isfinite(output.commandRawXY.y()) ||
            !std::isfinite(output.velocitySpXY.x()) ||
            !std::isfinite(output.velocitySpXY.y()))
        {
            throw std::runtime_error("XYControllerOutput tinh ra chua NaN/Inf");
        }

        return output;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("XYVelocityController::update loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("XYVelocityController::update gap loi khong xac dinh");
    }
}

float XYVelocityController::applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const
{
    try
    {
        if (!std::isfinite(commandVelocity) ||
            !std::isfinite(previousVelocity) ||
            !std::isfinite(accelLimit) ||
            !std::isfinite(dtSec))
        {
            throw std::runtime_error("tham so applySlew chua NaN/Inf");
        }

        const float dt = std::max(dtSec, 1e-3f);
        const float maxDeltaVelocity = accelLimit * dt;
        const float deltaVelocity = std::clamp(commandVelocity - previousVelocity, -maxDeltaVelocity, maxDeltaVelocity);

        const float outputVelocity = previousVelocity + deltaVelocity;

        if (!std::isfinite(outputVelocity))
        {
            throw std::runtime_error("applySlew tinh ra output khong hop le");
        }

        return outputVelocity;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("XYVelocityController::applySlew loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("XYVelocityController::applySlew gap loi khong xac dinh");
    }
}
} // namespace precision_land