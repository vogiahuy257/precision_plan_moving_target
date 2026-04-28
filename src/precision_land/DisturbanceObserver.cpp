#include "DisturbanceObserver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace precision_land
{
void DisturbanceObserver::configure(const DisturbanceObserverParams &params)
{
    try
    {
        if (!std::isfinite(params.tauSec) ||
            !std::isfinite(params.gain) ||
            !std::isfinite(params.maxBias) ||
            !std::isfinite(params.deadband))
        {
            throw std::runtime_error("DisturbanceObserverParams chua NaN/Inf");
        }

        if (params.tauSec <= 0.0f)
        {
            throw std::runtime_error("dob tauSec phai > 0");
        }

        if (params.gain < 0.0f)
        {
            throw std::runtime_error("dob gain phai >= 0");
        }

        if (params.maxBias < 0.0f)
        {
            throw std::runtime_error("dob maxBias phai >= 0");
        }

        if (params.deadband < 0.0f)
        {
            throw std::runtime_error("dob deadband phai >= 0");
        }

        params_ = params;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("DisturbanceObserver::configure loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("DisturbanceObserver::configure gap loi khong xac dinh");
    }
}

void DisturbanceObserver::reset()
{
    disturbanceHatXY_.setZero();
}

DisturbanceObserverOutput DisturbanceObserver::update(const DisturbanceObserverInput &input)
{
    try
    {
        if (!std::isfinite(input.dtSec) ||
            !std::isfinite(input.referenceVelocityXY.x()) ||
            !std::isfinite(input.referenceVelocityXY.y()) ||
            !std::isfinite(input.measuredVelocityXY.x()) ||
            !std::isfinite(input.measuredVelocityXY.y()))
        {
            throw std::runtime_error("DisturbanceObserverInput chua NaN/Inf");
        }

        DisturbanceObserverOutput output{};

        if (!params_.enabled)
        {
            disturbanceHatXY_.setZero();
            return output;
        }

        const float dt = std::max(input.dtSec, 1e-3f);

        if (!input.targetValid)
        {
            const float decay = std::clamp(1.0f - 2.0f * dt, 0.0f, 1.0f);
            disturbanceHatXY_ *= decay;

            output.estimatedDisturbanceXY = disturbanceHatXY_;
            output.compensationXY = params_.gain * disturbanceHatXY_;
            return output;
        }

        Eigen::Vector2f residualXY = input.referenceVelocityXY - input.measuredVelocityXY;
        residualXY = applyVectorDeadband(residualXY, params_.deadband);

        const float alpha = dt / (params_.tauSec + dt);

        disturbanceHatXY_ =
            (1.0f - alpha) * disturbanceHatXY_ +
            alpha * residualXY;

        disturbanceHatXY_ = clampVectorNorm(disturbanceHatXY_, params_.maxBias);

        output.estimatedDisturbanceXY = disturbanceHatXY_;
        output.compensationXY = params_.gain * disturbanceHatXY_;

        if (!std::isfinite(output.estimatedDisturbanceXY.x()) ||
            !std::isfinite(output.estimatedDisturbanceXY.y()) ||
            !std::isfinite(output.compensationXY.x()) ||
            !std::isfinite(output.compensationXY.y()))
        {
            throw std::runtime_error("DisturbanceObserverOutput tinh ra NaN/Inf");
        }

        return output;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("DisturbanceObserver::update loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("DisturbanceObserver::update gap loi khong xac dinh");
    }
}

Eigen::Vector2f DisturbanceObserver::clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const
{
    try
    {
        if (!std::isfinite(value.x()) ||
            !std::isfinite(value.y()) ||
            !std::isfinite(maxNorm))
        {
            throw std::runtime_error("tham so clampVectorNorm chua NaN/Inf");
        }

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
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("DisturbanceObserver::clampVectorNorm loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("DisturbanceObserver::clampVectorNorm gap loi khong xac dinh");
    }
}

Eigen::Vector2f DisturbanceObserver::applyVectorDeadband(const Eigen::Vector2f &value, float deadband) const
{
    try
    {
        if (!std::isfinite(value.x()) ||
            !std::isfinite(value.y()) ||
            !std::isfinite(deadband))
        {
            throw std::runtime_error("tham so applyVectorDeadband chua NaN/Inf");
        }

        const float norm = value.norm();

        if (norm <= deadband || norm < 1e-6f)
        {
            return Eigen::Vector2f::Zero();
        }

        return value * ((norm - deadband) / norm);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("DisturbanceObserver::applyVectorDeadband loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("DisturbanceObserver::applyVectorDeadband gap loi khong xac dinh");
    }
}
} // namespace precision_land