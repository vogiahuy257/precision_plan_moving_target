#include "PredictionModel.hpp"

#include <cmath>
#include <stdexcept>

namespace precision_land
{
PredictionOutput PredictionModel::predict(const PredictionInput &input) const
{
    try
    {
        if (!std::isfinite(input.leadDtSec) ||
            !std::isfinite(input.predictiveAccGain))
        {
            throw std::runtime_error("leadDtSec hoac predictiveAccGain khong hop le");
        }

        if (!std::isfinite(input.vehicle.positionWorld.x()) ||
            !std::isfinite(input.vehicle.positionWorld.y()) ||
            !std::isfinite(input.vehicle.positionWorld.z()) ||
            !std::isfinite(input.vehicle.velocityWorld.x()) ||
            !std::isfinite(input.vehicle.velocityWorld.y()) ||
            !std::isfinite(input.vehicle.velocityWorld.z()) ||
            !std::isfinite(input.vehicle.accelerationXY.x()) ||
            !std::isfinite(input.vehicle.accelerationXY.y()))
        {
            throw std::runtime_error("du lieu vehicle trong PredictionInput chua NaN/Inf");
        }

        if (!std::isfinite(input.target.positionWorld.x()) ||
            !std::isfinite(input.target.positionWorld.y()) ||
            !std::isfinite(input.target.positionWorld.z()))
        {
            throw std::runtime_error("du lieu target position trong PredictionInput chua NaN/Inf");
        }

        if (input.target.hasVelocity)
        {
            if (!std::isfinite(input.target.velocityWorld.x()) ||
                !std::isfinite(input.target.velocityWorld.y()) ||
                !std::isfinite(input.target.velocityWorld.z()))
            {
                throw std::runtime_error("du lieu target velocity trong PredictionInput chua NaN/Inf");
            }
        }

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
                0.5f * input.predictiveAccGain * input.vehicle.accelerationXY.x() * input.leadDtSec * input.leadDtSec;

            output.vehicleFutureWorld.y() +=
                input.vehicle.velocityWorld.y() * input.leadDtSec +
                0.5f * input.predictiveAccGain * input.vehicle.accelerationXY.y() * input.leadDtSec * input.leadDtSec;

            output.vehicleFutureWorld.z() +=
                input.vehicle.velocityWorld.z() * input.leadDtSec;
        }

        output.futureErrorXY.x() = output.targetFutureWorld.x() - output.vehicleFutureWorld.x();
        output.futureErrorXY.y() = output.targetFutureWorld.y() - output.vehicleFutureWorld.y();

        if (!std::isfinite(output.targetFutureWorld.x()) ||
            !std::isfinite(output.targetFutureWorld.y()) ||
            !std::isfinite(output.targetFutureWorld.z()) ||
            !std::isfinite(output.vehicleFutureWorld.x()) ||
            !std::isfinite(output.vehicleFutureWorld.y()) ||
            !std::isfinite(output.vehicleFutureWorld.z()) ||
            !std::isfinite(output.futureErrorXY.x()) ||
            !std::isfinite(output.futureErrorXY.y()))
        {
            throw std::runtime_error("PredictionOutput tinh ra chua NaN/Inf");
        }

        return output;
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error(std::string("PredictionModel::predict loi: ") + e.what());
    }
    catch (...)
    {
        throw std::runtime_error("PredictionModel::predict gap loi khong xac dinh");
    }
}
} // namespace precision_land