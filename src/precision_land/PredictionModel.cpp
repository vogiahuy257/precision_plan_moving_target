#include "PredictionModel.hpp"

namespace precision_land
{
PredictionOutput PredictionModel::predict(const PredictionInput &input) const
{
    PredictionOutput output;

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

        output.vehicleFutureWorld.z() += input.vehicle.velocityWorld.z() * input.leadDtSec;
    }

    output.futureErrorXY.x() = output.targetFutureWorld.x() - output.vehicleFutureWorld.x();
    output.futureErrorXY.y() = output.targetFutureWorld.y() - output.vehicleFutureWorld.y();

    return output;
}
} // namespace precision_land
