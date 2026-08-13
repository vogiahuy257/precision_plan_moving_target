#include "DropGate.hpp"

#include <algorithm>
#include <cmath>

DropGate::Output DropGate::update(
    const Input &input,
    const Limits &limits)
{
    Output output{};

    if (!input.valid ||
        !input.errorNE.allFinite() ||
        !input.covarianceNE.allFinite() ||
        !input.relativeVelocityNE.allFinite() ||
        !std::isfinite(input.heightErrorM) ||
        !std::isfinite(input.targetAgeSec))
    {
        reset();
        return output;
    }

    const Eigen::Matrix2f covariance =
        0.5f * (input.covarianceNE + input.covarianceNE.transpose());

    const float a = covariance(0, 0);
    const float b = covariance(0, 1);
    const float d = covariance(1, 1);

    if (a < 0.0f || d < 0.0f)
    {
        reset();
        return output;
    }

    const float discriminant = std::sqrt(
        std::max((a - d) * (a - d) + 4.0f * b * b, 0.0f));
    const float lambdaMax = std::max(0.5f * (a + d + discriminant), 0.0f);

    output.errorM = input.errorNE.norm();
    output.sigmaM = std::sqrt(lambdaMax);
    output.relativeVelocityMps = input.relativeVelocityNE.norm();

    output.errorOk =
        output.errorM <= std::max(limits.maxErrorM, 0.0f);
    output.uncertaintyOk =
        output.sigmaM <= std::max(limits.maxSigmaM, 0.0f);
    output.relativeVelocityOk =
        output.relativeVelocityMps <=
        std::max(limits.maxRelativeVelocityMps, 0.0f);
    output.altitudeOk =
        std::abs(input.heightErrorM) <=
        std::max(limits.maxHeightErrorM, 0.0f);
    output.targetAgeOk =
        input.targetAgeSec >= 0.0f &&
        input.targetAgeSec <= std::max(limits.maxTargetAgeSec, 0.0f);
    output.vehicleStateOk = input.vehicleReady;

    const bool gateOk =
        output.errorOk &&
        output.uncertaintyOk &&
        output.relativeVelocityOk &&
        output.altitudeOk &&
        output.targetAgeOk &&
        output.vehicleStateOk;

    if (gateOk)
    {
        ++_validCycles;
    }
    else
    {
        _validCycles = 0;
    }

    output.release =
        _validCycles >= std::max(limits.confirmCycles, 1);
    return output;
}

void DropGate::reset()
{
    _validCycles = 0;
}
