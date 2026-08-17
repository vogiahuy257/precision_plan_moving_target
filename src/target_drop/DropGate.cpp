#include "DropGate.hpp"

#include <algorithm>
#include <cmath>

DropGate::Output DropGate::update(const Input &input, const Limits &limits) const
{
    Output output{};

    if (!input.valid ||
        !input.releaseErrorXY.allFinite() ||
        !input.covarianceXY.allFinite() ||
        !std::isfinite(input.heightErrorM))
    {
        return output;
    }

    const Eigen::Matrix2f covariance =
        0.5f * (input.covarianceXY + input.covarianceXY.transpose());

    const float a = covariance(0, 0);
    const float b = covariance(0, 1);
    const float d = covariance(1, 1);

    if (a < 0.0f || d < 0.0f)
    {
        return output;
    }

    const float discriminant = std::sqrt(
        std::max((a - d) * (a - d) + 4.0f * b * b, 0.0f));
    const float lambdaMax = std::max(0.5f * (a + d + discriminant), 0.0f);

    output.releaseErrorM = input.releaseErrorXY.norm();
    output.sigmaM = std::sqrt(lambdaMax);

    output.errorOk =
        output.releaseErrorM <= std::max(limits.maxReleaseErrorM, 0.0f);
    output.uncertaintyOk =
        output.sigmaM <= std::max(limits.maxSigmaM, 0.0f);
    output.altitudeOk =
        std::abs(input.heightErrorM) <= std::max(limits.maxHeightErrorM, 0.0f);
    output.vehicleStateOk = input.vehicleReady;

    output.release =
        output.errorOk &&
        output.uncertaintyOk &&
        output.altitudeOk &&
        output.vehicleStateOk;

    return output;
}
