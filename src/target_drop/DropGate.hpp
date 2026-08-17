#pragma once

#include <Eigen/Core>

class DropGate
{
public:
    struct Input
    {
        Eigen::Vector2f releaseErrorXY{0.0f, 0.0f};
        Eigen::Matrix2f covarianceXY{Eigen::Matrix2f::Zero()};
        float heightErrorM{0.0f};
        bool vehicleReady{false};
        bool valid{false};
    };

    struct Limits
    {
        float maxReleaseErrorM{0.0f};
        float maxSigmaM{0.0f};
        float maxHeightErrorM{0.0f};
    };

    struct Output
    {
        float releaseErrorM{0.0f};
        float sigmaM{0.0f};
        bool errorOk{false};
        bool uncertaintyOk{false};
        bool altitudeOk{false};
        bool vehicleStateOk{false};
        bool release{false};
    };

    Output update(const Input &input, const Limits &limits) const;
};
