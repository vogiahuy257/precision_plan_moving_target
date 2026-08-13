#pragma once

#include <Eigen/Core>

class DropGate
{
public:
    struct Input
    {
        Eigen::Vector2f errorNE{0.0f, 0.0f};
        Eigen::Matrix2f covarianceNE{Eigen::Matrix2f::Zero()};
        Eigen::Vector2f relativeVelocityNE{0.0f, 0.0f};
        float heightErrorM{0.0f};
        float targetAgeSec{0.0f};
        bool vehicleReady{false};
        bool valid{false};
    };

    struct Limits
    {
        float maxErrorM{0.0f};
        float maxSigmaM{0.0f};
        float maxRelativeVelocityMps{0.0f};
        float maxHeightErrorM{0.0f};
        float maxTargetAgeSec{0.0f};
        int confirmCycles{1};
    };

    struct Output
    {
        float errorM{0.0f};
        float sigmaM{0.0f};
        float relativeVelocityMps{0.0f};
        bool errorOk{false};
        bool uncertaintyOk{false};
        bool relativeVelocityOk{false};
        bool altitudeOk{false};
        bool targetAgeOk{false};
        bool vehicleStateOk{false};
        bool release{false};
    };

    Output update(const Input &input, const Limits &limits);
    void reset();

private:
    int _validCycles{0};
};
