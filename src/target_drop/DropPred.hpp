#pragma once

#include <Eigen/Core>

class DropPred
{
public:
    using Matrix6f = Eigen::Matrix<float, 6, 6>;
    using Vector6f = Eigen::Matrix<float, 6, 1>;

    struct CvInput
    {
        Eigen::Vector2f positionXY{0.0f, 0.0f};
        Eigen::Vector2f velocityXY{0.0f, 0.0f};
        Eigen::Matrix4f covariance{Eigen::Matrix4f::Zero()};
        float predictionTimeSec{0.0f};
        float qAccX{0.0f};
        float qAccY{0.0f};
        bool valid{false};
    };

    struct CtraInput
    {
        Eigen::Vector2f positionXY{0.0f, 0.0f};
        float speedMps{0.0f};
        float headingRad{0.0f};
        float tangentialAccMps2{0.0f};
        float turnRateRadS{0.0f};
        Matrix6f covariance{Matrix6f::Zero()};
        float predictionTimeSec{0.0f};
        float qAcc{0.0f};
        float qTurnRate{0.0f};
        bool valid{false};
    };

    struct TargetOutput
    {
        Eigen::Vector2f positionXY{0.0f, 0.0f};
        Eigen::Vector2f velocityXY{0.0f, 0.0f};
        Eigen::Matrix2f covarianceXY{Eigen::Matrix2f::Zero()};
        float predictionTimeSec{0.0f};
        bool valid{false};
    };

    struct DropInput
    {
        Eigen::Vector3f velocityNed{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f vWindXyz{0.0f, 0.0f, 0.0f};
        float heightM{0.0f};
        float massKg{0.0f};
        float cd{0.0f};
        float areaM2{0.0f};
        float rhoAir{0.0f};

        // Numerical integration resolution of the nonlinear payload ODE.
        // It is not a delay/timeout and does not cap the predicted fall time.
        float integrationStepSec{0.005f};
        bool valid{false};
    };

    struct DropOutput
    {
        Eigen::Vector3f impactOffsetNed{0.0f, 0.0f, 0.0f};
        Eigen::Vector3f impactVelocityNed{0.0f, 0.0f, 0.0f};
        float impactTimeSec{0.0f};
        bool valid{false};
    };

    TargetOutput predictCv(const CvInput &input) const;
    TargetOutput predictCtra(const CtraInput &input) const;
    DropOutput predictDrop(const DropInput &input) const;

private:
    Eigen::Vector3f dropAcceleration(
        const Eigen::Vector3f &velocityNed,
        const DropInput &input) const;

    Vector6f ctraStep(const Vector6f &state, float dtSec) const;
    Matrix6f ctraJacobian(const Vector6f &state, float dtSec) const;
    float normalizeAngle(float angleRad) const;
};
