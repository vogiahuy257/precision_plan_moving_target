#pragma once

#include <Eigen/Core>

class CtraEkf
{
public:
    using Vector6d = Eigen::Matrix<double, 6, 1>;
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Vector2d = Eigen::Matrix<double, 2, 1>;
    using Matrix2d = Eigen::Matrix<double, 2, 2>;
    using Matrix26d = Eigen::Matrix<double, 2, 6>;

    struct Config
    {
        double qAcc{0.20};
        double qTurnRate{0.20};
        double rPosN{0.008};
        double rPosE{0.008};
        double nisThreshold{9.21};
        double turnRateEps{1e-3};
    };

    struct UpdateResult
    {
        bool accepted{false};
        double nis{0.0};
    };

    CtraEkf();
    explicit CtraEkf(const Config &config);

    void setConfig(const Config &config);
    void reset();

    void initialize(
        double pN,
        double pE,
        double speed,
        double heading,
        double tangentialAcc,
        double turnRate,
        const Matrix6d &initialCovariance);

    bool initialized() const;

    void predict(double dtSec);
    UpdateResult correct(const Vector2d &measurementNE);

    const Vector6d &state() const;
    const Matrix6d &covariance() const;

    Vector6d transition(const Vector6d &state, double dtSec) const;
    Matrix6d transitionJacobian(const Vector6d &state, double dtSec) const;

    static double wrapAngle(double angleRad);

private:
    void predictSingleStep(double dtSec);
    Matrix6d processNoise(double dtSec) const;
    void normalizeStateRepresentation();
    void stabilizeCovariance();

    Config config_{};
    Vector6d x_{Vector6d::Zero()};
    Matrix6d P_{Matrix6d::Identity()};
    bool initialized_{false};
};
