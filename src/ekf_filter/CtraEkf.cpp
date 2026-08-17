#include "CtraEkf.hpp"

#include <Eigen/Cholesky>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace
{
constexpr double kMinVariance = 1e-12;
}

CtraEkf::CtraEkf()
    : CtraEkf(Config{})
{
}

CtraEkf::CtraEkf(const Config &config)
{
    setConfig(config);
    reset();
}

void CtraEkf::setConfig(const Config &config)
{
    if (!std::isfinite(config.qAcc) || config.qAcc < 0.0 ||
        !std::isfinite(config.qTurnRate) || config.qTurnRate < 0.0 ||
        !std::isfinite(config.rPosN) || config.rPosN <= 0.0 ||
        !std::isfinite(config.rPosE) || config.rPosE <= 0.0 ||
        !std::isfinite(config.nisThreshold) || config.nisThreshold <= 0.0 ||
        !std::isfinite(config.turnRateEps) || config.turnRateEps <= 0.0)
    {
        throw std::runtime_error("Invalid CTRA EKF configuration");
    }

    config_ = config;
}

void CtraEkf::reset()
{
    x_.setZero();
    P_.setIdentity();
    initialized_ = false;
}

void CtraEkf::initialize(
    double pN,
    double pE,
    double speed,
    double heading,
    double tangentialAcc,
    double turnRate,
    const Matrix6d &initialCovariance)
{
    Vector6d state;
    state << pN,
        pE,
        speed,
        wrapAngle(heading),
        tangentialAcc,
        turnRate;

    if (!state.allFinite() || !initialCovariance.allFinite())
    {
        throw std::runtime_error("Non-finite EKF initialization");
    }

    x_ = state;
    P_ = 0.5 * (initialCovariance + initialCovariance.transpose());
    normalizeStateRepresentation();
    stabilizeCovariance();
    initialized_ = true;
}

bool CtraEkf::initialized() const
{
    return initialized_;
}

void CtraEkf::predict(double dtSec)
{
    if (!initialized_ || !std::isfinite(dtSec) || dtSec <= 0.0)
    {
        return;
    }

    // Propagate once with the exact timestamp-derived interval.
    // The CTRA state transition is analytic, so there is no fixed predict step.
    predictSingleStep(dtSec);
}

void CtraEkf::predictSingleStep(double dtSec)
{
    const Matrix6d F = transitionJacobian(x_, dtSec);
    const Vector6d nextState = transition(x_, dtSec);

    P_ = F * P_ * F.transpose() + processNoise(dtSec);
    x_ = nextState;
    x_(3) = wrapAngle(x_(3));
    normalizeStateRepresentation();
    stabilizeCovariance();
}

CtraEkf::UpdateResult CtraEkf::correct(const Vector2d &measurementNE)
{
    UpdateResult result{};

    if (!initialized_ || !measurementNE.allFinite())
    {
        return result;
    }

    Matrix26d H = Matrix26d::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;

    Matrix2d R = Matrix2d::Zero();
    R(0, 0) = config_.rPosN;
    R(1, 1) = config_.rPosE;

    const Vector2d innovation = measurementNE - H * x_;
    const Matrix2d S = H * P_ * H.transpose() + R;

    Eigen::LDLT<Matrix2d> ldlt(S);
    if (ldlt.info() != Eigen::Success)
    {
        return result;
    }

    const Vector2d solvedInnovation = ldlt.solve(innovation);
    if (ldlt.info() != Eigen::Success || !solvedInnovation.allFinite())
    {
        return result;
    }

    result.nis = innovation.dot(solvedInnovation);
    if (!std::isfinite(result.nis) || result.nis > config_.nisThreshold)
    {
        return result;
    }

    const Matrix2d SInv = ldlt.solve(Matrix2d::Identity());
    if (ldlt.info() != Eigen::Success || !SInv.allFinite())
    {
        return result;
    }

    const Eigen::Matrix<double, 6, 2> K = P_ * H.transpose() * SInv;
    x_ += K * innovation;
    x_(3) = wrapAngle(x_(3));

    const Matrix6d I = Matrix6d::Identity();
    const Matrix6d IKH = I - K * H;

    // Joseph form keeps P symmetric and numerically positive semi-definite.
    P_ = IKH * P_ * IKH.transpose() + K * R * K.transpose();
    normalizeStateRepresentation();
    stabilizeCovariance();

    result.accepted = true;
    return result;
}

const CtraEkf::Vector6d &CtraEkf::state() const
{
    return x_;
}

const CtraEkf::Matrix6d &CtraEkf::covariance() const
{
    return P_;
}

CtraEkf::Vector6d CtraEkf::transition(
    const Vector6d &state,
    double dtSec) const
{
    Vector6d next = state;

    const double pN = state(0);
    const double pE = state(1);
    const double speed = state(2);
    const double psi = state(3);
    const double acc = state(4);
    const double omega = state(5);
    const double dt = dtSec;

    if (std::abs(omega) > config_.turnRateEps)
    {
        const double theta = psi + omega * dt;
        const double omega2 = omega * omega;

        next(0) =
            pN +
            speed / omega * (std::sin(theta) - std::sin(psi)) +
            acc *
                (dt * std::sin(theta) / omega +
                 (std::cos(theta) - std::cos(psi)) / omega2);

        next(1) =
            pE +
            speed / omega * (std::cos(psi) - std::cos(theta)) +
            acc *
                (-dt * std::cos(theta) / omega +
                 (std::sin(theta) - std::sin(psi)) / omega2);
    }
    else
    {
        const double distance = speed * dt + 0.5 * acc * dt * dt;
        next(0) = pN + distance * std::cos(psi);
        next(1) = pE + distance * std::sin(psi);
    }

    next(2) = speed + acc * dt;
    next(3) = wrapAngle(psi + omega * dt);
    next(4) = acc;
    next(5) = omega;
    return next;
}

CtraEkf::Matrix6d CtraEkf::transitionJacobian(
    const Vector6d &state,
    double dtSec) const
{
    const double speed = state(2);
    const double psi = state(3);
    const double acc = state(4);
    const double omega = state(5);
    const double dt = dtSec;

    Matrix6d F = Matrix6d::Identity();
    F(2, 4) = dt;
    F(3, 5) = dt;

    if (std::abs(omega) > config_.turnRateEps)
    {
        const double theta = psi + omega * dt;
        const double sTheta = std::sin(theta);
        const double cTheta = std::cos(theta);
        const double sPsi = std::sin(psi);
        const double cPsi = std::cos(psi);
        const double omega2 = omega * omega;
        const double omega3 = omega2 * omega;

        const double dSin = sTheta - sPsi;
        const double dCos = cPsi - cTheta;

        F(0, 2) = dSin / omega;
        F(1, 2) = dCos / omega;

        F(0, 3) =
            speed / omega * (cTheta - cPsi) +
            acc *
                (dt * cTheta / omega +
                 (-sTheta + sPsi) / omega2);

        F(1, 3) =
            speed / omega * (-sPsi + sTheta) +
            acc *
                (dt * sTheta / omega +
                 (cTheta - cPsi) / omega2);

        F(0, 4) =
            dt * sTheta / omega +
            (cTheta - cPsi) / omega2;

        F(1, 4) =
            -dt * cTheta / omega +
            (sTheta - sPsi) / omega2;

        F(0, 5) =
            speed *
                (dt * cTheta / omega - dSin / omega2) +
            acc *
                (dt * dt * cTheta / omega -
                 2.0 * dt * sTheta / omega2 -
                 2.0 * (cTheta - cPsi) / omega3);

        F(1, 5) =
            speed *
                (dt * sTheta / omega - dCos / omega2) +
            acc *
                (dt * dt * sTheta / omega +
                 2.0 * dt * cTheta / omega2 -
                 2.0 * (sTheta - sPsi) / omega3);
    }
    else
    {
        const double distance = speed * dt + 0.5 * acc * dt * dt;
        const double turnSensitivity =
            0.5 * speed * dt * dt +
            (1.0 / 3.0) * acc * dt * dt * dt;

        F(0, 2) = dt * std::cos(psi);
        F(1, 2) = dt * std::sin(psi);

        F(0, 3) = -distance * std::sin(psi);
        F(1, 3) = distance * std::cos(psi);

        F(0, 4) = 0.5 * dt * dt * std::cos(psi);
        F(1, 4) = 0.5 * dt * dt * std::sin(psi);

        // Smooth omega -> 0 limit of the curved CTRA Jacobian.
        F(0, 5) = -turnSensitivity * std::sin(psi);
        F(1, 5) = turnSensitivity * std::cos(psi);
    }

    return F;
}

CtraEkf::Matrix6d CtraEkf::processNoise(double dtSec) const
{
    Matrix6d Q = Matrix6d::Zero();
    const double dt = dtSec;

    // Paper model: a and omega are slowly varying random walks.
    Q(4, 4) = config_.qAcc * dt;
    Q(5, 5) = config_.qTurnRate * dt;
    return Q;
}

void CtraEkf::normalizeStateRepresentation()
{
    if (x_(2) >= 0.0)
    {
        return;
    }

    x_(2) = -x_(2);
    x_(3) = wrapAngle(x_(3) + std::acos(-1.0));
    x_(4) = -x_(4);

    Matrix6d J = Matrix6d::Identity();
    J(2, 2) = -1.0;
    J(4, 4) = -1.0;
    P_ = J * P_ * J.transpose();
}

void CtraEkf::stabilizeCovariance()
{
    P_ = 0.5 * (P_ + P_.transpose());
    for (int i = 0; i < 6; ++i)
    {
        if (!std::isfinite(P_(i, i)) || P_(i, i) < kMinVariance)
        {
            P_(i, i) = kMinVariance;
        }
    }
}

double CtraEkf::wrapAngle(double angleRad)
{
    return std::atan2(std::sin(angleRad), std::cos(angleRad));
}
