#include "DropPred.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace
{
constexpr float kGravity = 9.80665f;
constexpr float kTurnRateEps = 1e-4f;
}

DropPred::TargetOutput DropPred::predictCv(const CvInput &input) const
{
    TargetOutput output{};

    if (!input.valid ||
        !input.positionNE.allFinite() ||
        !input.velocityNE.allFinite() ||
        !input.covariance.allFinite() ||
        !std::isfinite(input.predictionTimeSec) ||
        !std::isfinite(input.qAccN) ||
        !std::isfinite(input.qAccE) ||
        input.qAccN < 0.0f ||
        input.qAccE < 0.0f)
    {
        return output;
    }

    const float t = std::max(input.predictionTimeSec, 0.0f);

    Eigen::Matrix4f transition = Eigen::Matrix4f::Identity();
    transition(0, 2) = t;
    transition(1, 3) = t;

    // Match the discrete acceleration-noise model used by the uploaded KF:
    // Q_axis = q * [[0.25 t^4, 0.5 t^3], [0.5 t^3, t^2]].
    Eigen::Matrix4f processNoise = Eigen::Matrix4f::Zero();
    const float t2 = t * t;
    const float t3 = t2 * t;
    const float t4 = t2 * t2;

    processNoise(0, 0) = 0.25f * input.qAccN * t4;
    processNoise(0, 2) = 0.5f * input.qAccN * t3;
    processNoise(2, 0) = processNoise(0, 2);
    processNoise(2, 2) = input.qAccN * t2;

    processNoise(1, 1) = 0.25f * input.qAccE * t4;
    processNoise(1, 3) = 0.5f * input.qAccE * t3;
    processNoise(3, 1) = processNoise(1, 3);
    processNoise(3, 3) = input.qAccE * t2;

    const Eigen::Matrix4f predictedCovariance =
        transition * input.covariance * transition.transpose() + processNoise;

    output.positionNE = input.positionNE + input.velocityNE * t;
    output.velocityNE = input.velocityNE;
    output.covarianceNE =
        0.5f * (predictedCovariance.topLeftCorner<2, 2>() +
                predictedCovariance.topLeftCorner<2, 2>().transpose());
    output.predictionTimeSec = t;
    output.valid =
        output.positionNE.allFinite() &&
        output.velocityNE.allFinite() &&
        output.covarianceNE.allFinite();
    return output;
}

DropPred::Vector6f DropPred::ctraStep(const Vector6f &state, float dtSec) const
{
    Vector6f next = state;

    const float pN = state(0);
    const float pE = state(1);
    const float speed = state(2);
    const float heading = state(3);
    const float tangentialAcc = state(4);
    const float turnRate = state(5);
    const float dt = std::max(dtSec, 0.0f);

    if (std::abs(turnRate) > kTurnRateEps)
    {
        const float theta = heading + turnRate * dt;
        const float turnRate2 = turnRate * turnRate;

        next(0) =
            pN +
            speed / turnRate * (std::sin(theta) - std::sin(heading)) +
            tangentialAcc *
                (dt * std::sin(theta) / turnRate +
                 (std::cos(theta) - std::cos(heading)) / turnRate2);

        next(1) =
            pE +
            speed / turnRate * (std::cos(heading) - std::cos(theta)) +
            tangentialAcc *
                (-dt * std::cos(theta) / turnRate +
                 (std::sin(theta) - std::sin(heading)) / turnRate2);
    }
    else
    {
        const float distance =
            speed * dt + 0.5f * tangentialAcc * dt * dt;

        next(0) = pN + distance * std::cos(heading);
        next(1) = pE + distance * std::sin(heading);
    }

    next(2) = speed + tangentialAcc * dt;
    next(3) = normalizeAngle(heading + turnRate * dt);
    next(4) = tangentialAcc;
    next(5) = turnRate;
    return next;
}

DropPred::Matrix6f DropPred::ctraJacobian(
    const Vector6f &state,
    float dtSec) const
{
    Matrix6f jacobian = Matrix6f::Zero();
    const std::array<float, 6> epsilon{
        1e-3f,
        1e-3f,
        1e-3f,
        1e-4f,
        1e-3f,
        1e-4f};

    for (int column = 0; column < 6; ++column)
    {
        Vector6f plus = state;
        Vector6f minus = state;
        plus(column) += epsilon[static_cast<std::size_t>(column)];
        minus(column) -= epsilon[static_cast<std::size_t>(column)];

        const Vector6f fPlus = ctraStep(plus, dtSec);
        const Vector6f fMinus = ctraStep(minus, dtSec);
        Vector6f difference = fPlus - fMinus;
        difference(3) = normalizeAngle(fPlus(3) - fMinus(3));

        jacobian.col(column) =
            difference / (2.0f * epsilon[static_cast<std::size_t>(column)]);
    }

    return jacobian;
}

DropPred::TargetOutput DropPred::predictCtra(const CtraInput &input) const
{
    TargetOutput output{};

    if (!input.valid ||
        !input.positionNE.allFinite() ||
        !input.covariance.allFinite() ||
        !std::isfinite(input.speedMps) ||
        !std::isfinite(input.headingRad) ||
        !std::isfinite(input.tangentialAccMps2) ||
        !std::isfinite(input.turnRateRadS) ||
        !std::isfinite(input.predictionTimeSec) ||
        !std::isfinite(input.stepSec) ||
        !std::isfinite(input.qAcc) ||
        !std::isfinite(input.qTurnRate) ||
        input.stepSec <= 0.0f ||
        input.qAcc < 0.0f ||
        input.qTurnRate < 0.0f)
    {
        return output;
    }

    Vector6f state;
    state <<
        input.positionNE.x(),
        input.positionNE.y(),
        input.speedMps,
        normalizeAngle(input.headingRad),
        input.tangentialAccMps2,
        input.turnRateRadS;

    Matrix6f covariance =
        0.5f * (input.covariance + input.covariance.transpose());

    const float predictionTime = std::max(input.predictionTimeSec, 0.0f);
    float propagatedTime = 0.0f;

    while (propagatedTime < predictionTime)
    {
        const float dt =
            std::min(input.stepSec, predictionTime - propagatedTime);

        const Matrix6f transition = ctraJacobian(state, dt);
        Matrix6f processNoise = Matrix6f::Zero();
        processNoise(4, 4) = input.qAcc * dt;
        processNoise(5, 5) = input.qTurnRate * dt;

        covariance =
            transition * covariance * transition.transpose() + processNoise;
        state = ctraStep(state, dt);
        propagatedTime += dt;

        if (!state.allFinite() || !covariance.allFinite())
        {
            return output;
        }
    }

    output.positionNE = state.head<2>();
    output.velocityNE = Eigen::Vector2f(
        state(2) * std::cos(state(3)),
        state(2) * std::sin(state(3)));
    output.covarianceNE =
        0.5f * (covariance.topLeftCorner<2, 2>() +
                covariance.topLeftCorner<2, 2>().transpose());
    output.predictionTimeSec = predictionTime;
    output.valid =
        output.positionNE.allFinite() &&
        output.velocityNE.allFinite() &&
        output.covarianceNE.allFinite();
    return output;
}

Eigen::Vector3f DropPred::dropAcceleration(
    const Eigen::Vector3f &velocityNed,
    const DropInput &input) const
{
    const Eigen::Vector3f vRel = velocityNed - input.vWindNed;
    const float dragK =
        0.5f * input.rhoAir * input.cd * input.areaM2 / input.massKg;

    Eigen::Vector3f accel(0.0f, 0.0f, kGravity);
    accel -= dragK * vRel.norm() * vRel;
    return accel;
}

DropPred::DropOutput DropPred::predictDrop(const DropInput &input) const
{
    DropOutput output{};

    if (!input.valid ||
        !input.velocityNed.allFinite() ||
        !input.vWindNed.allFinite() ||
        !std::isfinite(input.heightM) ||
        !std::isfinite(input.massKg) ||
        !std::isfinite(input.cd) ||
        !std::isfinite(input.areaM2) ||
        !std::isfinite(input.rhoAir) ||
        !std::isfinite(input.dtSec) ||
        !std::isfinite(input.maxTimeSec) ||
        input.heightM <= 0.0f ||
        input.massKg <= 0.0f ||
        input.cd < 0.0f ||
        input.areaM2 < 0.0f ||
        input.rhoAir <= 0.0f ||
        input.dtSec <= 0.0f ||
        input.maxTimeSec <= 0.0f)
    {
        return output;
    }

    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    Eigen::Vector3f velocity = input.velocityNed;
    float timeSec = 0.0f;

    while (timeSec < input.maxTimeSec)
    {
        const Eigen::Vector3f prevPosition = position;
        const Eigen::Vector3f prevVelocity = velocity;
        const float prevTimeSec = timeSec;
        const float dt = std::min(input.dtSec, input.maxTimeSec - timeSec);

        const Eigen::Vector3f accel = dropAcceleration(velocity, input);
        position += velocity * dt;
        velocity += accel * dt;
        timeSec += dt;

        if (!position.allFinite() || !velocity.allFinite())
        {
            return output;
        }

        if (position.z() >= input.heightM)
        {
            const float dz = position.z() - prevPosition.z();
            const float alpha =
                (std::abs(dz) > 1e-6f)
                    ? std::clamp(
                          (input.heightM - prevPosition.z()) / dz,
                          0.0f,
                          1.0f)
                    : 1.0f;

            output.impactOffsetNed =
                prevPosition + alpha * (position - prevPosition);
            output.impactVelocityNed =
                prevVelocity + alpha * (velocity - prevVelocity);
            output.impactTimeSec = prevTimeSec + alpha * dt;
            output.valid =
                output.impactOffsetNed.allFinite() &&
                output.impactVelocityNed.allFinite() &&
                std::isfinite(output.impactTimeSec);
            return output;
        }
    }

    return output;
}

float DropPred::normalizeAngle(float angleRad) const
{
    return std::atan2(std::sin(angleRad), std::cos(angleRad));
}
