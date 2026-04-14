#include "FrameTransformer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace frame_transform
{
namespace
{
constexpr double kDegToRad = M_PI / 180.0;

std::string toLowerCopy(const std::string &value)
{
    std::string lower = value;
    std::transform(
        lower.begin(),
        lower.end(),
        lower.begin(),
        [](unsigned char c)
        {
            return static_cast<char>(std::tolower(c));
        });
    return lower;
}
} // namespace

FrameTransformer::FrameTransformer()
{
    config_ = makeBellyFixedCameraConfig(Eigen::Vector3d(0.0, 0.0, -0.1));
}

void FrameTransformer::setConfig(const Config &config)
{
    config_ = config;
}

void FrameTransformer::setVehicleState(const VehicleState &vehicleState)
{
    vehicleState_ = vehicleState;

    if (vehicleState_.worldFromBody.norm() > 1e-9)
    {
        vehicleState_.worldFromBody.normalize();
    }
    else
    {
        vehicleState_.worldFromBody.setIdentity();
    }
}

void FrameTransformer::setBodyFromMountQuaternion(const Eigen::Quaterniond &bodyFromMount)
{
    bodyFromMount_ = bodyFromMount;

    if (bodyFromMount_.norm() > 1e-9)
    {
        bodyFromMount_.normalize();
    }
    else
    {
        bodyFromMount_.setIdentity();
    }
}

void FrameTransformer::setBodyFromMountEulerDeg(double yawDeg, double pitchDeg, double rollDeg)
{
    const double yawRad = yawDeg * kDegToRad;
    const double pitchRad = pitchDeg * kDegToRad;
    const double rollRad = rollDeg * kDegToRad;

    const Eigen::Quaterniond qBodyFromMount =
        Eigen::AngleAxisd(yawRad, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(pitchRad, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(rollRad, Eigen::Vector3d::UnitX());

    setBodyFromMountQuaternion(qBodyFromMount);
}

Eigen::Vector3d FrameTransformer::opticalPositionToWorld(const Eigen::Vector3d &opticalPosition) const
{
    const Eigen::Matrix3d bodyFromMountMatrix = bodyFromMountQuaternion().toRotationMatrix();
    const Eigen::Vector3d mountPosition = config_.opticalToMountRotation * opticalPosition;
    const Eigen::Vector3d bodyPosition = config_.cameraOffsetBody + bodyFromMountMatrix * mountPosition;

    return vehicleState_.positionWorld + vehicleState_.worldFromBody.toRotationMatrix() * bodyPosition;
}

Eigen::Quaterniond FrameTransformer::opticalOrientationToWorld(const Eigen::Quaterniond &opticalOrientation) const
{
    Eigen::Quaterniond normalizedOpticalOrientation = opticalOrientation;
    if (normalizedOpticalOrientation.norm() > 1e-9)
    {
        normalizedOpticalOrientation.normalize();
    }
    else
    {
        normalizedOpticalOrientation.setIdentity();
    }

    const Eigen::Quaterniond mountFromOptical(config_.opticalToMountRotation);
    Eigen::Quaterniond worldOrientation =
        vehicleState_.worldFromBody * bodyFromMountQuaternion() * mountFromOptical * normalizedOpticalOrientation;

    worldOrientation.normalize();
    return worldOrientation;
}

Eigen::Quaterniond FrameTransformer::bodyFromMountQuaternion() const
{
    if (config_.mountMode == MountMode::BellyFixedCamera)
    {
        return Eigen::Quaterniond::Identity();
    }

    return bodyFromMount_;
}

Config FrameTransformer::makeBellyFixedCameraConfig(const Eigen::Vector3d &cameraOffsetBody)
{
    Config config;
    config.mountMode = MountMode::BellyFixedCamera;
    config.cameraOffsetBody = cameraOffsetBody;
    config.opticalToMountRotation << 0.0, -1.0, 0.0,
                                     1.0,  0.0, 0.0,
                                     0.0,  0.0, 1.0;
    return config;
}

Config FrameTransformer::makeBellyGimbalCameraConfig(const Eigen::Vector3d &cameraOffsetBody)
{
    Config config;
    config.mountMode = MountMode::BellyGimbalCamera;
    config.cameraOffsetBody = cameraOffsetBody;
    config.opticalToMountRotation << 0.0, 0.0, 1.0,
                                     1.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0;
    return config;
}

MountMode FrameTransformer::parseMountMode(const std::string &modeString)
{
    const std::string lowered = toLowerCopy(modeString);

    if (lowered == "belly_fixed_camera")
    {
        return MountMode::BellyFixedCamera;
    }

    if (lowered == "belly_gimbal_camera")
    {
        return MountMode::BellyGimbalCamera;
    }

    throw std::runtime_error("Unsupported transform mount mode: " + modeString);
}

std::string FrameTransformer::mountModeToString(MountMode mountMode)
{
    switch (mountMode)
    {
    case MountMode::BellyFixedCamera:
        return "belly_fixed_camera";
    case MountMode::BellyGimbalCamera:
        return "belly_gimbal_camera";
    default:
        return "belly_fixed_camera";
    }
}
} // namespace frame_transform
