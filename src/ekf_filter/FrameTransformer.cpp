#include "FrameTransformer.hpp"

void FrameTransformer::setCameraOffsetBody(const Eigen::Vector3d &offsetBody)
{
    cameraOffsetBody_ = offsetBody;
}

void FrameTransformer::setVehicleState(
    const Eigen::Vector3d &positionNed,
    const Eigen::Quaterniond &worldFromBody)
{
    vehiclePositionNed_ = positionNed;
    worldFromBody_ = worldFromBody;

    if (worldFromBody_.norm() > 1e-9)
    {
        worldFromBody_.normalize();
    }
    else
    {
        worldFromBody_.setIdentity();
    }
}

Eigen::Vector3d FrameTransformer::opticalPositionToWorld(
    const Eigen::Vector3d &opticalPosition) const
{
    const Eigen::Vector3d bodyPosition =
        cameraOffsetBody_ + opticalToBody_ * opticalPosition;

    return vehiclePositionNed_ +
           worldFromBody_.toRotationMatrix() * bodyPosition;
}

Eigen::Quaterniond FrameTransformer::opticalOrientationToWorld(
    const Eigen::Quaterniond &opticalOrientation) const
{
    Eigen::Quaterniond qOptical = opticalOrientation;
    if (qOptical.norm() > 1e-9)
    {
        qOptical.normalize();
    }
    else
    {
        qOptical.setIdentity();
    }

    const Eigen::Quaterniond bodyFromOptical(opticalToBody_);
    Eigen::Quaterniond result =
        worldFromBody_ * bodyFromOptical * qOptical;
    result.normalize();
    return result;
}
