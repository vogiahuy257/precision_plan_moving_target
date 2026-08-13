#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

class FrameTransformer
{
public:
    void setCameraOffsetBody(const Eigen::Vector3d &offsetBody);
    void setVehicleState(
        const Eigen::Vector3d &positionNed,
        const Eigen::Quaterniond &worldFromBody);

    Eigen::Vector3d opticalPositionToWorld(
        const Eigen::Vector3d &opticalPosition) const;

    Eigen::Quaterniond opticalOrientationToWorld(
        const Eigen::Quaterniond &opticalOrientation) const;

private:
    Eigen::Vector3d cameraOffsetBody_{0.0, 0.0, -0.1};
    Eigen::Vector3d vehiclePositionNed_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond worldFromBody_{Eigen::Quaterniond::Identity()};

    // Preserve the fixed belly-camera transform used by the uploaded KF.
    Eigen::Matrix3d opticalToBody_{
        (Eigen::Matrix3d() <<
             -1.0, 0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0, 0.0, 1.0)
            .finished()};
};
