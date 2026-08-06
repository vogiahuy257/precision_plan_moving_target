#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "DataStructs.hpp"

namespace frame_transform
{
using MountMode = kalman_filter_data::MountMode;
using TransformConfig = kalman_filter_data::TransformConfig;
using VehicleStateData = kalman_filter_data::VehicleStateData;

class FrameTransformer
{
public:
    FrameTransformer();

    void setConfig(const TransformConfig &config);
    void setVehicleState(const VehicleStateData &vehicleState);
    void setBodyFromMountQuaternion(const Eigen::Quaterniond &bodyFromMount);
    void setBodyFromMountEulerDeg(double yawDeg, double pitchDeg, double rollDeg);

    Eigen::Vector3d opticalPositionToWorld(const Eigen::Vector3d &opticalPosition) const;
    Eigen::Quaterniond opticalOrientationToWorld(const Eigen::Quaterniond &opticalOrientation) const;

    Eigen::Quaterniond bodyFromMountQuaternion() const;

    static TransformConfig makeBellyFixedCameraConfig(const Eigen::Vector3d &cameraOffsetBody);
    static TransformConfig makeBellyGimbalCameraConfig(const Eigen::Vector3d &cameraOffsetBody);
    static MountMode parseMountMode(const std::string &modeString);
    static std::string mountModeToString(MountMode mountMode);

private:
    TransformConfig config_{};
    VehicleStateData vehicleState_{};
    Eigen::Quaterniond bodyFromMount_{Eigen::Quaterniond::Identity()};
};

} // namespace frame_transform