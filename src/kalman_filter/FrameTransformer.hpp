#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace frame_transform
{
enum class MountMode
{
    BellyFixedCamera,
    BellyGimbalCamera
};

struct Config
{
    MountMode mountMode{MountMode::BellyFixedCamera};
    Eigen::Vector3d cameraOffsetBody{0.0, 0.0, -0.1};
    Eigen::Matrix3d opticalToMountRotation{Eigen::Matrix3d::Identity()};
};

struct VehicleState
{
    Eigen::Quaterniond worldFromBody{1.0, 0.0, 0.0, 0.0};
    Eigen::Vector3d positionWorld{0.0, 0.0, 0.0};
    Eigen::Vector3d velocityWorld{0.0, 0.0, 0.0};
    bool valid{false};
};

class FrameTransformer
{
public:
    FrameTransformer();

    void setConfig(const Config &config);
    void setVehicleState(const VehicleState &vehicleState);
    void setBodyFromMountQuaternion(const Eigen::Quaterniond &bodyFromMount);
    void setBodyFromMountEulerDeg(double yawDeg, double pitchDeg, double rollDeg);

    Eigen::Vector3d opticalPositionToWorld(const Eigen::Vector3d &opticalPosition) const;
    Eigen::Quaterniond opticalOrientationToWorld(const Eigen::Quaterniond &opticalOrientation) const;
    Eigen::Quaterniond bodyFromMountQuaternion() const;

    static Config makeBellyFixedCameraConfig(const Eigen::Vector3d &cameraOffsetBody);
    static Config makeBellyGimbalCameraConfig(const Eigen::Vector3d &cameraOffsetBody);
    static MountMode parseMountMode(const std::string &modeString);
    static std::string mountModeToString(MountMode mountMode);

private:
    Config config_{};
    VehicleState vehicleState_{};
    Eigen::Quaterniond bodyFromMount_{1.0, 0.0, 0.0, 0.0};
};
} // namespace frame_transform
