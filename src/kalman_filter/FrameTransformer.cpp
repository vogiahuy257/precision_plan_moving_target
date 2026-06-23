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

/**
 * Chuyen chuoi ve dang viet thuong de so sanh mode an toan hon.
 *
 * Input:
 *     value: chuoi dau vao can doi sang chu thuong
 *
 * Logic:
 *     Tao ban sao cua chuoi, sau do duyet tung ky tu va doi sang lower-case.
 *     Ham nay duoc dung de parse mount_mode khong phu thuoc viet hoa/viet thuong.
 *
 * Output:
 *     Tra ve chuoi da duoc chuyen ve viet thuong.
 */
std::string toLowerCopy(const std::string &value)
{
    try
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
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("frame_transform::toLowerCopy failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("frame_transform::toLowerCopy failed: unknown exception");
    }
}
} // namespace

/**
 * Khoi tao doi tuong FrameTransformer voi cau hinh mac dinh.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Mac dinh su dung belly_fixed_camera va camera offset theo truc body la
 *     (0.0, 0.0, -0.1). Cau hinh nay phu hop cho camera gan co dinh huong xuong.
 *
 * Output:
 *     Tao doi tuong FrameTransformer voi config_ mac dinh.
 */
FrameTransformer::FrameTransformer()
{
    try
    {
        config_ = makeBellyFixedCameraConfig(Eigen::Vector3d(0.0, 0.0, -0.1));
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::FrameTransformer failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::FrameTransformer failed: unknown exception");
    }
}

/**
 * Cap nhat cau hinh bien doi he truc cho FrameTransformer.
 *
 * Input:
 *     config: cau hinh bien doi gom mount mode, camera offset va ma tran quay optical->mount
 *
 * Logic:
 *     Gan truc tiep cau hinh moi vao bien config_ de cac phep bien doi ve sau su dung.
 *
 * Output:
 *     Khong co. Noi bo doi tuong duoc cap nhat config moi.
 */
void FrameTransformer::setConfig(const TransformConfig &config)
{
    try
    {
        config_ = config;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::setConfig failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::setConfig failed: unknown exception");
    }
}

/**
 * Cap nhat trang thai UAV trong he world.
 *
 * Input:
 *     vehicleState:
 *         - positionWorld: vi tri UAV trong he world
 *         - worldFromBody: quaternion quay tu body sang world
 *
 * Logic:
 *     Luu trang thai UAV vao bien noi bo. Dong thoi chuan hoa quaternion
 *     worldFromBody de tranh sai so so hoc. Neu quaternion khong hop le
 *     thi dua ve identity de dam bao an toan khi bien doi.
 *
 * Output:
 *     Khong co. Noi bo doi tuong duoc cap nhat vehicle state moi.
 */
void FrameTransformer::setVehicleState(const VehicleStateData &vehicleState)
{
    try
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
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::setVehicleState failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::setVehicleState failed: unknown exception");
    }
}

/**
 * Cap nhat quaternion quay tu mount sang body.
 *
 * Input:
 *     bodyFromMount: quaternion mo ta tu the cua mount trong he body
 *
 * Logic:
 *     Luu quaternion bodyFromMount vao bien noi bo va chuan hoa no.
 *     Neu quaternion khong hop le thi dua ve identity de tranh loi bien doi.
 *
 * Output:
 *     Khong co. Noi bo doi tuong duoc cap nhat quaternion mount moi.
 */
void FrameTransformer::setBodyFromMountQuaternion(const Eigen::Quaterniond &bodyFromMount)
{
    try
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
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::setBodyFromMountQuaternion failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "FrameTransformer::setBodyFromMountQuaternion failed: unknown exception");
    }
}

/**
 * Cap nhat quaternion quay tu mount sang body bang goc Euler don vi do.
 *
 * Input:
 *     yawDeg: goc quay quanh truc Z, don vi do
 *     pitchDeg: goc quay quanh truc Y, don vi do
 *     rollDeg: goc quay quanh truc X, don vi do
 *
 * Logic:
 *     Doi yaw/pitch/roll tu do sang radian, sau do tao quaternion bodyFromMount
 *     theo thu tu quay Z-Y-X. Cuoi cung goi lai ham setBodyFromMountQuaternion()
 *     de luu va chuan hoa quaternion.
 *
 * Output:
 *     Khong co. Noi bo doi tuong duoc cap nhat bodyFromMount_ moi.
 */
void FrameTransformer::setBodyFromMountEulerDeg(double yawDeg, double pitchDeg, double rollDeg)
{
    try
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
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::setBodyFromMountEulerDeg failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::setBodyFromMountEulerDeg failed: unknown exception");
    }
}

/**
 * Chuyen vi tri tu he optical cua camera sang he world.
 *
 * Input:
 *     opticalPosition: vector vi tri muc tieu trong he optical cua camera
 *
 * Logic:
 *     - Doi optical -> mount bang ma tran config_.opticalToMountRotation
 *     - Doi mount -> body bang quaternion bodyFromMountQuaternion()
 *     - Cong them camera offset trong he body
 *     - Doi body -> world bang quaternion worldFromBody cua UAV
 *     - Cong them vi tri UAV trong he world de ra vi tri cuoi cung cua target
 *
 * Output:
 *     Tra ve vector vi tri muc tieu trong he world.
 */
Eigen::Vector3d FrameTransformer::opticalPositionToWorld(const Eigen::Vector3d &opticalPosition) const
{
    try
    {
        const Eigen::Matrix3d bodyFromMountMatrix = bodyFromMountQuaternion().toRotationMatrix();
        const Eigen::Vector3d mountPosition = config_.opticalToMountRotation * opticalPosition;
        const Eigen::Vector3d bodyPosition = config_.cameraOffsetBody + bodyFromMountMatrix * mountPosition;

        return vehicleState_.positionWorld + vehicleState_.worldFromBody.toRotationMatrix() * bodyPosition;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::opticalPositionToWorld failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::opticalPositionToWorld failed: unknown exception");
    }
}

/**
 * Chuyen tu the quaternion tu he optical cua camera sang he world.
 *
 * Input:
 *     opticalOrientation: quaternion tu the cua muc tieu trong he optical
 *
 * Logic:
 *     - Chuan hoa quaternion dau vao de tranh sai so
 *     - Tao quaternion mountFromOptical tu ma tran quay optical->mount
 *     - Ghep cac phep quay theo thu tu:
 *           optical -> mount -> body -> world
 *     - Chuan hoa quaternion ket qua truoc khi tra ve
 *
 * Output:
 *     Tra ve quaternion tu the cua muc tieu trong he world.
 */
Eigen::Quaterniond FrameTransformer::opticalOrientationToWorld(const Eigen::Quaterniond &opticalOrientation) const
{
    try
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
            vehicleState_.worldFromBody *
            bodyFromMountQuaternion() *
            mountFromOptical *
            normalizedOpticalOrientation;

        worldOrientation.normalize();
        return worldOrientation;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::opticalOrientationToWorld failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "FrameTransformer::opticalOrientationToWorld failed: unknown exception");
    }
}

/**
 * Lay quaternion quay tu mount sang body phu hop voi mount mode hien tai.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     - Neu camera la belly_fixed_camera thi mount co dinh voi body,
 *       vi vay tra ve identity.
 *     - Neu camera la belly_gimbal_camera thi su dung quaternion bodyFromMount_
 *       da duoc cap nhat tu gimbal.
 *
 * Output:
 *     Tra ve quaternion quay tu mount sang body.
 */
Eigen::Quaterniond FrameTransformer::bodyFromMountQuaternion() const
{
    try
    {
        if (config_.mountMode == MountMode::BellyFixedCamera ||
            config_.mountMode == MountMode::BellyFixedCameraRight90)
        {
            return Eigen::Quaterniond::Identity();
        }

        return bodyFromMount_;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::bodyFromMountQuaternion failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::bodyFromMountQuaternion failed: unknown exception");
    }
}

/**
 * Tao cau hinh cho camera gan bung co dinh nhung bi xoay phai 90 do.
 *
 * Input:
 *     cameraOffsetBody: do lech vi tri camera trong he body cua UAV
 *
 * Logic:
 *     - Dat mount mode la BellyFixedCameraRight90
 *     - Camera van la fixed mount, khong dung gimbal
 *     - Camera van nhin xuong bung UAV
 *     - Anh/camera xoay phai 90 do so voi belly_fixed_camera mac dinh
 *
 * Output:
 *     Tra ve TransformConfig cho belly_fixed_camera_right_90.
 */
TransformConfig FrameTransformer::makeBellyFixedCameraRight90Config(
    const Eigen::Vector3d &cameraOffsetBody)
{
    try
    {
        TransformConfig config;
        config.mountMode = MountMode::BellyFixedCameraRight90;
        config.mountModeString = "belly_fixed_camera_right_90";
        config.cameraOffsetBody = cameraOffsetBody;

        config.opticalToMountRotation << -1.0, 0.0, 0.0,
                                         0.0, -1.0, 0.0,
                                         0.0, 0.0, 1.0;

        return config;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::makeBellyFixedCameraRight90Config failed: ") +
            exception.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "FrameTransformer::makeBellyFixedCameraRight90Config failed: unknown exception");
    }
}

/**
 * Tao cau hinh mac dinh cho camera gan bung co dinh.
 *
 * Input:
 *     cameraOffsetBody: do lech vi tri camera trong he body cua UAV
 *
 * Logic:
 *     - Dat mount mode la BellyFixedCamera
 *     - Dat mountModeString de dong bo voi config chung
 *     - Gan camera offset theo tham so dau vao
 *     - Gan ma tran quay optical->mount phu hop cho camera co dinh huong bung
 *
 * Output:
 *     Tra ve TransformConfig hoan chinh cho belly_fixed_camera.
 */
TransformConfig FrameTransformer::makeBellyFixedCameraConfig(const Eigen::Vector3d &cameraOffsetBody)
{
    try
    {
        TransformConfig config;
        config.mountMode = MountMode::BellyFixedCamera;
        config.mountModeString = "belly_fixed_camera";
        config.cameraOffsetBody = cameraOffsetBody;
        config.opticalToMountRotation << 0.0, -1.0, 0.0,
                                         1.0,  0.0, 0.0,
                                         0.0,  0.0, 1.0;
        return config;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::makeBellyFixedCameraConfig failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "FrameTransformer::makeBellyFixedCameraConfig failed: unknown exception");
    }
}

/**
 * Tao cau hinh mac dinh cho camera gan tren gimbal o bung UAV.
 *
 * Input:
 *     cameraOffsetBody: do lech vi tri camera trong he body cua UAV
 *
 * Logic:
 *     - Dat mount mode la BellyGimbalCamera
 *     - Dat mountModeString de dong bo voi config chung
 *     - Gan camera offset theo tham so dau vao
 *     - Gan ma tran quay optical->mount phu hop cho camera tren gimbal
 *
 * Output:
 *     Tra ve TransformConfig hoan chinh cho belly_gimbal_camera.
 */
TransformConfig FrameTransformer::makeBellyGimbalCameraConfig(const Eigen::Vector3d &cameraOffsetBody)
{
    try
    {
        TransformConfig config;
        config.mountMode = MountMode::BellyGimbalCamera;
        config.mountModeString = "belly_gimbal_camera";
        config.cameraOffsetBody = cameraOffsetBody;
        config.opticalToMountRotation << 0.0, 0.0, 1.0,
                                         1.0, 0.0, 0.0,
                                         0.0, 1.0, 0.0;
        return config;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::makeBellyGimbalCameraConfig failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "FrameTransformer::makeBellyGimbalCameraConfig failed: unknown exception");
    }
}

/**
 * Parse chuoi mount mode thanh enum noi bo.
 *
 * Input:
 *     modeString: chuoi mount mode doc tu param hoac config
 *
 * Logic:
 *     Doi chuoi dau vao ve viet thuong, sau do so sanh voi cac gia tri ho tro:
 *     - belly_fixed_camera
 *     - belly_gimbal_camera
 *     Neu khong khop thi nem exception de bao loi cau hinh.
 *
 * Output:
 *     Tra ve enum MountMode tuong ung.
 */
MountMode FrameTransformer::parseMountMode(const std::string &modeString)
{
    try
    {
        const std::string lowered = toLowerCopy(modeString);

        if (lowered == "belly_fixed_camera_right_90")
        {
            return MountMode::BellyFixedCameraRight90;
        }

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
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::parseMountMode failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::parseMountMode failed: unknown exception");
    }
}

/**
 * Chuyen enum mount mode thanh chuoi de debug hoac log.
 *
 * Input:
 *     mountMode: enum mount mode noi bo
 *
 * Logic:
 *     Doi enum sang chuoi dung voi ten param/config de log va debug de doc.
 *     Neu gia tri enum khong hop le thi tra ve belly_fixed_camera de an toan.
 *
 * Output:
 *     Tra ve chuoi mo ta mount mode.
 */
std::string FrameTransformer::mountModeToString(MountMode mountMode)
{
    try
    {
        switch (mountMode)
        {
        case MountMode::BellyFixedCamera:
            return "belly_fixed_camera";
        case MountMode::BellyGimbalCamera:
            return "belly_gimbal_camera";
        case MountMode::BellyFixedCameraRight90:
            return "belly_fixed_camera_right_90";
        default:
            return "belly_fixed_camera";
        }
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("FrameTransformer::mountModeToString failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("FrameTransformer::mountModeToString failed: unknown exception");
    }
}
} // namespace frame_transform