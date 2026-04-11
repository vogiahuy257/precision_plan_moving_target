#pragma once

#include <atomic>
#include <string>

#include <Eigen/Dense>
#include <opencv2/video/tracking.hpp>
#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

class KalmanFilterNode : public rclcpp::Node
{
public:
    /**
     * @brief Ham khoi tao node Kalman filter.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Khai bao parameter, doc parameter, tao subscriber/publisher
     *     va khoi tao bo loc Kalman.
     *
     * Output:
     *     Khoi tao node san sang nhan/publish du lieu.
     */
    KalmanFilterNode();

private:
    // =========================
    // Cau hinh kich thuoc state
    // =========================
    static constexpr int stateSize = 8;
    static constexpr int measurementSize = 5;

    enum StateIndex
    {
        IDX_PX = 0,
        IDX_PY = 1,
        IDX_PZ = 2,
        IDX_VX = 3,
        IDX_VY = 4,
        IDX_VZ = 5,
        IDX_BVX = 6,
        IDX_BVY = 7
    };

    // =========================
    // Ham khoi tao / parameter
    // =========================
    /**
     * @brief Khai bao cac ROS parameter.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Dang ky toan bo topic va tham so dung trong node.
     *
     * Output:
     *     Parameter duoc dang ky trong node.
     */
    void declareParameters();

    /**
     * @brief Doc gia tri parameter tu ROS parameter server.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Lay gia tri parameter da khai bao va luu vao bien noi bo.
     *
     * Output:
     *     Bien noi bo duoc cap nhat theo parameter.
     */
    void loadParameters();

    /**
     * @brief Khoi tao cau truc Kalman filter.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Tao transition matrix, measurement matrix, noise covariance
     *     va trang thai ban dau cho bo loc.
     *
     * Output:
     *     Bo loc Kalman san sang de predict/correct.
     */
    void initKalman();

    // =========================
    // Callback ROS
    // =========================
    /**
     * @brief Callback nhan pose target tu perception.
     *
     * Input:
     *     msg: geometry_msgs::msg::PoseStamped::SharedPtr
     *
     * Logic:
     *     Chuyen pose tu optical sang world/NED, tinh dt, predict,
     *     tao measurement pose + pseudo velocity roi correct Kalman,
     *     sau do publish pose/velocity da loc.
     *
     * Output:
     *     Publish target pose va target velocity da loc.
     */
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    /**
     * @brief Callback nhan lenh reset trang thai.
     *
     * Input:
     *     msg: std_msgs::msg::String::SharedPtr
     *
     * Logic:
     *     Neu nhan RESET thi reset bo loc va publish zero.
     *     Neu nhan ACTIVE thi bo trang thai force zero.
     *
     * Output:
     *     Trang thai node duoc cap nhat theo lenh reset.
     */
    void resetCallback(const std_msgs::msg::String::SharedPtr msg);

    /**
     * @brief Callback nhan co target valid.
     *
     * Input:
     *     msg: std_msgs::msg::Bool::SharedPtr
     *
     * Logic:
     *     Khi target hop le thi bo co force zero de node tiep tuc hoat dong.
     *
     * Output:
     *     Cap nhat co force zero.
     */
    void validCallback(const std_msgs::msg::Bool::SharedPtr msg);

    /**
     * @brief Callback nhan vehicle odometry tu PX4.
     *
     * Input:
     *     msg: px4_msgs::msg::VehicleOdometry::SharedPtr
     *
     * Logic:
     *     Doc quaternion, vi tri, van toc drone trong he NED va luu lai.
     *
     * Output:
     *     Cap nhat trang thai drone trong world/NED.
     */
    void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

    // =========================
    // Ham Kalman
    // =========================
    /**
     * @brief Chuan hoa dt de tranh dt am hoac khong hop le.
     *
     * Input:
     *     dt: double
     *
     * Logic:
     *     Neu dt <= 0 thi tra ve 0, nguoc lai giu nguyen.
     *
     * Output:
     *     dt da duoc chuan hoa.
     */
    double sanitizeDt(double dt) const;

    /**
     * @brief Thuc hien buoc predict cua Kalman theo dt.
     *
     * Input:
     *     dt: double
     *
     * Logic:
     *     Cap nhat transition matrix va process noise covariance,
     *     sau do goi predict cua OpenCV KalmanFilter.
     *
     * Output:
     *     Trang thai du doan statePre/statePost duoc cap nhat.
     */
    void predict(double dt);

    /**
     * @brief Tao pseudo velocity measurement tu chuyen dong tuong doi.
     *
     * Input:
     *     measurementWorld: Eigen::Vector3d
     *         Vi tri target do duoc trong world/NED.
     *     dt: double
     *         Khoang thoi gian giua hai measurement lien tiep.
     *     targetVelMeasX: double&
     *     targetVelMeasY: double&
     *
     * Logic:
     *     Tinh vi tri tuong doi target-drone, lay sai phan theo thoi gian
     *     de suy ra relative velocity, tu do tao pseudo measurement van toc
     *     cua target trong world/NED.
     *
     * Output:
     *     Tra ve true neu tao duoc pseudo measurement hop le.
     *     Tra ve false neu khong du dieu kien tinh velocity.
     */
    bool computeVelocityMeasurementFromRelativeMotion(
        const Eigen::Vector3d &measurementWorld,
        double dt,
        double &targetVelMeasX,
        double &targetVelMeasY);

    /**
     * @brief Dong bo measurement pose ve current time cua node.
     *
     * Input:
     *     measurementWorld: Eigen::Vector3d
     *         Pose target do duoc tai thoi diem chup anh.
     *     measurementTimestamp: rclcpp::Time
     *         Timestamp cua measurement.
     *
     * Logic:
     *     Dung velocity estimate hien tai cua target va drone de day
     *     measurement pose tu luc chup anh toi thoi diem hien tai.
     *
     * Output:
     *     Pose target da duoc sync len current time.
     */
    Eigen::Vector3d syncMeasurementPositionToCurrentTime(
        const Eigen::Vector3d &measurementWorld,
        const rclcpp::Time &measurementTimestamp) const;

    // =========================
    // Chuyen he truc / orientation
    // =========================
    /**
     * @brief Tao ma tran quay tu optical frame sang NED frame.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Dinh nghia quy uoc truc optical -> NED dung trong he thong.
     *
     * Output:
     *     Eigen::Matrix3d phep quay optical sang NED.
     */
    Eigen::Matrix3d opticalToNedRotation() const;

    /**
     * @brief Chuyen vi tri target tu optical frame sang world/NED.
     *
     * Input:
     *     opticalPosition: Eigen::Vector3d
     *
     * Logic:
     *     Doi truc optical sang NED, cong camera offset body,
     *     sau do quay theo tu the drone va cong vi tri drone trong world.
     *
     * Output:
     *     Vi tri target trong world/NED.
     */
    Eigen::Vector3d measurementOpticalToWorldPosition(
        const Eigen::Vector3d &opticalPosition) const;

    /**
     * @brief Chuyen orientation tag tu optical frame sang world/NED.
     *
     * Input:
     *     quaternionMessage: geometry_msgs::msg::Quaternion
     *
     * Logic:
     *     Chuan hoa quaternion dau vao, doi tu optical sang NED,
     *     roi nhan voi quaternion cua drone de ra orientation trong world.
     *
     * Output:
     *     Quaternion target trong world/NED.
     */
    Eigen::Quaterniond transformTagOrientationToWorld(
        const geometry_msgs::msg::Quaternion &quaternionMessage) const;

    // =========================
    // Publish / reset
    // =========================
    /**
     * @brief Publish pose/velocity da loc cua target.
     *
     * Input:
     *     publishTimestamp: rclcpp::Time
     *
     * Logic:
     *     Publish raw pose, filtered pose va filtered velocity.
     *     Velocity publish ra la van toc target day du trong world:
     *     vx_out = vx + bvx, vy_out = vy + bvy, vz_out = vz.
     *
     * Output:
     *     Publish len cac topic pose/velocity da cau hinh.
     */
    void publishEstimatedState(const rclcpp::Time &publishTimestamp);

    /**
     * @brief Publish gia tri zero khi reset hoac force zero.
     *
     * Input:
     *     publishTimestamp: rclcpp::Time
     *
     * Logic:
     *     Giu pose cuoi cung, velocity bang 0 de controller khong nhan
     *     du lieu chuyen dong sai khi target mat hoac he reset.
     *
     * Output:
     *     Publish hold pose va zero velocity.
     */
    void publishZero(const rclcpp::Time &publishTimestamp);

    /**
     * @brief Reset trang thai noi bo cua Kalman va cac bien nho.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Xoa trang thai khoi tao, timestamp, covariance va bien cache.
     *
     * Output:
     *     Node quay ve trang thai chua khoi tao.
     */
    void resetState();

    // =========================
    // Subscriber / Publisher
    // =========================
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr poseSub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr resetSub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr validSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicleOdomSub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoseRawPub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoseFilteredPub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetVelocityFilteredPub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debugDtPub_;

    // =========================
    // Topic name
    // =========================
    std::string inputTargetPoseTopic_;
    std::string resetCommandTopic_;
    std::string targetValidTopic_;
    std::string vehicleOdometryTopic_;

    std::string targetPoseRawTopic_;
    std::string targetPoseFilteredTopic_;
    std::string targetVelocityTopic_;

    std::string outputFrameId_;

    // =========================
    // Parameter Kalman
    // =========================
    double qAccX_{0.0121};
    double qAccY_{0.0121};
    double qAccZ_{0.0010};

    double qBiasVx_{0.0001};
    double qBiasVy_{0.0001};

    double rPosX_{0.0008};
    double rPosY_{0.0008};
    double rPosZ_{0.0040};

    double rVelX_{0.25};
    double rVelY_{0.25};

    double camOffsetX_{0.0};
    double camOffsetY_{0.0};
    double camOffsetZ_{-0.1};

    double maxPredictDt_{0.15};
    double staleMeasurementThresholdSec_{0.2};
    double smallNegativeDtToleranceSec_{0.02};

    double biasFromRelVelGain_{1.3};
    double biasFromRelPosGain_{0.15};
    double biasLimit_{5.0};

    // =========================
    // Bien trang thai noi bo
    // =========================
    cv::KalmanFilter kf_;

    bool initialized_{false};
    bool vehicleOdomValid_{false};
    bool lastRelativeWorldValid_{false};

    std::atomic<bool> forceZero_{false};

    rclcpp::Time lastMeasurementTimestamp_{0, 0, RCL_ROS_TIME};
    rclcpp::Time lastPredictTimestamp_{0, 0, RCL_ROS_TIME};

    Eigen::Quaterniond vehicleQned_{1.0, 0.0, 0.0, 0.0};
    Eigen::Vector3d vehiclePosNed_{0.0, 0.0, 0.0};
    Eigen::Vector3d vehicleVelNed_{0.0, 0.0, 0.0};

    Eigen::Vector3d lastMeasurementWorld_{0.0, 0.0, 0.0};
    Eigen::Vector3d lastMeasurementWorldSync_{0.0, 0.0, 0.0};
    Eigen::Vector3d lastRelativeWorld_{0.0, 0.0, 0.0};

    geometry_msgs::msg::Quaternion lastOrientation_;
};