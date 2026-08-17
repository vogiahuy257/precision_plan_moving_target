#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

#include "DataStructs.hpp"
#include "DynamicMeasurementNoise.hpp"
#include "FrameTransformer.hpp"
#include "DebugLogger.hpp"

class KalmanFilterNode : public rclcpp::Node
{
public:
    KalmanFilterNode();

private:
    /**
     * Cac trang thai chan xu ly chinh cua node.
     */
    enum class ProcessingBlockState
    {
        None,
        WaitVehicleOdom,
        WaitVehicleLocalPos,
        WaitFirstMeasurement,
        ForceZeroHold
    };

    /**
     * Khai bao toan bo parameter su dung trong node.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Khai bao nhom topic, frame_id, debug, noise va transform.
     *
     * Output:
     *     Parameter duoc dang ky de co the override bang yaml.
     */
    void declareParameters();

    /**
     * Doc parameter tu ROS parameter server vao data trung tam.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Lay cac gia tri param va luu vao data_.config.
     *
     * Output:
     *     data_.config duoc cap nhat day du.
     */
    void loadParameters();

    /**
     * Khoi tao FrameTransformer theo config transform hien tai.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Parse mount mode, tao config tuong ung va dong bo vehicle state.
     *     Neu parse loi thi fallback ve belly_fixed_camera.
     *
     * Output:
     *     frameTransformer_ san sang cho phep bien doi.
     */
    void initFrameTransformer();

    /**
     * Khoi tao bo loc Kalman voi state 6 bien va measurement 3 bien.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Tao measurement matrix chi do truc tiep vi tri, nap measurement noise
     *     va gan covariance ban dau cho state van toc lon hon de filter mem hon.
     *
     * Output:
     *     kf_ duoc khoi tao day du.
     */
    void initKalman();

    /**
     * Reset toan bo state runtime cua node.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Xoa trang thai Kalman, measurement cu va reset timing de bat dau lai sach.
     *
     * Output:
     *     data_ va kf_ tro ve trang thai moi khoi dong.
     */
    void resetState();

    /**
     * Dong bo vehicle state sang FrameTransformer.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Danh dau vehicle valid khi da co ca odom va local position.
     *
     * Output:
     *     frameTransformer_ nhan vehicle state moi nhat.
     */
    void updateFrameTransformerVehicleState();

    /**
     * Callback nhan measurement pose target trong he optical.
     *
     * Input:
     *     msg: geometry_msgs::msg::PoseStamped::SharedPtr
     *
     * Logic:
     *     Chuyen pose optical sang world, luu measurement moi nhat va correct Kalman.
     *     Neu day la measurement dau tien thi dung no de init state.
     *
     * Output:
     *     data_.targetMeasurement va kf_ duoc cap nhat.
     */
    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    /**
     * Callback nhan lenh RESET hoac ACTIVE.
     *
     * Input:
     *     msg: std_msgs::msg::String::SharedPtr
     *
     * Logic:
     *     RESET thi reset filter va bat che do hold.
     *     ACTIVE thi tat forceZero va cho phep publish lai.
     *
     * Output:
     *     data_.runtime duoc cap nhat.
     */
    void resetCallback(const std_msgs::msg::String::SharedPtr msg);

    /**
     * Callback nhan odometry quaternion cua drone.
     *
     * Input:
     *     msg: px4_msgs::msg::VehicleOdometry::SharedPtr
     *
     * Logic:
     *     Chuyen quaternion q[] sang Eigen, chuan hoa va cap nhat flag odom valid.
     *
     * Output:
     *     data_.vehicle.worldFromBody duoc cap nhat.
     */
    void vehicleOdometryCallback(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);

    /**
     * Callback nhan local position NED cua drone.
     *
     * Input:
     *     msg: px4_msgs::msg::VehicleLocalPosition::SharedPtr
     *
     * Logic:
     *     Kiem tra finite, cap nhat vi tri va van toc world/NED hien tai.
     *
     * Output:
     *     data_.vehicle.positionWorld va velocityWorld duoc cap nhat.
     */
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    /**
     * Xu ly 1 measurement moi va publish ngay lap tuc theo timestamp cua measurement.
     *
     * Input:
     *     measurementTimestamp: timestamp goc cua measurement dang xu ly
     *     measurement: vector measurement vi tri world [x y z]^T
     *
     * Logic:
     *     - Danh gia block state hien tai
     *     - Tinh dt chinh xac tu 2 timestamp measurement lien tiep
     *     - Predict neu dt > 0
     *     - Correct bang measurement moi
     *     - Publish output ngay sau khi xu ly xong, khong qua timer 33 ms
     *
     * Output:
     *     State output duoc publish ngay sau khi xu ly xong, header.stamp la thoi diem publish thuc te.
     *     measurementTimestamp chi duoc dung de tinh predict dt.
     */
    void processAndPublishMeasurement(
        const rclcpp::Time &measurementTimestamp,
        const cv::Mat &measurement);

    /**
     * Buoc predict cua Kalman theo mo hinh constant velocity.
     *
     * Input:
     *     dt: buoc thoi gian giua 2 lan predict
     *
     * Logic:
     *     Dung mo hinh white acceleration cho tung truc x, y, z de tao process noise.
     *
     * Output:
     *     kf_ duoc predict sang trang thai moi.
     */
    void predict(double dt);

    /**
     * Cap nhat ma tran nhieu do R dong truoc buoc correct.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     - Lay khoang cach camera den target lam xap xi h.
     *     - Lay norm van toc goc drone de uoc luong muc rung/lac attitude.
     *     - Tang R_x, R_y theo cong thuc (h * sigma_theta)^2.
     *     - Gioi han phan tang them de tranh Kalman bo measurement qua manh.
     *
     * Output:
     *     kf_.measurementNoiseCov duoc cap nhat theo dieu kien bay hien tai.
     */
    void updateDynamicMeasurementNoise();

    /**
     * Publish raw measurement, filtered position va estimated velocity.
     *
     * Input:
     *     nowTimestamp: thoi gian dong dau output
     *
     * Logic:
     *     Lay state hien tai tu Kalman roi publish ra 3 topic.
     *     Orientation dung orientation world gan nhat cua target.
     *
     * Output:
     *     Tat ca topic output duoc publish.
     */
    void publishEstimatedState(const rclcpp::Time &nowTimestamp);

    /**
     * Publish state hold de drone hover tai cho.
     *
     * Input:
     *     nowTimestamp: thoi gian dong dau output
     *
     * Logic:
     *     - Vi output dang la world pose, neu publish (0,0,0) drone se bi keo ve goc world
     *     - De hover tai cho, pose raw va filtered phai publish vi tri world hien tai cua UAV
     *     - Van toc output duoc dua ve 0 de dung yeu cau di chuyen
     *     - Neu chua co vehicle local position thi moi fallback ve 0
     *
     * Output:
     *     Downstream nhan duoc lenh giu vi tri hien tai va van toc bang 0.
     */
    void publishZero(const rclcpp::Time &nowTimestamp);

    /**
     * Ghi log tong hop state hien tai cua node.
     *
     * Input:
     *     prefix: chuoi mo ta bo canh can in truoc state
     *
     * Logic:
     *     Tong hop cac flag runtime va timing quan trong de debug.
     *
     * Output:
     *     In ra 1 dong state summary.
     */
    void logStateSummary(const std::string &prefix);

    /**
     * Cap nhat ly do dang o che do forceZero / hold.
     *
     * Input:
     *     reason: ly do kich hoat forceZero
     *
     * Logic:
     *     Luu lai nguyen nhan de report block state co the in dung ngu canh.
     *
     * Output:
     *     forceZeroReason_ duoc cap nhat.
     */
    void setForceZeroReason(const std::string &reason);

    /**
     * Danh gia trang thai block hien tai cua node.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Chon 1 nguyen nhan block duy nhat, uu tien:
     *     forceZero -> odom -> local position -> first measurement.
     *
     * Output:
     *     Trang thai block hien tai.
     */
    ProcessingBlockState evaluateProcessingBlockState() const;

    /**
     * Tao message log tu trang thai block.
     *
     * Input:
     *     state: trang thai block hien tai
     *
     * Logic:
     *     Chuyen enum sang message ro rang de in ra terminal.
     *
     * Output:
     *     Chuoi mo ta block hien tai.
     */
    std::string buildProcessingBlockMessage(ProcessingBlockState state) const;

    /**
     * Log block state chi 1 lan khi co thay doi.
     *
     * Input:
     *     state: trang thai block hien tai
     *
     * Logic:
     *     Neu state khong doi thi khong log lai.
     *     Neu state doi thi log 1 lan.
     *
     * Output:
     *     Terminal khong bi spam log.
     */
    void reportProcessingBlockState(ProcessingBlockState state);

private:
    static constexpr int stateSize = 6;
    static constexpr int measurementSize = 3;

    kalman_filter_data::SystemData data_{};

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr poseSub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr resetSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr vehicleOdomSub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicleLocalPosSub_;

    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoseRawPub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetPoseFilteredPub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr targetRelVelPub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr targetCovariancePub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr targetProcessNoisePub_;


    cv::KalmanFilter kf_;
    frame_transform::FrameTransformer frameTransformer_;
    DynamicMeasurementNoiseEstimator dynamicMeasurementNoiseEstimator_;
    DebugLogger debugLogger_;

    std::string forceZeroReason_{"Kalman paused: startup"};
    ProcessingBlockState lastProcessingBlockState_{ProcessingBlockState::None};
};