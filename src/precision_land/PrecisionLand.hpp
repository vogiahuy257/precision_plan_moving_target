#pragma once

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <px4_msgs/msg/vehicle_command_ack.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <std_msgs/msg/string.hpp>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>

#include "ControlTypes.hpp"
#include "XYVelocityController.hpp"
#include "DescentZController.hpp"
#include "PredictionModel.hpp"
#include "DisarmController.hpp"
#include "PrecisionLandDebugLogger.hpp"
#include "PipelineTimingCollector.hpp"

class PrecisionLand : public px4_ros2::ModeBase
{
public:
    explicit PrecisionLand(rclcpp::Node &node);

    void onActivate() override;
    void onDeactivate() override;
    void updateSetpoint(float dt_s) override;

private:
    enum class State
    {
        Search,
        Descend,
        Finished
    };

    struct TargetWorldData
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        Eigen::Vector3d velocity{0.0, 0.0, 0.0};

        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        rclcpp::Time velocityTimestamp{0, 0, RCL_ROS_TIME};

        bool validPose{false};
        bool validVelocity{false};
    };

private:
    /**
     * Load toàn bộ parameter và cấu hình các controller.
     *
     * Input:
     *     khong co
     *
     * Logic:
     *     - Khai báo parameter mặc định
     *     - Đọc parameter từ ROS2 param server
     *     - Cấu hình XY controller, Z controller, DisarmController
     *     - Bật/tắt debug logger
     *
     * Output:
     *     cập nhật các biến param nội bộ
     */
    void loadParameters();

    /**
     * Giữ UAV hover tại chỗ.
     *
     * Input:
     *     khong co
     *
     * Logic:
     *     - Xuất setpoint velocity = 0 cho cả XYZ
     *
     * Output:
     *     publish trajectory setpoint hover
     */
    void Hover();

    /**
     * Callback nhận target raw sau đổi sang NED.
     *
     * Input:
     *     msg: PoseStamped target raw
     *
     * Logic:
     *     - Lưu giá trị raw mới nhất để phục vụ debug logger
     *
     * Output:
     *     cập nhật _latestTargetRawWorld và _latestTargetRawValid
     */
    void targetPoseRawCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void gimbalAttCallback(const geometry_msgs::msg::Vector3::SharedPtr msg);
    void vehicleCommandAckCallback(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg);

    void handleSearchState(bool targetLost);
    void handleDescendState(float dt_s, bool targetLost);
    void handleFinishedState();

    void updateTargetLostStatus(bool targetLost);
    bool checkTargetTimeout() const;
    void switchToState(State state);
    std::string stateName(State state) const;

    float computeLeadTimeSec(float dt_s, const rclcpp::Time &ctrlStartNow) const;
    precision_land::PredictionInput buildPredictionInput(float dt_s, const rclcpp::Time &ctrlStartNow);
    Eigen::Vector2f estimateVehicleAccelerationXY(float dt_s);

    void publishPredictedTargetDebug(const rclcpp::Time &stamp, const Eigen::Vector3f &targetFutureWorld);
    void publishTimingDebug(
        const rclcpp::Time &ctrlStartNow,
        const rclcpp::Time &ctrlEndNow,
        const rclcpp::Time &cmdPubNow);

    /**
     * Ghi nhóm timing vào sample debug.
     *
     * Input:
     *     sample: object sample cần điền dữ liệu
     *     ctrlStartNow: thời điểm bắt đầu xử lý control
     *     ctrlEndNow: thời điểm kết thúc xử lý control
     *     cmdPubNow: thời điểm publish setpoint
     *
     * Logic:
     *     - Tính poseWaitDt, velWaitDt, controlProcessingDt
     *     - Tính sendCmdDt và totalImageToCmdDt
     *
     * Output:
     *     cập nhật sample.timing
     */
    void fillDebugTimingSample(
        precision_land::PrecisionLandDebugSample &sample,
        const rclcpp::Time &ctrlStartNow,
        const rclcpp::Time &ctrlEndNow,
        const rclcpp::Time &cmdPubNow) const;

    /**
     * Ghi 1 sample debug của vòng điều khiển hiện tại.
     *
     * Input:
     *     ctrlStartNow: thời điểm bắt đầu control
     *     ctrlEndNow: thời điểm kết thúc control
     *     cmdPubNow: thời điểm publish command
     *     predictionOutput: đầu ra dự đoán target tương lai
     *     xyInput: input XY controller
     *     xyOutput: output XY controller
     *     disarmOutput: output DisarmController
     *     vz: velocity setpoint trục Z cuối
     *     altitudeNow: độ cao tuyệt đối hiện tại
     *
     * Logic:
     *     - Gom toàn bộ biến cần debug vào PrecisionLandDebugSample
     *     - Gọi _debugLogger.logSample(sample)
     *
     * Output:
     *     thêm 1 dòng log vào buffer CSV nếu debug đang bật
     */
    void logDebugSample(
        const rclcpp::Time &ctrlStartNow,
        const rclcpp::Time &ctrlEndNow,
        const rclcpp::Time &cmdPubNow,
        const precision_land::PredictionOutput &predictionOutput,
        const precision_land::XYControllerInput &xyInput,
        const precision_land::XYControllerOutput &xyOutput,
        const precision_land::DisarmControllerOutput &disarmOutput,
        float vz,
        float altitudeNow);

private:
    rclcpp::Node &_node;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectorySetpoint;
    std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicleLocalPosition;
    std::shared_ptr<px4_ros2::OdometryAttitude> _vehicleAttitude;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetPoseRawSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetPoseSub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _targetVelocitySub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr _vehicleLandDetectedSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicleLocalPosSub;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr _gimbalSub;
    rclcpp::Subscription<px4_msgs::msg::VehicleCommandAck>::SharedPtr _vehicleCommandAckSub;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _gimbalSeqPub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _debugTargetPredPub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _debugDtPub;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr _vehicleCommandPub;

    std::string _targetPoseRawTopic;
    std::string _targetPoseTopic;
    std::string _targetVelocityTopic;
    std::string _vehicleLandDetectedTopic;
    std::string _vehicleLocalPositionTopic;
    std::string _gimbalCommandTopic;
    std::string _gimbalAttitudeTopic;

    float _paramPidDeadband{0.05f};
    float _paramTargetTimeout{3.0f};

    float _paramDescentKp{0.9f};
    float _paramDescentKi{0.01f};
    float _paramDescentKd{0.0f};
    float _paramDescentMaxVelocity{10.0f};
    float _paramSlewAcc{10.0f};

    float _paramLandZoneZ{0.5f};
    float _paramDescentVel{0.5f};

    float _paramDescentGateRadius{0.3f};
    float _paramVmin{0.45f};
    float _paramVmax{0.8f};

    bool _paramUsePredictiveError{true};
    float _paramPredictionDtMax{0.75f};
    float _paramControlExtraLeadSec{0.25f};

    float _paramPredictiveAccGain{0.0f};
    float _paramPredictiveAccLpfAlpha{0.4f};
    float _paramPredictiveAccMax{4.0f};

    bool _paramDebugLogger{false};

    precision_land::XYVelocityController _xyVelocityController;
    precision_land::DescentZController _descentZController;
    precision_land::PredictionModel _predictionModel;
    precision_land::DisarmController _disarmController;
    precision_land::PrecisionLandDebugLogger _debugLogger;
    precision_land::PipelineTimingCollector _pipelineTimingCollector;

    State _state{State::Search};
    TargetWorldData _targetWorld{};

    rclcpp::Time _imageTimestamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time _targetPoseRxNow{0, 0, RCL_ROS_TIME};
    rclcpp::Time _targetVelRxNow{0, 0, RCL_ROS_TIME};

    bool _searchStarted{false};
    bool _targetLostPrev{true};

    bool _distBottomValid{false};
    float _zDistBottom{0.0f};

    bool _landDetected{false};

    bool _gimbalReady{false};
    bool _gimbalValid{false};
    float _gimbalPitchDeg{0.0f};
    Eigen::Quaterniond _qGimbal{Eigen::Quaterniond::Identity()};

    bool _yawSpInit{false};
    float _yawSp{0.0f};

    float _approachAltitude{0.0f};

    float _prevVehicleVelX{0.0f};
    float _prevVehicleVelY{0.0f};
    float _vehicleAccXFilt{0.0f};
    float _vehicleAccYFilt{0.0f};
    bool _prevVehicleVelValid{false};

    Eigen::Vector3f _latestTargetRawWorld{Eigen::Vector3f::Zero()};
    bool _latestTargetRawValid{false};

    std::string _paramDisarmMode{"enabled"};
    std::string _paramDisarmAltitudeSource{"dist_bottom"};
    float _paramDisarmHeight{0.06f};
    float _paramDisarmLateralErrorThreshold{0.10f};
    float _paramDisarmVerticalSpeedThreshold{0.15f};
    bool _paramDisarmAllowLandedImmediate{true};
};