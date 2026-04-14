#pragma once

#include <optional>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_command_ack.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "ControlTypes.hpp"
#include "DescentZController.hpp"
#include "PredictionModel.hpp"
#include "XYVelocityController.hpp"

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

    struct TargetWorldState
    {
        Eigen::Vector3d position{Eigen::Vector3d::Zero()};
        Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};

        rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
        rclcpp::Time velocityTimestamp{0, 0, RCL_ROS_TIME};

        bool validPose{false};
        bool validVelocity{false};
    };

private:
    /**
     * Nạp toàn bộ parameter ROS cho controller.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - declare và get toàn bộ topic/controller parameter.
     *     - cấu hình predictor, controller XY và controller Z.
     *
     * Output:
     *     Cập nhật member parameter nội bộ.
     */
    void loadParameters();

    /**
     * Callback nhận local position của UAV.
     *
     * Input:
     *     msg: px4_msgs::msg::VehicleLocalPosition::SharedPtr
     *
     * Logic:
     *     Lưu dist_bottom để debug/giám sát.
     *
     * Output:
     *     Cập nhật z_dist_bottom.
     */
    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

    /**
     * Callback trạng thái land detected từ PX4.
     *
     * Input:
     *     msg: px4_msgs::msg::VehicleLandDetected::SharedPtr
     *
     * Logic:
     *     Ghi nhận UAV đã chạm đất hay chưa.
     *
     * Output:
     *     Cập nhật _land_detected.
     */
    void vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);

    /**
     * Callback nhận attitude gimbal.
     *
     * Input:
     *     msg: geometry_msgs::msg::Vector3::SharedPtr
     *
     * Logic:
     *     - Lưu pitch gimbal.
     *     - Đánh dấu gimbal ready khi pitch gần nhìn xuống.
     *     - Tạo quaternion gimbal để dùng về sau nếu cần.
     *
     * Output:
     *     Cập nhật _gimbal_pitch_deg, _gimbal_ready, _q_gimbal.
     */
    void gimbalAttCallback(const geometry_msgs::msg::Vector3::SharedPtr msg);

    /**
     * Callback nhận pose target trong NED/world.
     *
     * Input:
     *     msg: geometry_msgs::msg::PoseStamped::SharedPtr
     *
     * Logic:
     *     Lưu pose target và timestamp tương ứng.
     *
     * Output:
     *     Cập nhật _targetWorld.position và _targetWorld.timestamp.
     */
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    /**
     * Callback nhận velocity target trong NED/world.
     *
     * Input:
     *     msg: geometry_msgs::msg::PoseStamped::SharedPtr
     *
     * Logic:
     *     Lưu velocity target và timestamp tương ứng.
     *
     * Output:
     *     Cập nhật _targetWorld.velocity và _targetWorld.velocityTimestamp.
     */
    void targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    /**
     * Hover tại chỗ khi chưa có target hoặc target bị mất.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     Gửi velocity setpoint bằng 0.
     *
     * Output:
     *     Publish setpoint hover cho PX4.
     */
    void Hover();

    /**
     * Chuyển state nội bộ của mode.
     *
     * Input:
     *     state: state mới.
     *
     * Logic:
     *     Chỉ cập nhật state hiện tại.
     *
     * Output:
     *     Cập nhật _state.
     */
    void switchToState(State state);

    /**
     * Đổi enum state ra chuỗi để log.
     *
     * Input:
     *     state: state cần chuyển.
     *
     * Logic:
     *     Trả về tên state tương ứng.
     *
     * Output:
     *     std::string tên state.
     */
    std::string stateName(State state) const;

    /**
     * Kiểm tra target timeout.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     Nếu quá lâu không có pose target mới thì xem như mất target.
     *
     * Output:
     *     true nếu target bị timeout.
     */
    bool checkTargetTimeout() const;

    /**
     * Ước lượng gia tốc XY của UAV từ sai phân vận tốc.
     *
     * Input:
     *     dt_s: chu kỳ loop hiện tại.
     *
     * Logic:
     *     - Tính sai phân vận tốc.
     *     - Clamp biên gia tốc.
     *     - Lọc low-pass để giảm nhiễu.
     *
     * Output:
     *     Eigen::Vector2f gia tốc XY đã lọc.
     */
    Eigen::Vector2f estimateVehicleAccelerationXY(float dt_s);

    /**
     * Cập nhật log target lost/acquired.
     *
     * Input:
     *     targetLost: trạng thái target hiện tại.
     *
     * Logic:
     *     So sánh với trạng thái trước đó để chỉ log khi có chuyển trạng thái.
     *
     * Output:
     *     Cập nhật _target_lost_prev.
     */
    void updateTargetLostStatus(bool targetLost);

    /**
     * Xử lý state Search.
     *
     * Input:
     *     targetLost: target có đang mất hay không.
     *
     * Logic:
     *     Nếu đã có target thì chuyển sang Descend, ngược lại hover.
     *
     * Output:
     *     Cập nhật state hoặc setpoint hover.
     */
    void handleSearchState(bool targetLost);

    /**
     * Xử lý state Descend.
     *
     * Input:
     *     dt_s: chu kỳ loop hiện tại.
     *     targetLost: target có đang mất hay không.
     *
     * Logic:
     *     - Build input cho predictor.
     *     - Tính future error.
     *     - Gọi controller XY và controller Z.
     *     - Publish setpoint cuối cùng cho PX4.
     *
     * Output:
     *     Publish trajectory setpoint mới.
     */
    void handleDescendState(float dt_s, bool targetLost);

    /**
     * Xử lý state Finished.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     Báo mode hoàn thành và điều khiển gimbal ngẩng lên.
     *
     * Output:
     *     Gửi lệnh hoàn thành mode.
     */
    void handleFinishedState();

    /**
     * Tính thời gian lead để dự đoán tương lai.
     *
     * Input:
     *     dt_s: chu kỳ loop hiện tại.
     *     ctrlStartNow: thời điểm bắt đầu tính điều khiển.
     *
     * Logic:
     *     Cộng tuổi dữ liệu pose/velocity, dt loop và lead thêm rồi clamp.
     *
     * Output:
     *     leadDtSec dùng cho predictor.
     */
    float computeLeadTimeSec(float dt_s, const rclcpp::Time &ctrlStartNow) const;

    /**
     * Build input cho predictor từ trạng thái UAV và target hiện tại.
     *
     * Input:
     *     dt_s: chu kỳ loop hiện tại.
     *     ctrlStartNow: thời điểm bắt đầu tính điều khiển.
     *
     * Logic:
     *     Gom toàn bộ dữ liệu target, vehicle, acceleration và lead time.
     *
     * Output:
     *     precision_land::PredictionInput hoàn chỉnh.
     */
    precision_land::PredictionInput buildPredictionInput(float dt_s, const rclcpp::Time &ctrlStartNow);

    /**
     * Publish target tương lai để debug.
     *
     * Input:
     *     stamp: timestamp debug.
     *     targetFutureWorld: vị trí target tương lai.
     *
     * Logic:
     *     Publish PoseStamped trên topic debug.
     *
     * Output:
     *     Publish /debug/precision_land/target_pose_pred_world.
     */
    void publishPredictedTargetDebug(const rclcpp::Time &stamp, const Eigen::Vector3f &targetFutureWorld);

    /**
     * Publish toàn bộ timing debug của controller.
     *
     * Input:
     *     ctrlStartNow: thời điểm bắt đầu tính control.
     *     ctrlEndNow: thời điểm tính control xong.
     *     cmdPubNow: thời điểm publish lệnh.
     *
     * Logic:
     *     Gọi helper publish JSON timing.
     *
     * Output:
     *     Publish /debug_dt/precision_land.
     */
    void publishTimingDebug(
        const rclcpp::Time &ctrlStartNow,
        const rclcpp::Time &ctrlEndNow,
        const rclcpp::Time &cmdPubNow);

private:
    rclcpp::Node &_node;

    std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectory_setpoint;
    std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicle_local_position;
    std::shared_ptr<px4_ros2::OdometryAttitude> _vehicle_attitude;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_sub;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _target_velocity_sub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr _vehicle_land_detected_sub;
    rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicle_local_pos_sub;
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr _gimbal_sub;

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _gimbal_seq_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _debug_target_pred_pub;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _debug_dt_pub;

    std::string _targetPoseTopic;
    std::string _targetVelocityTopic;
    std::string _vehicleLandDetectedTopic;
    std::string _vehicleLocalPositionTopic;
    std::string _gimbalCommandTopic;
    std::string _gimbalAttitudeTopic;

    float _param_pid_deadband{0.05f};
    float _param_target_timeout{3.0f};

    float _param_descent_kp{1.0f};
    float _param_descent_ki{0.0f};
    float _param_descent_kd{0.0f};
    float _param_descent_max_velocity{3.0f};
    float _param_slew_acc{2.5f};

    float _param_land_zone_z{0.5f};
    float _param_descent_vel{0.4f};

    float _param_descent_gate_radius{0.3f};
    float _param_vmin{0.45f};
    float _param_vmax{0.8f};

    bool _param_use_predictive_error{true};
    float _param_prediction_dt_max{0.5f};
    float _param_control_extra_lead_sec{0.0f};

    float _param_predictive_acc_gain{0.0f};
    float _param_predictive_acc_lpf_alpha{0.5f};
    float _param_predictive_acc_max{5.0f};

    TargetWorldState _targetWorld;

    rclcpp::Time imageTimestamp{0, 0, RCL_ROS_TIME};
    rclcpp::Time _targetPoseRxNow{0, 0, RCL_ROS_TIME};
    rclcpp::Time _targetVelRxNow{0, 0, RCL_ROS_TIME};

    State _state{State::Search};

    bool _search_started{false};
    bool _target_lost_prev{true};
    bool _land_detected{false};

    bool _yawSpInit{false};
    float _yaw_sp{0.0f};

    float _prevVehicleVelX{0.0f};
    float _prevVehicleVelY{0.0f};
    float _vehicleAccXFilt{0.0f};
    float _vehicleAccYFilt{0.0f};
    bool _prevVehicleVelValid{false};
    float _approach_altitude{0.0f};
    float z_dist_bottom{0.0f};

    float _gimbal_pitch_deg{0.0f};
    bool _gimbal_ready{false};
    bool _gimbal_valid{false};
    Eigen::Quaterniond _q_gimbal{1.0, 0.0, 0.0, 0.0};

    precision_land::PredictionModel _predictionModel;
    precision_land::XYVelocityController _xyVelocityController;
    precision_land::DescentZController _descentZController;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr vehicleCommandPub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleCommandAck>::SharedPtr vehicleCommandAckSub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr _vehicle_command_pub;
    rclcpp::Subscription<px4_msgs::msg::VehicleCommandAck>::SharedPtr _vehicle_command_ack_sub;

    bool _disarm_sent{false};

    void publishVehicleCommand(uint16_t command, float param1, float param2);
    void sendDisarmCommand();
    void vehicleCommandAckCallback(const px4_msgs::msg::VehicleCommandAck::SharedPtr msg);
    float _param_disarm_height{0.06f};

    bool _dist_bottom_valid{false};
};
