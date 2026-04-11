#pragma once

#include <optional>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/trajectory.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

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
    void loadParameters();

    void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);
    void vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);
    void gimbalAttCallback(const geometry_msgs::msg::Vector3::SharedPtr msg);
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);

    void Hover();
    void switchToState(State state);
    std::string stateName(State state) const;
    bool checkTargetTimeout() const;

    Eigen::Vector2f estimateVehicleAccelerationXY(float dt_s);

    Eigen::Vector2f predictFutureRelativeErrorQuadratic(
        float errX,
        float errY,
        float relativeVelX,
        float relativeVelY,
        float vehicleAccX,
        float vehicleAccY,
        float predictionDt) const;

    float applySlew(
        float commandVelocity,
        float previousVelocity,
        float accelLimit,
        float dt_s) const;

    Eigen::Vector2f pidVelocityXY(
        float errPredX,
        float errPredY,
        float dt_s);

    float computeDescentVelocity(float errX, float errY);

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

    float _vxFilt{0.0f};
    float _vyFilt{0.0f};

    float _velXIntegral{0.0f};
    float _velYIntegral{0.0f};

    float _prevErrPredX{0.0f};
    float _prevErrPredY{0.0f};
    bool _prevErrPredValid{false};

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
};