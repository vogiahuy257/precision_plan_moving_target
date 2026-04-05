#pragma once

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_land_detected.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/odometry/attitude.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <px4_ros2/setpoint_types/trajectory.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

class PrecisionLand : public px4_ros2::ModeBase
{
public:
	explicit PrecisionLand(rclcpp::Node& node);

	void onActivate() override;
	void onDeactivate() override;
	void updateSetpoint(float dt_s) override;

private:
	struct TargetState
	{
		Eigen::Vector3d position{Eigen::Vector3d::Zero()};
		rclcpp::Time timestamp{0, 0, RCL_ROS_TIME};
		bool valid() const
		{
			return std::isfinite(position.x()) &&
			       std::isfinite(position.y()) &&
			       std::isfinite(position.z()) &&
			       timestamp.nanoseconds() > 0;
		}
	};

	enum class State
	{
		Search,
		Descend,
		Finished
	};

	void loadParameters();

	void targetRelativePositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
	void targetRelativeVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
	void vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg);
	void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg);

	void hover();

	float applyDeadband(float value, float deadband) const;
	float applySlew(float v_cmd, float v_prev, float a_lim, float dt_s) const;

	Eigen::Vector2f computeTrackVelocity(float dt_s);
	float computeDescentVelocity(float relative_radius_xy) const;

	bool checkTargetTimeout() const;
	bool isRelativeVelocityFresh() const;

	std::string stateName(State s) const;
	void switchToState(State s);

	void logCsvFlight(
		float dt_s,
		float rel_pos_x,
		float rel_pos_y,
		float rel_pos_z,
		float rel_vel_x,
		float rel_vel_y,
		float rel_vel_z,
		float radius_xy,
		float rel_speed_xy,
		float vx_cmd_raw,
		float vy_cmd_raw,
		float vx_cmd_out,
		float vy_cmd_out,
		float vz_cmd,
		bool allow_descent,
		float drone_x,
		float drone_y,
		float drone_z,
		float drone_vx,
		float drone_vy,
		float drone_vz);

	std::string makeTimestampedCsvPath();

private:
	rclcpp::Node& _node;

	std::shared_ptr<px4_ros2::TrajectorySetpointType> _trajectory_setpoint;
	std::shared_ptr<px4_ros2::OdometryLocalPosition> _vehicle_local_position;
	std::shared_ptr<px4_ros2::OdometryAttitude> _vehicle_attitude;

	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _target_relative_position_sub;
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr _target_relative_velocity_sub;
	rclcpp::Subscription<px4_msgs::msg::VehicleLandDetected>::SharedPtr _vehicle_land_detected_sub;
	rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr _vehicle_local_pos_sub;

	TargetState _target_relative_position{};
	TargetState _target_relative_velocity{};

	State _state{State::Search};

	bool _search_started{false};
	bool _target_lost_prev{false};
	bool _land_detected{false};

	float _yaw_sp{0.0f};
	bool _yaw_sp_init{false};

	float _vx_filt{0.0f};
	float _vy_filt{0.0f};

	float z_dist_bottom{0.0f};
	float _approach_altitude{0.0f};

	float _log_vx_cmd_raw{0.0f};
	float _log_vy_cmd_raw{0.0f};
	float _log_vx_out{0.0f};
	float _log_vy_out{0.0f};

	float _param_pid_deadband{0.05f};
	float _param_target_timeout{5.0f};

	float _param_descent_kp{1.5f};
	float _param_descent_ki{0.0f};
	float _param_descent_kd{0.0f};
	float _param_descent_max_velocity{3.0f};
	float _param_slew_acc{10.0f};

	float _param_land_zone_z{1.0f};
	float _param_descent_vel{0.4f};

	float _param_match_position_threshold{0.25f};
	float _param_match_velocity_threshold{0.20f};

	bool _csv_enable{false};
	std::ofstream _csv;
	std::string _csv_path;
	std::string _log_dir{"./log_precision_land"};
	bool _csv_header_written{false};
};