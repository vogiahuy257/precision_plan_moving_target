#include "PrecisionLand.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/utils/geometry.hpp>

namespace
{
const std::string kModeName = "PLHEOC";
constexpr bool kEnableDebugOutput = true;

constexpr char kTargetRelativePositionTopic[] = "/target_fusion/relative_position_predicted";
constexpr char kTargetRelativeVelocityTopic[] = "/target_fusion/relative_velocity";
constexpr char kVehicleLandDetectedTopic[] = "/fmu/out/vehicle_land_detected";
constexpr char kVehicleLocalPositionTopic[] = "/fmu/out/vehicle_local_position";
}

using namespace px4_ros2::literals;

PrecisionLand::PrecisionLand(rclcpp::Node& node)
	: ModeBase(node, kModeName),
	  _node(node)
{
	_trajectory_setpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
	_vehicle_local_position = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
	_vehicle_attitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);

	_target_relative_position_sub =
		_node.create_subscription<geometry_msgs::msg::PoseStamped>(
			kTargetRelativePositionTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::targetRelativePositionCallback, this, std::placeholders::_1));

	_target_relative_velocity_sub =
		_node.create_subscription<geometry_msgs::msg::PoseStamped>(
			kTargetRelativeVelocityTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::targetRelativeVelocityCallback, this, std::placeholders::_1));

	_vehicle_land_detected_sub =
		_node.create_subscription<px4_msgs::msg::VehicleLandDetected>(
			kVehicleLandDetectedTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::vehicleLandDetectedCallback, this, std::placeholders::_1));

	_vehicle_local_pos_sub =
		_node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
			kVehicleLocalPositionTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::vehicleLocalPositionCallback, this, std::placeholders::_1));

	loadParameters();
	modeRequirements().manual_control = false;
}

void PrecisionLand::loadParameters()
{
	_node.declare_parameter<float>("PID_deadband", 0.05f);
	_node.declare_parameter<float>("target_timeout", 5.0f);

	_node.declare_parameter<float>("descent_kp_pid", 1.5f);
	_node.declare_parameter<float>("descent_ki_pid", 0.0f);
	_node.declare_parameter<float>("descent_kd_pid", 0.8f);
	_node.declare_parameter<float>("descent_max_velocity", 3.0f);
	_node.declare_parameter<float>("slew_acc", 10.0f);

	_node.declare_parameter<float>("land_zone_z", 1.0f);
	_node.declare_parameter<float>("descent_vel", 0.4f);

	_node.declare_parameter<float>("match_position_threshold", 0.25f);
	_node.declare_parameter<float>("match_velocity_threshold", 0.20f);

	_node.get_parameter("PID_deadband", _param_pid_deadband);
	_node.get_parameter("target_timeout", _param_target_timeout);

	_node.get_parameter("descent_kp_pid", _param_descent_kp);
	_node.get_parameter("descent_ki_pid", _param_descent_ki);
	_node.get_parameter("descent_kd_pid", _param_descent_kd);
	_node.get_parameter("descent_max_velocity", _param_descent_max_velocity);
	_node.get_parameter("slew_acc", _param_slew_acc);

	_node.get_parameter("land_zone_z", _param_land_zone_z);
	_node.get_parameter("descent_vel", _param_descent_vel);

	_node.get_parameter("match_position_threshold", _param_match_position_threshold);
	_node.get_parameter("match_velocity_threshold", _param_match_velocity_threshold);
}

void PrecisionLand::targetRelativePositionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
	if (!_search_started) {
		return;
	}

	_target_relative_position.position = Eigen::Vector3d(
		msg->pose.position.x,
		msg->pose.position.y,
		msg->pose.position.z);

	rclcpp::Time msg_ts = msg->header.stamp;
	if (msg_ts.nanoseconds() == 0) {
		msg_ts = _node.now();
	}
	_target_relative_position.timestamp = msg_ts;
}

void PrecisionLand::targetRelativeVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
	if (!_search_started) {
		return;
	}

	_target_relative_velocity.position = Eigen::Vector3d(
		msg->pose.position.x,
		msg->pose.position.y,
		msg->pose.position.z);

	rclcpp::Time msg_ts = msg->header.stamp;
	if (msg_ts.nanoseconds() == 0) {
		msg_ts = _node.now();
	}
	_target_relative_velocity.timestamp = msg_ts;
}

void PrecisionLand::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
	z_dist_bottom = msg->dist_bottom;
}

void PrecisionLand::vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg)
{
	_land_detected = msg->landed;
}

void PrecisionLand::onActivate()
{
	_vx_filt = 0.0f;
	_vy_filt = 0.0f;

	_yaw_sp_init = false;
	_search_started = true;
	_target_lost_prev = false;
	_land_detected = false;

	if (_csv_enable) {
		if (_csv.is_open()) {
			_csv.close();
		}

		_csv_path = makeTimestampedCsvPath();
		if (!_csv_path.empty()) {
			_csv.open(_csv_path, std::ios::out);
			_csv_header_written = false;
		}
	}

	switchToState(State::Search);
}

void PrecisionLand::hover()
{
	RCLCPP_INFO_THROTTLE(_node.get_logger(), *(_node.get_clock()), 2000, "Hovering...");
	_trajectory_setpoint->update(Eigen::Vector3f(0.0f, 0.0f, 0.0f), std::nullopt, std::nullopt);
}

void PrecisionLand::onDeactivate()
{
	_search_started = false;

	if (_csv.is_open()) {
		_csv.flush();
		_csv.close();
	}
}

void PrecisionLand::updateSetpoint(float dt_s)
{
	if (!_yaw_sp_init) {
		_yaw_sp = px4_ros2::quaternionToYaw(_vehicle_attitude->attitude());
		_yaw_sp_init = true;
	}

	const bool target_lost = checkTargetTimeout();

	if (target_lost && !_target_lost_prev) {
		RCLCPP_INFO(_node.get_logger(), "Target lost (state=%s)", stateName(_state).c_str());
	} else if (!target_lost && _target_lost_prev) {
		RCLCPP_INFO(_node.get_logger(), "Target acquired");
	}
	_target_lost_prev = target_lost;

	switch (_state) {
	case State::Search: {
		if (!target_lost) {
			switchToState(State::Descend);
			break;
		}

		hover();
		break;
	}

	case State::Descend: {
		if (target_lost) {
			switchToState(State::Search);
			break;
		}

		const Eigen::Vector3f p = _vehicle_local_position->positionNed();
		const Eigen::Vector3f v = _vehicle_local_position->velocityNed();

		const float rel_pos_x = static_cast<float>(_target_relative_position.position.x());
		const float rel_pos_y = static_cast<float>(_target_relative_position.position.y());
		const float rel_pos_z = static_cast<float>(_target_relative_position.position.z());

		const float rel_vel_x = static_cast<float>(_target_relative_velocity.position.x());
		const float rel_vel_y = static_cast<float>(_target_relative_velocity.position.y());
		const float rel_vel_z = static_cast<float>(_target_relative_velocity.position.z());

		const float radius_xy = std::sqrt(rel_pos_x * rel_pos_x + rel_pos_y * rel_pos_y);
		const float rel_speed_xy = std::sqrt(rel_vel_x * rel_vel_x + rel_vel_y * rel_vel_y);

		_approach_altitude = std::abs(p.z());

		const Eigen::Vector2f vel_xy = computeTrackVelocity(dt_s);

		const bool allow_descent =
			(radius_xy < _param_match_position_threshold) &&
			(rel_speed_xy < _param_match_velocity_threshold);

		const float vz = allow_descent ? computeDescentVelocity(radius_xy) : 0.0f;

		_trajectory_setpoint->update(
			Eigen::Vector3f(vel_xy.x(), vel_xy.y(), vz),
			std::nullopt,
			_yaw_sp);

		RCLCPP_INFO_THROTTLE(
			_node.get_logger(),
			*(_node.get_clock()),
			1000,
			"[PL] r=%.3f | vrel=%.3f | vx=%.3f | vy=%.3f | vz=%.3f | allow_descent=%d",
			radius_xy,
			rel_speed_xy,
			vel_xy.x(),
			vel_xy.y(),
			vz,
			static_cast<int>(allow_descent));

		logCsvFlight(
			dt_s,
			rel_pos_x, rel_pos_y, rel_pos_z,
			rel_vel_x, rel_vel_y, rel_vel_z,
			radius_xy,
			rel_speed_xy,
			_log_vx_cmd_raw, _log_vy_cmd_raw,
			_log_vx_out, _log_vy_out,
			vz,
			allow_descent,
			p.x(), p.y(), p.z(),
			v.x(), v.y(), v.z());

		if (_land_detected) {
			switchToState(State::Finished);
		}
		break;
	}

	case State::Finished: {
		RCLCPP_WARN(_node.get_logger(), "[PL] Finished");
		ModeBase::completed(px4_ros2::Result::Success);
		return;
	}
	}
}

float PrecisionLand::applyDeadband(float value, float deadband) const
{
	if (std::abs(value) < deadband) {
		return 0.0f;
	}
	return value;
}

float PrecisionLand::applySlew(float v_cmd, float v_prev, float a_lim, float dt_s) const
{
	const float dt = std::max(dt_s, 1e-3f);
	const float dv_max = a_lim * dt;
	const float dv = std::clamp(v_cmd - v_prev, -dv_max, dv_max);
	return v_prev + dv;
}

Eigen::Vector2f PrecisionLand::computeTrackVelocity(float dt_s)
{
	const float rel_pos_x = applyDeadband(
		static_cast<float>(_target_relative_position.position.x()),
		_param_pid_deadband);

	const float rel_pos_y = applyDeadband(
		static_cast<float>(_target_relative_position.position.y()),
		_param_pid_deadband);

	const float rel_vel_x = applyDeadband(
		static_cast<float>(_target_relative_velocity.position.x()),
		_param_pid_deadband);

	const float rel_vel_y = applyDeadband(
		static_cast<float>(_target_relative_velocity.position.y()),
		_param_pid_deadband);

	float vx_cmd =
		_param_descent_kp * rel_pos_x +
		_param_descent_kd * rel_vel_x;

	float vy_cmd =
		_param_descent_kp * rel_pos_y +
		_param_descent_kd * rel_vel_y;

	vx_cmd = std::clamp(vx_cmd, -_param_descent_max_velocity, _param_descent_max_velocity);
	vy_cmd = std::clamp(vy_cmd, -_param_descent_max_velocity, _param_descent_max_velocity);

	_vx_filt = applySlew(vx_cmd, _vx_filt, _param_slew_acc, dt_s);
	_vy_filt = applySlew(vy_cmd, _vy_filt, _param_slew_acc, dt_s);

	const float vx_out = std::clamp(_vx_filt, -_param_descent_max_velocity, _param_descent_max_velocity);
	const float vy_out = std::clamp(_vy_filt, -_param_descent_max_velocity, _param_descent_max_velocity);

	_log_vx_cmd_raw = vx_cmd;
	_log_vy_cmd_raw = vy_cmd;
	_log_vx_out = vx_out;
	_log_vy_out = vy_out;

	return Eigen::Vector2f(vx_out, vy_out);
}

float PrecisionLand::computeDescentVelocity(float relative_radius_xy) const
{
	const float z_dist = std::abs(z_dist_bottom);

	if (z_dist < _param_land_zone_z) {
		return std::abs(_param_descent_vel);
	}

	const float radius_ratio = std::clamp(
		relative_radius_xy / std::max(_param_match_position_threshold, 1e-3f),
		0.0f,
		1.0f);

	const float scale = 1.0f - radius_ratio;
	return std::abs(_param_descent_vel) * scale;
}

bool PrecisionLand::checkTargetTimeout() const
{
	if (!_target_relative_position.valid()) {
		return true;
	}

	const double target_age_s = (_node.now() - _target_relative_position.timestamp).seconds();
	if (target_age_s > _param_target_timeout) {
		return true;
	}

	return !isRelativeVelocityFresh();
}

bool PrecisionLand::isRelativeVelocityFresh() const
{
	if (!_target_relative_velocity.valid()) {
		return false;
	}

	const double velocity_age_s = (_node.now() - _target_relative_velocity.timestamp).seconds();
	return velocity_age_s <= _param_target_timeout;
}

std::string PrecisionLand::stateName(State s) const
{
	switch (s) {
	case State::Search:   return "Search";
	case State::Descend:  return "Descend";
	case State::Finished: return "Finished";
	default:              return "Unknown";
	}
}

void PrecisionLand::switchToState(State s)
{
	_state = s;
}

void PrecisionLand::logCsvFlight(
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
	float drone_vz)
{
	if (!_csv_enable || !_csv.is_open()) {
		return;
	}

	const double t = _node.now().seconds();
	const bool target_lost = checkTargetTimeout();
	const double target_age_s = _target_relative_position.valid()
		? (_node.now() - _target_relative_position.timestamp).seconds()
		: -1.0;

	if (!_csv_header_written) {
		_csv
			<< "t_sec,dt_s,state,target_lost,target_age_s,landed,"
			<< "rel_pos_x,rel_pos_y,rel_pos_z,"
			<< "rel_vel_x,rel_vel_y,rel_vel_z,"
			<< "radius_xy,rel_speed_xy,"
			<< "vx_cmd_raw,vy_cmd_raw,vx_cmd_out,vy_cmd_out,vz_cmd,"
			<< "allow_descent,"
			<< "drone_x,drone_y,drone_z,"
			<< "drone_vx,drone_vy,drone_vz,"
			<< "dist_bottom,approach_altitude"
			<< "\n";

		_csv_header_written = true;
	}

	_csv
		<< t << "," << dt_s << ","
		<< stateName(_state) << ","
		<< (target_lost ? 1 : 0) << ","
		<< target_age_s << ","
		<< (_land_detected ? 1 : 0) << ","
		<< rel_pos_x << "," << rel_pos_y << "," << rel_pos_z << ","
		<< rel_vel_x << "," << rel_vel_y << "," << rel_vel_z << ","
		<< radius_xy << "," << rel_speed_xy << ","
		<< vx_cmd_raw << "," << vy_cmd_raw << ","
		<< vx_cmd_out << "," << vy_cmd_out << "," << vz_cmd << ","
		<< (allow_descent ? 1 : 0) << ","
		<< drone_x << "," << drone_y << "," << drone_z << ","
		<< drone_vx << "," << drone_vy << "," << drone_vz << ","
		<< z_dist_bottom << "," << _approach_altitude
		<< "\n";
}

std::string PrecisionLand::makeTimestampedCsvPath()
{
	namespace fs = std::filesystem;

	try {
		if (!fs::exists(_log_dir)) {
			fs::create_directories(_log_dir);
		}
	} catch (const std::exception& e) {
		RCLCPP_ERROR(_node.get_logger(), "Cannot create log dir '%s': %s", _log_dir.c_str(), e.what());
		return "";
	}

	const auto now_tp = std::chrono::system_clock::now();
	const std::time_t now_c = std::chrono::system_clock::to_time_t(now_tp);

	std::tm tm{};
#ifdef _WIN32
	localtime_s(&tm, &now_c);
#else
	localtime_r(&now_c, &tm);
#endif

	std::ostringstream ss;
	ss << "precision_land_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";

	return (std::filesystem::path(_log_dir) / ss.str()).string();
}

int main(int argc, char* argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<PrecisionLand>>(kModeName, kEnableDebugOutput));
	rclcpp::shutdown();
	return 0;
}