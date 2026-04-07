#include "PrecisionLand.hpp"

#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/utils/geometry.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace
{
const std::string kModeName = "PLHEOC";
constexpr bool kEnableDebugOutput = true;

// Ngưỡng cho phép lệch thời gian giữa pose và velocity từ Kalman
constexpr double kMaxPoseVelocityStampSkewSec = 0.05;
}

using namespace px4_ros2::literals;

PrecisionLand::PrecisionLand(rclcpp::Node &node)
	: ModeBase(node, kModeName),
	  _node(node)
{
	_trajectory_setpoint = std::make_shared<px4_ros2::TrajectorySetpointType>(*this);
	_vehicle_local_position = std::make_shared<px4_ros2::OdometryLocalPosition>(*this);
	_vehicle_attitude = std::make_shared<px4_ros2::OdometryAttitude>(*this);

	loadParameters();

	_target_pose_sub =
		_node.create_subscription<geometry_msgs::msg::PoseStamped>(
			_targetPoseTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::targetPoseCallback, this, std::placeholders::_1));

	_target_velocity_sub =
		_node.create_subscription<geometry_msgs::msg::PoseStamped>(
			_targetVelocityTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::targetVelocityCallback, this, std::placeholders::_1));

	_vehicle_land_detected_sub =
		_node.create_subscription<px4_msgs::msg::VehicleLandDetected>(
			_vehicleLandDetectedTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::vehicleLandDetectedCallback, this, std::placeholders::_1));

	_vehicle_local_pos_sub =
		_node.create_subscription<px4_msgs::msg::VehicleLocalPosition>(
			_vehicleLocalPositionTopic,
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLand::vehicleLocalPositionCallback, this, std::placeholders::_1));

	_gimbal_sub =
		_node.create_subscription<geometry_msgs::msg::Vector3>(
			_gimbalAttitudeTopic,
			rclcpp::QoS(10).best_effort(),
			std::bind(&PrecisionLand::gimbalAttCallback, this, std::placeholders::_1));

	_gimbal_seq_pub =
		_node.create_publisher<std_msgs::msg::String>(
			_gimbalCommandTopic,
			rclcpp::QoS(1).best_effort());

	_debug_target_measurement_pub =
		_node.create_publisher<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/target_pose_measurement_world",
			rclcpp::QoS(1).best_effort());

	_debug_target_current_pub =
		_node.create_publisher<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/target_pose_current_world",
			rclcpp::QoS(1).best_effort());

	_debug_target_pred_pub =
		_node.create_publisher<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/target_pose_pred_world",
			rclcpp::QoS(1).best_effort());

	_debug_setpoint_velocity_pub =
		_node.create_publisher<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/setpoint_velocity",
			rclcpp::QoS(1).best_effort());

	modeRequirements().manual_control = false;
}

void PrecisionLand::loadParameters()
{
	_node.declare_parameter<std::string>("topics.target_pose", "/KalmanFilter/target_pose_est_NED");
	_node.declare_parameter<std::string>("topics.target_velocity", "/KalmanFilter/target_velocity_est_NED");
	_node.declare_parameter<std::string>("topics.vehicle_land_detected", "/fmu/out/vehicle_land_detected");
	_node.declare_parameter<std::string>("topics.vehicle_local_position", "/fmu/out/vehicle_local_position");
	_node.declare_parameter<std::string>("topics.gimbal_command", "/gimbal/cmd/sequence");
	_node.declare_parameter<std::string>("topics.gimbal_attitude", "/gimbal/state/attitude");

	_node.declare_parameter<float>("PID_deadband", 0.05f);
	_node.declare_parameter<float>("target_timeout", 2.0f);

	_node.declare_parameter<float>("descent_kp_pid", 1.0f);
	_node.declare_parameter<float>("descent_ki_pid", 0.0f);
	_node.declare_parameter<float>("descent_kd_pid", 0.0f);
	_node.declare_parameter<float>("descent_max_velocity", 3.0f);
	_node.declare_parameter<float>("slew_acc", 2.0f);

	_node.declare_parameter<float>("land_zone_z", 0.2f);
	_node.declare_parameter<float>("descent_vel", 0.3f);
	
	_node.declare_parameter<float>("descent_gate_radius", 0.06f);
	_node.declare_parameter<float>("vmin", 0.12f);
	_node.declare_parameter<float>("vmax", 0.45f);

	_node.declare_parameter<bool>("use_predictive_error", true);
	_node.declare_parameter<float>("predictive_acc_gain", 1.0f);

	_node.declare_parameter<float>("prediction_dt_max", 10.0f);

	_node.get_parameter("topics.target_pose", _targetPoseTopic);
	_node.get_parameter("topics.target_velocity", _targetVelocityTopic);
	_node.get_parameter("topics.vehicle_land_detected", _vehicleLandDetectedTopic);
	_node.get_parameter("topics.vehicle_local_position", _vehicleLocalPositionTopic);
	_node.get_parameter("topics.gimbal_command", _gimbalCommandTopic);
	_node.get_parameter("topics.gimbal_attitude", _gimbalAttitudeTopic);

	_node.get_parameter("PID_deadband", _param_pid_deadband);
	_node.get_parameter("target_timeout", _param_target_timeout);

	_node.get_parameter("descent_kp_pid", _param_descent_kp);
	_node.get_parameter("descent_ki_pid", _param_descent_ki);
	_node.get_parameter("descent_kd_pid", _param_descent_kd);
	_node.get_parameter("descent_max_velocity", _param_descent_max_velocity);
	_node.get_parameter("slew_acc", _param_slew_acc);

	_node.get_parameter("land_zone_z", _param_land_zone_z);
	_node.get_parameter("descent_vel", _param_descent_vel);

	_node.get_parameter("descent_gate_radius", _param_descent_gate_radius);
	_node.get_parameter("vmin", _param_vmin);
	_node.get_parameter("vmax", _param_vmax);

	_node.get_parameter("use_predictive_error", _param_use_predictive_error);
	_node.get_parameter("predictive_acc_gain", _param_predictive_acc_gain);
	_node.get_parameter("prediction_dt_max", _param_prediction_dt_max);
}

void PrecisionLand::vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
{
	z_dist_bottom = msg->dist_bottom;
}

void PrecisionLand::vehicleLandDetectedCallback(const px4_msgs::msg::VehicleLandDetected::SharedPtr msg)
{
	_land_detected = msg->landed;
}

void PrecisionLand::gimbalAttCallback(const geometry_msgs::msg::Vector3::SharedPtr msg)
{
	_gimbal_pitch_deg = static_cast<float>(msg->y);
	_gimbal_ready = std::abs(_gimbal_pitch_deg) > 80.0f;

	const double yaw = msg->x * M_PI / 180.0;
	const double pitch = msg->y * M_PI / 180.0;
	const double roll = msg->z * M_PI / 180.0;

	_q_gimbal =
		Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
		Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
		Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

	_q_gimbal.normalize();
	_gimbal_valid = true;
}

void PrecisionLand::targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
	if (!_search_started)
	{
		return;
	}

	_targetWorld.position = Eigen::Vector3d(
		msg->pose.position.x,
		msg->pose.position.y,
		msg->pose.position.z);

	rclcpp::Time msgTimestamp = msg->header.stamp;
	if (msgTimestamp.nanoseconds() == 0)
	{
		msgTimestamp = _node.now();
	}

	_targetWorld.timestamp = msgTimestamp;
	_targetWorld.validPose = true;
}

void PrecisionLand::targetVelocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
	_targetWorld.velocity = Eigen::Vector3d(
		msg->pose.position.x,
		msg->pose.position.y,
		msg->pose.position.z);

	rclcpp::Time msgTimestamp = msg->header.stamp;
	if (msgTimestamp.nanoseconds() == 0)
	{
		msgTimestamp = _node.now();
	}

	_targetWorld.velocityTimestamp = msgTimestamp;
	_targetWorld.validVelocity = true;
}

void PrecisionLand::onActivate()
{
	_vxFilt = 0.0f;
	_vyFilt = 0.0f;
	_velXIntegral = 0.0f;
	_velYIntegral = 0.0f;

	_prevErrPredX = 0.0f;
	_prevErrPredY = 0.0f;
	_prevErrPredValid = false;

	_search_started = true;
	_yawSpInit = false;

	switchToState(State::Search);
}

void PrecisionLand::Hover()
{
	RCLCPP_INFO_THROTTLE(_node.get_logger(), *(_node.get_clock()), 2000, "Hovering...");
	_trajectory_setpoint->update(
		Eigen::Vector3f(0.0f, 0.0f, 0.0f),
		std::nullopt,
		std::nullopt);
}

void PrecisionLand::onDeactivate()
{
	_search_started = false;
}

void PrecisionLand::updateSetpoint(float dt_s)
{
	const bool targetLost = checkTargetTimeout();

	if (targetLost && !_target_lost_prev)
	{
		RCLCPP_INFO(_node.get_logger(), "Target lost (state=%s)", stateName(_state).c_str());
	}
	else if (!targetLost && _target_lost_prev)
	{
		RCLCPP_INFO(_node.get_logger(), "Target acquired");
	}
	_target_lost_prev = targetLost;

	switch (_state)
	{
	case State::Search:
	{
		if (!targetLost && _targetWorld.validPose)
		{
			switchToState(State::Descend);
			break;
		}

		Hover();
		break;
	}

	case State::Descend:
	{
		if (targetLost)
		{
			switchToState(State::Search);
			break;
		}

		const rclcpp::Time controlNow = _node.now();

		const Eigen::Vector3f dronePosition = _vehicle_local_position->positionNed();

		const Eigen::Vector3f targetPositionAtMeasurement(
			static_cast<float>(_targetWorld.position.x()),
			static_cast<float>(_targetWorld.position.y()),
			static_cast<float>(_targetWorld.position.z()));

		const double poseAgeSecRaw = (controlNow - _targetWorld.timestamp).seconds();
		const bool poseAgeValid = poseAgeSecRaw >= 0.0;

		double poseVelocitySkewSec = 1e9;
		if (_targetWorld.validVelocity)
		{
			poseVelocitySkewSec =
				std::abs((_targetWorld.velocityTimestamp - _targetWorld.timestamp).seconds());
		}

		const bool useVelocityPrediction =
			_param_use_predictive_error &&
			_targetWorld.validVelocity &&
			poseAgeValid &&
			poseVelocitySkewSec <= kMaxPoseVelocityStampSkewSec;

		Eigen::Vector3f targetVelocityWorld(0.0f, 0.0f, 0.0f);
		if (useVelocityPrediction)
		{
			targetVelocityWorld.x() = static_cast<float>(_targetWorld.velocity.x());
			targetVelocityWorld.y() = static_cast<float>(_targetWorld.velocity.y());
			targetVelocityWorld.z() = static_cast<float>(_targetWorld.velocity.z());
		}

		// Bước 1: đồng bộ target từ measurement time -> current control time
		double syncDtSec = poseAgeSecRaw;
		if (syncDtSec < 0.0)
		{
			syncDtSec = 0.0;
		}

		syncDtSec = std::min(
			syncDtSec,
			static_cast<double>(std::max(_param_prediction_dt_max, 0.0f)));

		Eigen::Vector3f targetPositionCurrent = targetPositionAtMeasurement;
		if (useVelocityPrediction)
		{
			targetPositionCurrent =
				targetPositionAtMeasurement + targetVelocityWorld * static_cast<float>(syncDtSec);
		}

		const Eigen::Vector3f relativePositionCurrent = targetPositionCurrent - dronePosition;

		const float errX = relativePositionCurrent.x();
		const float errY = relativePositionCurrent.y();

		// Bước 2: dự đoán thêm một đoạn ngắn cho control horizon
		double controlLeadDtSec = std::min(static_cast<double>(std::max(dt_s, 0.0f)),static_cast<double>(std::max(_param_prediction_dt_max, 0.0f)));

		if (!useVelocityPrediction)
		{
			controlLeadDtSec = 0.0;
		}

		const float controlLeadDt = static_cast<float>(controlLeadDtSec);

		Eigen::Vector2f errorForControl(errX, errY);
		if (useVelocityPrediction && controlLeadDt > 0.0f)
		{
			errorForControl = predictFutureRelativeErrorQuadratic(
				errX,
				errY,
				targetVelocityWorld.x(),
				targetVelocityWorld.y(),
				controlLeadDt);
		}

		_approach_altitude = std::abs(dronePosition.z());

		Eigen::Vector2f velocityXY = pidVelocityXY(
			errorForControl.x(),
			errorForControl.y(),
			dt_s);

		float vz = computeDescentVelocity(errX, errY);


		const float zDist = std::abs(z_dist_bottom);

		// Rất gần mặt đất: ngừng điều khiển XY, không ép hạ mạnh nữa
		if (zDist > 0.0f && zDist <= 0.028f)
		{
			resetXYController();
			velocityXY = Eigen::Vector2f(0.0f, 0.0f);
			vz = 0.6f;
		}

		if (!_yawSpInit)
		{
			_yaw_sp = px4_ros2::quaternionToYaw(_vehicle_attitude->attitude());
			_yawSpInit = true;
		}
		
		geometry_msgs::msg::PoseStamped debugMeasurementMsg;
		debugMeasurementMsg.header.stamp = _targetWorld.timestamp;
		debugMeasurementMsg.header.frame_id = "map";
		debugMeasurementMsg.pose.position.x = targetPositionAtMeasurement.x();
		debugMeasurementMsg.pose.position.y = targetPositionAtMeasurement.y();
		debugMeasurementMsg.pose.position.z = targetPositionAtMeasurement.z();
		debugMeasurementMsg.pose.orientation.w = 1.0;
		debugMeasurementMsg.pose.orientation.x = 0.0;
		debugMeasurementMsg.pose.orientation.y = 0.0;
		debugMeasurementMsg.pose.orientation.z = 0.0;
		_debug_target_measurement_pub->publish(debugMeasurementMsg);

		geometry_msgs::msg::PoseStamped debugCurrentMsg;
		debugCurrentMsg.header.stamp = controlNow;
		debugCurrentMsg.header.frame_id = "map";
		debugCurrentMsg.pose.position.x = targetPositionCurrent.x();
		debugCurrentMsg.pose.position.y = targetPositionCurrent.y();
		debugCurrentMsg.pose.position.z = targetPositionCurrent.z();
		debugCurrentMsg.pose.orientation.w = 1.0;
		debugCurrentMsg.pose.orientation.x = 0.0;
		debugCurrentMsg.pose.orientation.y = 0.0;
		debugCurrentMsg.pose.orientation.z = 0.0;
		_debug_target_current_pub->publish(debugCurrentMsg);

		geometry_msgs::msg::PoseStamped debugPredMsg;
		debugPredMsg.header.stamp = controlNow;
		debugPredMsg.header.frame_id = "map";
		debugPredMsg.pose.position.x = dronePosition.x() + errorForControl.x();
		debugPredMsg.pose.position.y = dronePosition.y() + errorForControl.y();
		debugPredMsg.pose.position.z = targetPositionCurrent.z();
		debugPredMsg.pose.orientation.w = 1.0;
		debugPredMsg.pose.orientation.x = 0.0;
		debugPredMsg.pose.orientation.y = 0.0;
		debugPredMsg.pose.orientation.z = 0.0;
		_debug_target_pred_pub->publish(debugPredMsg);

		geometry_msgs::msg::PoseStamped debugSetpointVelocityMsg;
		debugSetpointVelocityMsg.header.stamp = controlNow;
		debugSetpointVelocityMsg.header.frame_id = "map";
		debugSetpointVelocityMsg.pose.position.x = velocityXY.x();
		debugSetpointVelocityMsg.pose.position.y = velocityXY.y();
		debugSetpointVelocityMsg.pose.position.z = vz;
		debugSetpointVelocityMsg.pose.orientation.w = 1.0;
		debugSetpointVelocityMsg.pose.orientation.x = 0.0;
		debugSetpointVelocityMsg.pose.orientation.y = 0.0;
		debugSetpointVelocityMsg.pose.orientation.z = 0.0;
		_debug_setpoint_velocity_pub->publish(debugSetpointVelocityMsg);

		_trajectory_setpoint->update(
			Eigen::Vector3f(velocityXY.x(), velocityXY.y(), vz),
			std::nullopt,
			std::nullopt);

		if (_land_detected)
		{
			switchToState(State::Finished);
		}
		break;
	}

	case State::Finished:
	{
		RCLCPP_WARN(_node.get_logger(), "[PL] Finished");

		std_msgs::msg::String msg;
		msg.data = "CENTER_LOOKUP_FOLLOW";
		_gimbal_seq_pub->publish(msg);

		ModeBase::completed(px4_ros2::Result::Success);
		return;
	}
	}
}

void PrecisionLand::resetXYController()
{
	_velXIntegral = 0.0f;
	_velYIntegral = 0.0f;
	_prevErrPredX = 0.0f;
	_prevErrPredY = 0.0f;
	_prevErrPredValid = false;
	_vxFilt = 0.0f;
	_vyFilt = 0.0f;
}

Eigen::Vector2f PrecisionLand::predictFutureRelativeErrorQuadratic(
	float errX,
	float errY,
	float targetVelX,
	float targetVelY,
	float predictionDt) const
{
	const float dt = std::clamp(predictionDt,0.0f,std::max(_param_prediction_dt_max, 0.0f));

	const Eigen::Vector3f vehicleVelocity = _vehicle_local_position->velocityNed();
	// const Eigen::Vector3f vehicleAcceleration = _vehicle_local_position->accelerationNed();

	 float errDotX = targetVelX - vehicleVelocity.x();
	 float errDotY = targetVelY - vehicleVelocity.y();

	//  const float accScale = std::max(_param_predictive_acc_gain, 0.0f);
	//  float errDDotX = -accScale * vehicleAcceleration.x();
	//  float errDDotY = -accScale * vehicleAcceleration.y();

	 float errPredX = errX + errDotX * dt;// + 0.5f * errDDotX * dt * dt;
	 float errPredY = errY + errDotY * dt;// + 0.5f * errDDotY * dt * dt;

	return Eigen::Vector2f(errPredX, errPredY);
}

float PrecisionLand::applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dt_s) const
{
	const float dt = std::max(dt_s, 1e-3f);
	const float maxDeltaVelocity = accelLimit * dt;
	const float deltaVelocity = std::clamp(commandVelocity - previousVelocity, -maxDeltaVelocity, maxDeltaVelocity);
	return previousVelocity + deltaVelocity;
}

Eigen::Vector2f PrecisionLand::pidVelocityXY(
	float errPredX,
	float errPredY,
	float dt_s)
{
	const float dt = std::max(dt_s, 1e-3f);

	const float Xp = _param_descent_kp * errPredX;
	const float Yp = _param_descent_kp * errPredY;

	if (std::abs(errPredX) > _param_pid_deadband)
	{
		_velXIntegral += errPredX * dt;
	}
	if (std::abs(errPredY) > _param_pid_deadband)
	{
		_velYIntegral += errPredY * dt;
	}

	float Xi = 0.0f;
	float Yi = 0.0f;

	if (_param_descent_ki > 1e-6f)
	{
		const float maxIntegral = 0.15f * _param_descent_max_velocity / _param_descent_ki;
		_velXIntegral = std::clamp(_velXIntegral, -maxIntegral, maxIntegral);
		_velYIntegral = std::clamp(_velYIntegral, -maxIntegral, maxIntegral);
		Xi = _param_descent_ki * _velXIntegral;
		Yi = _param_descent_ki * _velYIntegral;
	}

	float Xd = 0.0f;
	float Yd = 0.0f;

	if (_param_descent_kd > 1e-6f && _prevErrPredValid)
	{
		const float errPredDotX = (errPredX - _prevErrPredX) / dt;
		const float errPredDotY = (errPredY - _prevErrPredY) / dt;

		Xd = _param_descent_kd * errPredDotX;
		Yd = _param_descent_kd * errPredDotY;
	}

	_prevErrPredX = errPredX;
	_prevErrPredY = errPredY;
	_prevErrPredValid = true;

	float vxCmd = Xp + Xi + Xd;
	float vyCmd = Yp + Yi + Yd;

	vxCmd = std::clamp(vxCmd, -_param_descent_max_velocity, _param_descent_max_velocity);
	vyCmd = std::clamp(vyCmd, -_param_descent_max_velocity, _param_descent_max_velocity);

	_vxFilt = applySlew(vxCmd, _vxFilt, _param_slew_acc, dt);
	_vyFilt = applySlew(vyCmd, _vyFilt, _param_slew_acc, dt);

	const float vxOut = std::clamp(_vxFilt, -_param_descent_max_velocity, _param_descent_max_velocity);
	const float vyOut = std::clamp(_vyFilt, -_param_descent_max_velocity, _param_descent_max_velocity);

	return Eigen::Vector2f(vxOut, vyOut);
}

float PrecisionLand::computeDescentVelocity(float errX, float errY)
{
	const float zDist = std::abs(z_dist_bottom);

	if (zDist < _param_land_zone_z)
	{
		return std::abs(_param_descent_vel);
	}

	const float lateralError = std::sqrt(errX * errX + errY * errY);

	if (lateralError >= _param_descent_gate_radius)
	{
		return 0.0f;
	}

	const float scale =
		1.0f - (lateralError / std::max(_param_descent_gate_radius, 1e-6f));

	const float scaleClamped = std::clamp(scale, 0.0f, 1.0f);

	return _param_vmin + (_param_vmax - _param_vmin) * scaleClamped;
}

bool PrecisionLand::checkTargetTimeout() const
{
	if (!_targetWorld.validPose)
	{
		return true;
	}

	return ((_node.now() - _targetWorld.timestamp).seconds() > _param_target_timeout);
}

std::string PrecisionLand::stateName(State state) const
{
	switch (state)
	{
	case State::Search:
		return "Search";
	case State::Descend:
		return "Descend";
	case State::Finished:
		return "Finished";
	default:
		return "Unknown";
	}
}

void PrecisionLand::switchToState(State state)
{
	_state = state;
}

int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<px4_ros2::NodeWithMode<PrecisionLand>>(kModeName, kEnableDebugOutput));
	rclcpp::shutdown();
	return 0;
}