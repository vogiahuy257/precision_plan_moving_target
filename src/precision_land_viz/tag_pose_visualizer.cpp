#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <px4_msgs/msg/vehicle_local_position.hpp>

#include <gz/transport/Node.hh>
#include <gz/msgs/marker.pb.h>
#include <gz/msgs/empty.pb.h>

#include <string>
#include <cmath>

class PrecisionLandDebugVisualizer : public rclcpp::Node
{
public:
	PrecisionLandDebugVisualizer()
		: Node("precision_land_debug_visualizer")
	{
		measurementSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/target_pose_measurement_world",
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLandDebugVisualizer::measurementCallback, this, std::placeholders::_1));

		currentSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/target_pose_current_world",
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLandDebugVisualizer::currentCallback, this, std::placeholders::_1));

		predSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/target_pose_pred_world",
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLandDebugVisualizer::predCallback, this, std::placeholders::_1));

		setpointVelSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
			"/debug/precision_land/setpoint_velocity",
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLandDebugVisualizer::setpointVelCallback, this, std::placeholders::_1));

		vehicleLocalPosSub_ = create_subscription<px4_msgs::msg::VehicleLocalPosition>(
			"/fmu/out/vehicle_local_position",
			rclcpp::QoS(1).best_effort(),
			std::bind(&PrecisionLandDebugVisualizer::vehicleLocalPositionCallback, this, std::placeholders::_1));

		RCLCPP_INFO(get_logger(), "PrecisionLand debug visualizer started");
	}

private:
	void vehicleLocalPositionCallback(const px4_msgs::msg::VehicleLocalPosition::SharedPtr msg)
	{
		// Swap X/Y khi lưu để vẽ đúng trên Gazebo
		droneX_ = static_cast<double>(msg->y);
		droneY_ = static_cast<double>(msg->x);
		droneZ_ = static_cast<double>(msg->z);
		haveDronePose_ = true;
	}

	void measurementCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
	{
		publishBoxMarker("pl_measurement", 1, msg, 0.5, 0.5, 0.04, 0.0f, 1.0f, 0.0f, 0.75f);
		publishTextMarker("pl_measurement_text", 101, msg, "measurement", 0.0f, 1.0f, 0.0f, 1.0f);
	}

	void currentCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
	{
		publishBoxMarker("pl_current", 2, msg, 0.5, 0.5, 0.04, 0.0f, 0.4f, 1.0f, 0.75f);
		publishTextMarker("pl_current_text", 102, msg, "current", 0.0f, 0.4f, 1.0f, 1.0f);
	}

	void predCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
	{
		publishBoxMarker("pl_pred", 3, msg, 0.5, 0.5, 0.04, 1.0f, 1.0f, 0.0f, 0.75f);
		publishTextMarker("pl_pred_text", 103, msg, "pred", 1.0f, 1.0f, 0.0f, 1.0f);
	}

	void setpointVelCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
	{
		if (!haveDronePose_)
		{
			return;
		}

		// Swap X/Y cho vector velocity
		const double vx = msg->pose.position.y;
		const double vy = msg->pose.position.x;
		const double vz = msg->pose.position.z;

		const double norm = std::sqrt(vx * vx + vy * vy + vz * vz);
		if (norm < 1e-4)
		{
			return;
		}

		const double scale = 0.5;
		const double ex = droneX_ + vx * scale;
		const double ey = droneY_ + vy * scale;
		const double ez = droneZ_ + vz * scale;

		gz::msgs::Marker markerMsg;
		markerMsg.set_ns("pl_setpoint_vel");
		markerMsg.set_id(10);
		markerMsg.set_action(gz::msgs::Marker::ADD_MODIFY);
		markerMsg.set_type(gz::msgs::Marker::LINE_STRIP);
		markerMsg.set_visibility(gz::msgs::Marker::GUI);

		auto *p0 = markerMsg.add_point();
		p0->set_x(droneX_);
		p0->set_y(droneY_);
		p0->set_z(droneZ_);

		auto *p1 = markerMsg.add_point();
		p1->set_x(ex);
		p1->set_y(ey);
		p1->set_z(ez);

		markerMsg.mutable_material()->mutable_ambient()->set_r(1.0f);
		markerMsg.mutable_material()->mutable_ambient()->set_g(0.0f);
		markerMsg.mutable_material()->mutable_ambient()->set_b(0.0f);
		markerMsg.mutable_material()->mutable_ambient()->set_a(1.0f);

		markerMsg.mutable_material()->mutable_diffuse()->set_r(1.0f);
		markerMsg.mutable_material()->mutable_diffuse()->set_g(0.0f);
		markerMsg.mutable_material()->mutable_diffuse()->set_b(0.0f);
		markerMsg.mutable_material()->mutable_diffuse()->set_a(1.0f);

		markerMsg.set_layer(0);
		markerMsg.mutable_scale()->set_x(0.06);
		markerMsg.mutable_scale()->set_y(0.06);
		markerMsg.mutable_scale()->set_z(0.06);

		markerMsg.mutable_lifetime()->set_sec(0);
		markerMsg.mutable_lifetime()->set_nsec(0);

		callMarkerService(markerMsg);

		publishVelocityTextMarker("pl_setpoint_vel_text", 110, ex, ey, ez, "setpoint_vel", 1.0f, 0.0f, 0.0f, 1.0f);
	}

	void publishBoxMarker(
		const std::string &ns,
		int id,
		const geometry_msgs::msg::PoseStamped::SharedPtr msg,
		double sx,
		double sy,
		double sz,
		float r,
		float g,
		float b,
		float a)
	{
		gz::msgs::Marker markerMsg;
		markerMsg.set_ns(ns);
		markerMsg.set_id(id);
		markerMsg.set_action(gz::msgs::Marker::ADD_MODIFY);
		markerMsg.set_type(gz::msgs::Marker::BOX);
		markerMsg.set_visibility(gz::msgs::Marker::GUI);

		// Swap X/Y khi vẽ box
		markerMsg.mutable_pose()->mutable_position()->set_x(msg->pose.position.y);
		markerMsg.mutable_pose()->mutable_position()->set_y(msg->pose.position.x);
		markerMsg.mutable_pose()->mutable_position()->set_z(msg->pose.position.z);

		markerMsg.mutable_pose()->mutable_orientation()->set_w(msg->pose.orientation.w);
		markerMsg.mutable_pose()->mutable_orientation()->set_x(msg->pose.orientation.x);
		markerMsg.mutable_pose()->mutable_orientation()->set_y(msg->pose.orientation.y);
		markerMsg.mutable_pose()->mutable_orientation()->set_z(msg->pose.orientation.z);

		markerMsg.mutable_scale()->set_x(sx);
		markerMsg.mutable_scale()->set_y(sy);
		markerMsg.mutable_scale()->set_z(sz);

		markerMsg.mutable_material()->mutable_ambient()->set_r(r);
		markerMsg.mutable_material()->mutable_ambient()->set_g(g);
		markerMsg.mutable_material()->mutable_ambient()->set_b(b);
		markerMsg.mutable_material()->mutable_ambient()->set_a(a);

		markerMsg.mutable_material()->mutable_diffuse()->set_r(r);
		markerMsg.mutable_material()->mutable_diffuse()->set_g(g);
		markerMsg.mutable_material()->mutable_diffuse()->set_b(b);
		markerMsg.mutable_material()->mutable_diffuse()->set_a(a);

		markerMsg.mutable_lifetime()->set_sec(0);
		markerMsg.mutable_lifetime()->set_nsec(0);

		callMarkerService(markerMsg);
	}

	void publishTextMarker(
		const std::string &ns,
		int id,
		const geometry_msgs::msg::PoseStamped::SharedPtr msg,
		const std::string &text,
		float r,
		float g,
		float b,
		float a)
	{
		gz::msgs::Marker markerMsg;
		markerMsg.set_ns(ns);
		markerMsg.set_id(id);
		markerMsg.set_action(gz::msgs::Marker::ADD_MODIFY);
		markerMsg.set_type(gz::msgs::Marker::TEXT);
		markerMsg.set_visibility(gz::msgs::Marker::GUI);

		// Swap X/Y khi vẽ text
		markerMsg.mutable_pose()->mutable_position()->set_x(msg->pose.position.y);
		markerMsg.mutable_pose()->mutable_position()->set_y(msg->pose.position.x);
		markerMsg.mutable_pose()->mutable_position()->set_z(msg->pose.position.z + 0.18);

		markerMsg.mutable_pose()->mutable_orientation()->set_w(1.0);
		markerMsg.mutable_pose()->mutable_orientation()->set_x(0.0);
		markerMsg.mutable_pose()->mutable_orientation()->set_y(0.0);
		markerMsg.mutable_pose()->mutable_orientation()->set_z(0.0);

		markerMsg.set_text(text);

		markerMsg.mutable_scale()->set_x(0.18);
		markerMsg.mutable_scale()->set_y(0.18);
		markerMsg.mutable_scale()->set_z(0.18);

		markerMsg.mutable_material()->mutable_ambient()->set_r(r);
		markerMsg.mutable_material()->mutable_ambient()->set_g(g);
		markerMsg.mutable_material()->mutable_ambient()->set_b(b);
		markerMsg.mutable_material()->mutable_ambient()->set_a(a);

		markerMsg.mutable_material()->mutable_diffuse()->set_r(r);
		markerMsg.mutable_material()->mutable_diffuse()->set_g(g);
		markerMsg.mutable_material()->mutable_diffuse()->set_b(b);
		markerMsg.mutable_material()->mutable_diffuse()->set_a(a);

		markerMsg.mutable_lifetime()->set_sec(0);
		markerMsg.mutable_lifetime()->set_nsec(0);

		callMarkerService(markerMsg);
	}

	void publishVelocityTextMarker(
		const std::string &ns,
		int id,
		double x,
		double y,
		double z,
		const std::string &text,
		float r,
		float g,
		float b,
		float a)
	{
		gz::msgs::Marker markerMsg;
		markerMsg.set_ns(ns);
		markerMsg.set_id(id);
		markerMsg.set_action(gz::msgs::Marker::ADD_MODIFY);
		markerMsg.set_type(gz::msgs::Marker::TEXT);
		markerMsg.set_visibility(gz::msgs::Marker::GUI);

		markerMsg.mutable_pose()->mutable_position()->set_x(x);
		markerMsg.mutable_pose()->mutable_position()->set_y(y);
		markerMsg.mutable_pose()->mutable_position()->set_z(z + 0.12);

		markerMsg.mutable_pose()->mutable_orientation()->set_w(1.0);
		markerMsg.mutable_pose()->mutable_orientation()->set_x(0.0);
		markerMsg.mutable_pose()->mutable_orientation()->set_y(0.0);
		markerMsg.mutable_pose()->mutable_orientation()->set_z(0.0);

		markerMsg.set_text(text);

		markerMsg.mutable_scale()->set_x(0.16);
		markerMsg.mutable_scale()->set_y(0.16);
		markerMsg.mutable_scale()->set_z(0.16);

		markerMsg.mutable_material()->mutable_ambient()->set_r(r);
		markerMsg.mutable_material()->mutable_ambient()->set_g(g);
		markerMsg.mutable_material()->mutable_ambient()->set_b(b);
		markerMsg.mutable_material()->mutable_ambient()->set_a(a);

		markerMsg.mutable_material()->mutable_diffuse()->set_r(r);
		markerMsg.mutable_material()->mutable_diffuse()->set_g(g);
		markerMsg.mutable_material()->mutable_diffuse()->set_b(b);
		markerMsg.mutable_material()->mutable_diffuse()->set_a(a);

		markerMsg.mutable_lifetime()->set_sec(0);
		markerMsg.mutable_lifetime()->set_nsec(0);

		callMarkerService(markerMsg);
	}

	void callMarkerService(const gz::msgs::Marker &markerMsg)
	{
		gz::msgs::Empty response;
		bool result = false;
		const unsigned int timeoutMs = 300;

		const bool executed = gzNode_.Request("/marker", markerMsg, timeoutMs, response, result);

		if (!executed || !result)
		{
			RCLCPP_DEBUG(get_logger(), "Marker service call failed or timed out");
		}
	}

private:
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr measurementSub_;
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr currentSub_;
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr predSub_;
	rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr setpointVelSub_;
	rclcpp::Subscription<px4_msgs::msg::VehicleLocalPosition>::SharedPtr vehicleLocalPosSub_;

	gz::transport::Node gzNode_;

	double droneX_{0.0};
	double droneY_{0.0};
	double droneZ_{0.0};
	bool haveDronePose_{false};
};

int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<PrecisionLandDebugVisualizer>());
	rclcpp::shutdown();
	return 0;
}