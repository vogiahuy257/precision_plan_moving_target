#include <cmath>
#include <cstdint>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"

#include "px4_msgs/msg/vehicle_imu.hpp"
#include "sensor_msgs/msg/imu.hpp"

class Px4VehicleImuBridge : public rclcpp::Node
{
public:
  Px4VehicleImuBridge()
  : Node("px4_vehicle_imu_bridge_node")
  {
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic", "/fmu/out/vehicle_imu");

    output_topic_ = this->declare_parameter<std::string>(
      "output_topic", "/imu0");

    frame_id_ = this->declare_parameter<std::string>(
      "frame_id", "imu_link");

    convert_frd_to_flu_ = this->declare_parameter<bool>(
      "convert_frd_to_flu", true);

    // false = dung PX4 timestamp_sample + ROS offset.
    // Day la mode nen dung cho VINS.
    // true = dung thoi gian ROS luc node nhan message, chi de debug.
    use_ros_receive_time_ = this->declare_parameter<bool>(
      "use_ros_receive_time", false);

    angular_velocity_cov_ = this->declare_parameter<double>(
      "angular_velocity_covariance", 1.0e-6);

    linear_acceleration_cov_ = this->declare_parameter<double>(
      "linear_acceleration_covariance", 1.0e-4);

    imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
      output_topic_, rclcpp::QoS(200).reliable());

    imu_sub_ = this->create_subscription<px4_msgs::msg::VehicleImu>(
      input_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&Px4VehicleImuBridge::imuCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "PX4 VehicleImu bridge started");
    RCLCPP_INFO(this->get_logger(), "input_topic          : %s", input_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "output_topic         : %s", output_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "frame_id             : %s", frame_id_.c_str());
    RCLCPP_INFO(this->get_logger(), "convert_frd_to_flu   : %s", convert_frd_to_flu_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "use_ros_receive_time : %s", use_ros_receive_time_ ? "true" : "false");
    RCLCPP_INFO(this->get_logger(), "NOTE: no extra IMU filtering is applied");
  }

private:
  void imuCallback(const px4_msgs::msg::VehicleImu::SharedPtr msg)
  {
    const double delta_angle_dt_s =
      static_cast<double>(msg->delta_angle_dt) * 1.0e-6;

    const double delta_velocity_dt_s =
      static_cast<double>(msg->delta_velocity_dt) * 1.0e-6;

    if (delta_angle_dt_s <= 0.0 || delta_velocity_dt_s <= 0.0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Invalid VehicleImu dt: delta_angle_dt=%u, delta_velocity_dt=%u",
        msg->delta_angle_dt,
        msg->delta_velocity_dt);
      return;
    }

    sensor_msgs::msg::Imu imu_msg;

    // ============================================================
    // Timestamp
    // ============================================================
    // VINS can timestamp IMU on dinh.
    // Khong nen dung this->now() lam timestamp mac dinh, vi DDS co the jitter/burst.
    if (use_ros_receive_time_) {
      imu_msg.header.stamp = this->now();
    } else {
      const uint64_t t_us = msg->timestamp_sample > 0 ? msg->timestamp_sample : msg->timestamp;

      if (t_us == 0) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "VehicleImu timestamp is zero");
        return;
      }

      // Bo message neu PX4 timestamp khong tang.
      if (last_px4_timestamp_us_ != 0 && t_us <= last_px4_timestamp_us_) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Non-monotonic VehicleImu timestamp: current=%llu last=%llu",
          static_cast<unsigned long long>(t_us),
          static_cast<unsigned long long>(last_px4_timestamp_us_));
        return;
      }

      last_px4_timestamp_us_ = t_us;

      const int64_t px4_time_ns = static_cast<int64_t>(t_us) * 1000LL;

      if (!timestamp_offset_initialized_) {
        const int64_t ros_now_ns = this->now().nanoseconds();
        px4_to_ros_offset_ns_ = ros_now_ns - px4_time_ns;
        timestamp_offset_initialized_ = true;

        RCLCPP_INFO(
          this->get_logger(),
          "PX4->ROS timestamp offset initialized: offset_ns=%lld",
          static_cast<long long>(px4_to_ros_offset_ns_));
      }

      const int64_t stamp_ns = px4_to_ros_offset_ns_ + px4_time_ns;

      if (stamp_ns <= 0) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(),
          *this->get_clock(),
          2000,
          "Invalid converted ROS timestamp: stamp_ns=%lld",
          static_cast<long long>(stamp_ns));
        return;
      }

      imu_msg.header.stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
      imu_msg.header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
    }

    imu_msg.header.frame_id = frame_id_;

    // ============================================================
    // Convert PX4 delta measurement to standard IMU measurement
    // ============================================================
    double wx = static_cast<double>(msg->delta_angle[0]) / delta_angle_dt_s;
    double wy = static_cast<double>(msg->delta_angle[1]) / delta_angle_dt_s;
    double wz = static_cast<double>(msg->delta_angle[2]) / delta_angle_dt_s;

    double ax = static_cast<double>(msg->delta_velocity[0]) / delta_velocity_dt_s;
    double ay = static_cast<double>(msg->delta_velocity[1]) / delta_velocity_dt_s;
    double az = static_cast<double>(msg->delta_velocity[2]) / delta_velocity_dt_s;

    // PX4 body frame: FRD = x front, y right, z down
    // ROS body frame: FLU = x front, y left, z up
    if (convert_frd_to_flu_) {
      wy = -wy;
      wz = -wz;

      ay = -ay;
      az = -az;
    }

    imu_msg.angular_velocity.x = wx;
    imu_msg.angular_velocity.y = wy;
    imu_msg.angular_velocity.z = wz;

    imu_msg.linear_acceleration.x = ax;
    imu_msg.linear_acceleration.y = ay;
    imu_msg.linear_acceleration.z = az;

    // Orientation khong co trong VehicleImu.
    imu_msg.orientation_covariance[0] = -1.0;

    imu_msg.angular_velocity_covariance[0] = angular_velocity_cov_;
    imu_msg.angular_velocity_covariance[4] = angular_velocity_cov_;
    imu_msg.angular_velocity_covariance[8] = angular_velocity_cov_;

    imu_msg.linear_acceleration_covariance[0] = linear_acceleration_cov_;
    imu_msg.linear_acceleration_covariance[4] = linear_acceleration_cov_;
    imu_msg.linear_acceleration_covariance[8] = linear_acceleration_cov_;

    imu_pub_->publish(imu_msg);
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string frame_id_;

  bool convert_frd_to_flu_{true};
  bool use_ros_receive_time_{false};

  bool timestamp_offset_initialized_{false};
  int64_t px4_to_ros_offset_ns_{0};
  uint64_t last_px4_timestamp_us_{0};

  double angular_velocity_cov_{1.0e-6};
  double linear_acceleration_cov_{1.0e-4};

  rclcpp::Subscription<px4_msgs::msg::VehicleImu>::SharedPtr imu_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Px4VehicleImuBridge>());
  rclcpp::shutdown();
  return 0;
}