#include "GimbalController.hpp"

#include <functional>
#include <memory>
#include <cmath>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

constexpr char GimbalController::kGzImuTopic[];
constexpr char GimbalController::kStateAttTopic[];

namespace
{
inline double rad2deg(double rad)
{
    return rad * 180.0 / M_PI;
}
}

GimbalController::GimbalController()
: Node("gimbal_controller")
{
    auto qos = rclcpp::SensorDataQoS();

    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        kGzImuTopic,
        qos,
        std::bind(&GimbalController::imuCallback, this, std::placeholders::_1));

    att_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
        kStateAttTopic,
        qos);

    RCLCPP_INFO(this->get_logger(), "Subscribed GZ IMU topic: %s", kGzImuTopic);
    RCLCPP_INFO(this->get_logger(), "Publishing attitude topic: %s", kStateAttTopic);
}

void GimbalController::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    // Quaternion từ Gazebo IMU theo FLU
    tf2::Quaternion q_gimbal_flu(
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z,
        msg->orientation.w);

    // Giống logic PX4 GZGimbal: FLU -> FRD
    // matrix::Quatf(0, 1, 0, 0) nghĩa là w=0, x=1, y=0, z=0
    // tf2::Quaternion constructor: (x, y, z, w)
    tf2::Quaternion q_flu_to_frd(1.0, 0.0, 0.0, 0.0);

    tf2::Quaternion q_gimbal_frd =
        q_flu_to_frd * q_gimbal_flu * q_flu_to_frd.inverse();

    q_gimbal_frd.normalize();

    double roll_rad = 0.0;
    double pitch_rad = 0.0;
    double yaw_rad = 0.0;
    tf2::Matrix3x3(q_gimbal_frd).getRPY(roll_rad, pitch_rad, yaw_rad);

    // PHẢI khớp với KalmanFilter:
    // x = yaw (deg)
    // y = pitch (deg)
    // z = roll (deg)
    geometry_msgs::msg::Vector3 out_msg;
    out_msg.x = rad2deg(yaw_rad);
    out_msg.y = rad2deg(pitch_rad);
    out_msg.z = rad2deg(roll_rad);

    att_pub_->publish(out_msg);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GimbalController>());
    rclcpp::shutdown();
    return 0;
}