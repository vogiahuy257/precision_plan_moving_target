#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/vector3.hpp>

class GimbalController : public rclcpp::Node
{
public:
    GimbalController();

private:
    static constexpr char kGzImuTopic[] = "/world/aruco/model/x500_gimbal_0/link/camera_link/sensor/camera_imu/imu";
    static constexpr char kStateAttTopic[] = "/gimbal/state/attitude";

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr att_pub_;
};