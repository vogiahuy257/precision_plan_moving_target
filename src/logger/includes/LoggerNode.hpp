#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class LoggerNode : public rclcpp::Node
{
public:
    LoggerNode();
    ~LoggerNode() override;

private:
    static constexpr std::size_t kBatchSize = 10;

    void enableCallback(const std_msgs::msg::Bool::SharedPtr msg);
    void rawPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void estimatePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void velocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void covarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void processNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    void startRecording();
    void stopRecording();
    void addRow();
    void writeBuffer();

    static std::int64_t stampNs(const builtin_interfaces::msg::Time &stamp);
    static std::string makeFileName();

    std::string outputDir_;
    bool recording_{false};

    geometry_msgs::msg::PoseStamped rawPose_{};
    geometry_msgs::msg::PoseStamped estimatePose_{};
    geometry_msgs::msg::PoseStamped velocity_{};
    std::vector<double> covariance_{};

    double qX_{0.0};
    double qY_{0.0};
    std::int64_t lastStampNs_{0};

    std::ofstream file_{};
    std::vector<std::string> buffer_{};

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enableSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr rawPoseSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr estimatePoseSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr velocitySub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr covarianceSub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr processNoiseSub_;
};
