#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
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

    // KF logger callbacks.
    void rawPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void estimatePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void velocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void covarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
    void processNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    // TargetDrop controller evaluation callbacks.
    void controlErrorCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);
    void controlOutputCallback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

    void startRecording();
    void stopRecording();

    void addKfRow();
    void tryAddControlRow();

    void writeKfBuffer();
    void writeControlBuffer();

    static std::int64_t stampNs(const builtin_interfaces::msg::Time &stamp);
    static std::string makeFileName(const std::string &suffix);

    std::string kfOutputDir_;
    std::string controllerOutputDir_;
    bool recording_{false};

    // KF cache.
    geometry_msgs::msg::PoseStamped rawPose_{};
    geometry_msgs::msg::PoseStamped estimatePose_{};
    geometry_msgs::msg::PoseStamped velocity_{};
    std::vector<double> covariance_{};
    double qX_{0.0};
    double qY_{0.0};
    std::int64_t lastKfStampNs_{0};

    // Controller cache.
    geometry_msgs::msg::Vector3Stamped controlError_{};
    geometry_msgs::msg::Vector3Stamped controlOutput_{};
    std::int64_t lastControlStampNs_{0};

    std::ofstream kfFile_{};
    std::ofstream controllerFile_{};
    std::vector<std::string> kfBuffer_{};
    std::vector<std::string> controllerBuffer_{};

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr enableSub_;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr rawPoseSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr estimatePoseSub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr velocitySub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr covarianceSub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr processNoiseSub_;

    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr controlErrorSub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr controlOutputSub_;
};
