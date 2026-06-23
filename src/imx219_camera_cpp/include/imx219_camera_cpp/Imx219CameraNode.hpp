#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

#include <opencv2/opencv.hpp>

class Imx219CameraNode : public rclcpp::Node
{
public:
    explicit Imx219CameraNode();
    ~Imx219CameraNode() override;

private:
    void declareParameters();
    void loadParameters();
    void setupPublishers();
    void setupCameraInfo();
    void openCamera();
    void publishFrame();

    sensor_msgs::msg::Image makeImageMsg(
        const cv::Mat &frame,
        const rclcpp::Time &stamp) const;

private:
    int width_{1280};
    int height_{720};
    double fps_{30.0};

    std::string output_encoding_{"bgr8"};
    std::string frame_id_{"camera_optical_frame"};
    std::string image_topic_{"/camera/image"};
    std::string camera_info_topic_{"/camera/camera_info"};
    std::string camera_info_url_{""};
    std::string gst_pipeline_{""};

    bool publish_camera_info_{false};

    cv::VideoCapture capture_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

    std::shared_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;

    rclcpp::TimerBase::SharedPtr timer_;
};
