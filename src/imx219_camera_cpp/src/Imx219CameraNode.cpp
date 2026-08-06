#include "imx219_camera_cpp/Imx219CameraNode.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <sensor_msgs/image_encodings.hpp>

using namespace std::chrono_literals;

Imx219CameraNode::Imx219CameraNode()
    : Node("imx219_camera_node")
{
    declareParameters();
    loadParameters();
    setupPublishers();
    setupCameraInfo();
    openCamera();

    const auto period = std::chrono::duration<double>(1.0 / fps_);

    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&Imx219CameraNode::publishFrame, this));

    if (publish_camera_info_) {
        RCLCPP_INFO(get_logger(), "camera_info       : enabled");
    } else {
        RCLCPP_WARN(get_logger(), "camera_info       : disabled, no valid calibration YAML");
    }
}

Imx219CameraNode::~Imx219CameraNode()
{
    if (capture_.isOpened()) {
        capture_.release();
    }
}

void Imx219CameraNode::declareParameters()
{
    declare_parameter<int>("width", 1280);
    declare_parameter<int>("height", 720);
    declare_parameter<double>("fps", 30.0);

    declare_parameter<std::string>("frame_id", "camera_optical_frame");

    declare_parameter<std::string>("image_topic", "/camera/image");
    declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

    // Dạng chuẩn camera_info_manager:
    // file:///home/pihuy/.../camera_info.yaml
    // Nếu rỗng hoặc file không tồn tại thì KHÔNG publish camera_info.
    declare_parameter<std::string>("camera_info_url", "/home/pihuy/precision_plan_moving_target/calibration/ost.yaml");

    // Pipeline để trong YAML, không hard-code trong logic node.
    declare_parameter<std::string>("gst_pipeline", "");

    // bgr8: ảnh màu OpenCV BGR
    // mono8: ảnh trắng đen grayscale
    declare_parameter<std::string>("output_encoding", "mono8");
}

void Imx219CameraNode::loadParameters()
{
    width_ = get_parameter("width").as_int();
    height_ = get_parameter("height").as_int();
    fps_ = get_parameter("fps").as_double();

    frame_id_ = get_parameter("frame_id").as_string();

    image_topic_ = get_parameter("image_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();

    camera_info_url_ = get_parameter("camera_info_url").as_string();
    gst_pipeline_ = get_parameter("gst_pipeline").as_string();
    output_encoding_ = get_parameter("output_encoding").as_string();

    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("Invalid camera width/height");
    }

    if (fps_ <= 0.0) {
        throw std::runtime_error("Invalid camera fps");
    }

    if (gst_pipeline_.empty()) {
        throw std::runtime_error("gst_pipeline parameter is empty");
    }

    if (output_encoding_ != sensor_msgs::image_encodings::BGR8 &&
        output_encoding_ != sensor_msgs::image_encodings::MONO8) {
        throw std::runtime_error("output_encoding must be either bgr8 or mono8");
    }
}

void Imx219CameraNode::setupPublishers()
{
    const auto qos = rclcpp::SensorDataQoS();

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        image_topic_,
        qos);
}

void Imx219CameraNode::setupCameraInfo()
{
    publish_camera_info_ = false;

    if (camera_info_url_.empty()) {
        return;
    }

    if (camera_info_url_.rfind("file://", 0) != 0) {
        return;
    }

    const std::string path = camera_info_url_.substr(std::string("file://").size());

    if (!std::filesystem::exists(path)) {
        return;
    }

    camera_info_manager_ =
        std::make_shared<camera_info_manager::CameraInfoManager>(
            this,
            "imx219",
            camera_info_url_);

    if (!camera_info_manager_->isCalibrated()) {
        return;
    }

    camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
        camera_info_topic_,
        rclcpp::SensorDataQoS());

    publish_camera_info_ = true;
}

void Imx219CameraNode::openCamera()
{
    capture_.open(gst_pipeline_, cv::CAP_GSTREAMER);
}

sensor_msgs::msg::Image Imx219CameraNode::makeImageMsg(
    const cv::Mat &frame,
    const rclcpp::Time &stamp) const
{
    sensor_msgs::msg::Image msg;

    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id_;

    msg.height = static_cast<uint32_t>(frame.rows);
    msg.width = static_cast<uint32_t>(frame.cols);
    msg.encoding = output_encoding_;
    msg.is_bigendian = false;
    msg.step = static_cast<uint32_t>(frame.cols * frame.elemSize());

    const size_t data_size = frame.total() * frame.elemSize();
    msg.data.resize(data_size);
    std::memcpy(msg.data.data(), frame.data, data_size);

    return msg;
}
void Imx219CameraNode::publishFrame()
{
    cv::Mat frame;

    if (!capture_.read(frame) || frame.empty()) {
        return;
    }

    cv::Mat output_frame;

    if (output_encoding_ == sensor_msgs::image_encodings::MONO8) {

        if (frame.type() == CV_8UC1) {
            // Pipeline đã xuất GRAY8 rồi.
            // Không cần convert nữa, publish thẳng mono8.
            output_frame = frame;

        } else if (frame.type() == CV_8UC3) {
            // Pipeline xuất BGR, convert sang mono8.
            cv::cvtColor(frame, output_frame, cv::COLOR_BGR2GRAY);

        } else {
            return;
        }

    } else if (output_encoding_ == sensor_msgs::image_encodings::BGR8) {

        if (frame.type() == CV_8UC3) {
            // Pipeline đã xuất BGR rồi.
            output_frame = frame;

        } else if (frame.type() == CV_8UC1) {
            // Pipeline xuất GRAY8 nhưng user yêu cầu bgr8.
            cv::cvtColor(frame, output_frame, cv::COLOR_GRAY2BGR);

        } else {
            return;
        }

    } else {
        return;
    }

    if (!output_frame.isContinuous()) {
        output_frame = output_frame.clone();
    }

    const rclcpp::Time stamp = now();

    sensor_msgs::msg::Image image_msg = makeImageMsg(output_frame, stamp);
    image_pub_->publish(image_msg);

    if (publish_camera_info_ && camera_info_pub_ && camera_info_manager_) {
        sensor_msgs::msg::CameraInfo info_msg =
            camera_info_manager_->getCameraInfo();

        info_msg.header.stamp = stamp;
        info_msg.header.frame_id = frame_id_;

        info_msg.width = image_msg.width;
        info_msg.height = image_msg.height;

        camera_info_pub_->publish(info_msg);
    }
}