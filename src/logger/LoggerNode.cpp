#include "LoggerNode.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

LoggerNode::LoggerNode()
    : Node("logger")
{
    outputDir_ = declare_parameter<std::string>(
        "output_dir",
        "/home/pihuy/precision_plan_moving_target/data_logs/KF");

    buffer_.reserve(kBatchSize);

    const auto dataQos = rclcpp::QoS(10).best_effort();
    const auto controlQos = rclcpp::QoS(1).reliable().transient_local();

    enableSub_ = create_subscription<std_msgs::msg::Bool>(
        "/logger/enable", controlQos,
        std::bind(&LoggerNode::enableCallback, this, std::placeholders::_1));

    rawPoseSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/KF/target_pose_NED", dataQos,
        std::bind(&LoggerNode::rawPoseCallback, this, std::placeholders::_1));

    estimatePoseSub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/KF/target_pose_est_NED", dataQos,
        std::bind(&LoggerNode::estimatePoseCallback, this, std::placeholders::_1));

    velocitySub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/KF/target_velocity_est_NED", dataQos,
        std::bind(&LoggerNode::velocityCallback, this, std::placeholders::_1));

    covarianceSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/KF/target_covariance_NE", dataQos,
        std::bind(&LoggerNode::covarianceCallback, this, std::placeholders::_1));

    processNoiseSub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        "/KF/process_noise", controlQos,
        std::bind(&LoggerNode::processNoiseCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Ready | waiting /logger/enable");
}

LoggerNode::~LoggerNode()
{
    stopRecording();
}

void LoggerNode::enableCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    if (msg->data && !recording_)
        startRecording();
    else if (!msg->data && recording_)
        stopRecording();
}

void LoggerNode::rawPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (recording_)
        rawPose_ = *msg;
}

void LoggerNode::estimatePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (recording_)
        estimatePose_ = *msg;
}

void LoggerNode::velocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (recording_)
        velocity_ = *msg;
}

void LoggerNode::covarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!recording_ || msg->data.size() < 16)
        return;

    covariance_ = msg->data;
    addRow();
}

void LoggerNode::processNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg->data.size() >= 2)
    {
        qX_ = msg->data[0];
        qY_ = msg->data[1];
    }
}

void LoggerNode::startRecording()
{
    std::filesystem::create_directories(outputDir_);
    const std::string path = outputDir_ + "/" + makeFileName();

    file_.open(path);
    if (!file_.is_open())
    {
        RCLCPP_ERROR(get_logger(), "Cannot open %s", path.c_str());
        return;
    }

    file_ << "timestamp_ns,dt_s,"
          << "meas_x,meas_y,meas_z,"
          << "est_x,est_y,est_z,"
          << "vel_x,vel_y,vel_z,"
          << "P00,P01,P02,P03,P10,P11,P12,P13,"
          << "P20,P21,P22,P23,P30,P31,P32,P33,"
          << "q_acc_x,q_acc_y\n";

    buffer_.clear();
    lastStampNs_ = 0;
    recording_ = true;

    RCLCPP_INFO(get_logger(), "Recording: %s", path.c_str());
}

void LoggerNode::stopRecording()
{
    if (!recording_ && !file_.is_open())
        return;

    recording_ = false;
    writeBuffer();

    if (file_.is_open())
    {
        file_.flush();
        file_.close();
    }

    RCLCPP_INFO(get_logger(), "Recording stopped");
}

void LoggerNode::addRow()
{
    const auto rawStamp = stampNs(rawPose_.header.stamp);
    const auto estStamp = stampNs(estimatePose_.header.stamp);
    const auto velStamp = stampNs(velocity_.header.stamp);

    if (rawStamp <= 0 || rawStamp != estStamp || rawStamp != velStamp)
        return;

    if (rawStamp == lastStampNs_)
        return;

    const double dt = lastStampNs_ > 0
        ? static_cast<double>(rawStamp - lastStampNs_) * 1e-9
        : 0.0;

    std::ostringstream row;
    row << std::setprecision(12)
        << rawStamp << ',' << dt << ','
        << rawPose_.pose.position.x << ','
        << rawPose_.pose.position.y << ','
        << rawPose_.pose.position.z << ','
        << estimatePose_.pose.position.x << ','
        << estimatePose_.pose.position.y << ','
        << estimatePose_.pose.position.z << ','
        << velocity_.pose.position.x << ','
        << velocity_.pose.position.y << ','
        << velocity_.pose.position.z;

    for (std::size_t i = 0; i < 16; ++i)
        row << ',' << covariance_[i];

    row << ',' << qX_ << ',' << qY_ << '\n';

    buffer_.push_back(row.str());
    lastStampNs_ = rawStamp;

    if (buffer_.size() >= kBatchSize)
        writeBuffer();
}

void LoggerNode::writeBuffer()
{
    if (!file_.is_open() || buffer_.empty())
        return;

    for (const auto &row : buffer_)
        file_ << row;

    file_.flush();
    buffer_.clear();
}

std::int64_t LoggerNode::stampNs(const builtin_interfaces::msg::Time &stamp)
{
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
           static_cast<std::int64_t>(stamp.nanosec);
}

std::string LoggerNode::makeFileName()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);

    std::ostringstream name;
    name << std::put_time(&tm, "%Y%m%d_%H%M%S")
         << "_kalmanfilter.csv";
    return name.str();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LoggerNode>());
    rclcpp::shutdown();
    return 0;
}
