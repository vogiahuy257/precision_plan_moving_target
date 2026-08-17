#include "LoggerNode.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <sstream>

LoggerNode::LoggerNode()
    : Node("logger")
{
    kfOutputDir_ = declare_parameter<std::string>(
        "output_dir",
        "/home/pihuy/precision_plan_moving_target/data_logs/KF");

    controllerOutputDir_ = declare_parameter<std::string>(
        "controller_output_dir",
        "/home/pihuy/precision_plan_moving_target/data_logs/TargetDrop");

    kfBuffer_.reserve(kBatchSize);
    controllerBuffer_.reserve(kBatchSize);

    const auto dataQos = rclcpp::QoS(10).best_effort();
    const auto controlQos = rclcpp::QoS(1).reliable().transient_local();
    const auto controllerQos = rclcpp::QoS(50).reliable();

    enableSub_ = create_subscription<std_msgs::msg::Bool>(
        "/logger/enable", controlQos,
        std::bind(&LoggerNode::enableCallback, this, std::placeholders::_1));

    // Existing KF data.
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

    // TargetDrop controller evaluation data.
    controlErrorSub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/TargetDrop/control_error", controllerQos,
        std::bind(&LoggerNode::controlErrorCallback, this, std::placeholders::_1));

    controlOutputSub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
        "/TargetDrop/control_output", controllerQos,
        std::bind(&LoggerNode::controlOutputCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Ready | waiting /logger/enable");
}

LoggerNode::~LoggerNode()
{
    stopRecording();
}

void LoggerNode::enableCallback(const std_msgs::msg::Bool::SharedPtr msg)
{
    if (msg == nullptr)
    {
        return;
    }

    if (msg->data && !recording_)
    {
        startRecording();
    }
    else if (!msg->data && recording_)
    {
        stopRecording();
    }
}

void LoggerNode::rawPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (recording_ && msg != nullptr)
    {
        rawPose_ = *msg;
    }
}

void LoggerNode::estimatePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (recording_ && msg != nullptr)
    {
        estimatePose_ = *msg;
    }
}

void LoggerNode::velocityCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    if (recording_ && msg != nullptr)
    {
        velocity_ = *msg;
    }
}

void LoggerNode::covarianceCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!recording_ || msg == nullptr || msg->data.size() < 16)
    {
        return;
    }

    covariance_ = msg->data;
    addKfRow();
}

void LoggerNode::processNoiseCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (msg != nullptr && msg->data.size() >= 2)
    {
        qX_ = msg->data[0];
        qY_ = msg->data[1];
    }
}

void LoggerNode::controlErrorCallback(
    const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    if (!recording_ || msg == nullptr)
    {
        return;
    }

    controlError_ = *msg;
    tryAddControlRow();
}

void LoggerNode::controlOutputCallback(
    const geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
{
    if (!recording_ || msg == nullptr)
    {
        return;
    }

    controlOutput_ = *msg;
    tryAddControlRow();
}

void LoggerNode::startRecording()
{
    std::filesystem::create_directories(kfOutputDir_);
    std::filesystem::create_directories(controllerOutputDir_);

    const std::string kfPath =
        kfOutputDir_ + "/" + makeFileName("kalmanfilter");

    const std::string controllerPath =
        controllerOutputDir_ + "/" + makeFileName("targetdrop_controller");

    kfFile_.open(kfPath);
    controllerFile_.open(controllerPath);

    if (!kfFile_.is_open() || !controllerFile_.is_open())
    {
        RCLCPP_ERROR(
            get_logger(),
            "Cannot open logger files | KF=%s | Controller=%s",
            kfPath.c_str(),
            controllerPath.c_str());

        if (kfFile_.is_open())
        {
            kfFile_.close();
        }

        if (controllerFile_.is_open())
        {
            controllerFile_.close();
        }

        return;
    }

    kfFile_ << "timestamp_ns,dt_s,"
            << "meas_x,meas_y,meas_z,"
            << "est_x,est_y,est_z,"
            << "vel_x,vel_y,vel_z,"
            << "P00,P01,P02,P03,P10,P11,P12,P13,"
            << "P20,P21,P22,P23,P30,P31,P32,P33,"
            << "q_acc_x,q_acc_y\n";

    // Only the two controller data groups requested for response evaluation:
    // tracking error and final controller output.
    controllerFile_
        << "timestamp_ns,"
        << "error_x_m,error_y_m,error_z_m,"
        << "output_vx_m_s,output_vy_m_s,output_vd_m_s\n";

    kfBuffer_.clear();
    controllerBuffer_.clear();

    lastKfStampNs_ = 0;
    lastControlStampNs_ = 0;

    rawPose_ = {};
    estimatePose_ = {};
    velocity_ = {};
    covariance_.clear();
    controlError_ = {};
    controlOutput_ = {};

    recording_ = true;

    RCLCPP_INFO(get_logger(), "KF recording: %s", kfPath.c_str());
    RCLCPP_INFO(get_logger(), "Controller recording: %s", controllerPath.c_str());
}

void LoggerNode::stopRecording()
{
    if (!recording_ && !kfFile_.is_open() && !controllerFile_.is_open())
    {
        return;
    }

    recording_ = false;

    writeKfBuffer();
    writeControlBuffer();

    if (kfFile_.is_open())
    {
        kfFile_.flush();
        kfFile_.close();
    }

    if (controllerFile_.is_open())
    {
        controllerFile_.flush();
        controllerFile_.close();
    }

    RCLCPP_INFO(get_logger(), "Recording stopped");
}

void LoggerNode::addKfRow()
{
    if (!kfFile_.is_open() || covariance_.size() < 16)
    {
        return;
    }

    const auto rawStamp = stampNs(rawPose_.header.stamp);
    const auto estStamp = stampNs(estimatePose_.header.stamp);
    const auto velStamp = stampNs(velocity_.header.stamp);

    if (rawStamp <= 0 || rawStamp != estStamp || rawStamp != velStamp)
    {
        return;
    }

    if (rawStamp == lastKfStampNs_)
    {
        return;
    }

    const double dt = lastKfStampNs_ > 0
        ? static_cast<double>(rawStamp - lastKfStampNs_) * 1e-9
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
    {
        row << ',' << covariance_[i];
    }

    row << ',' << qX_ << ',' << qY_ << '\n';

    kfBuffer_.push_back(row.str());
    lastKfStampNs_ = rawStamp;

    if (kfBuffer_.size() >= kBatchSize)
    {
        writeKfBuffer();
    }
}

void LoggerNode::tryAddControlRow()
{
    if (!controllerFile_.is_open())
    {
        return;
    }

    const auto errorStamp = stampNs(controlError_.header.stamp);
    const auto outputStamp = stampNs(controlOutput_.header.stamp);

    // TargetDrop publishes both messages with the same control-loop timestamp.
    if (errorStamp <= 0 || errorStamp != outputStamp)
    {
        return;
    }

    if (errorStamp == lastControlStampNs_)
    {
        return;
    }

    std::ostringstream row;
    row << std::setprecision(12)
        << errorStamp << ','
        << controlError_.vector.x << ','
        << controlError_.vector.y << ','
        << controlError_.vector.z << ','
        << controlOutput_.vector.x << ','
        << controlOutput_.vector.y << ','
        << controlOutput_.vector.z << '\n';

    controllerBuffer_.push_back(row.str());
    lastControlStampNs_ = errorStamp;

    if (controllerBuffer_.size() >= kBatchSize)
    {
        writeControlBuffer();
    }
}

void LoggerNode::writeKfBuffer()
{
    if (!kfFile_.is_open() || kfBuffer_.empty())
    {
        return;
    }

    for (const auto &row : kfBuffer_)
    {
        kfFile_ << row;
    }

    kfFile_.flush();
    kfBuffer_.clear();
}

void LoggerNode::writeControlBuffer()
{
    if (!controllerFile_.is_open() || controllerBuffer_.empty())
    {
        return;
    }

    for (const auto &row : controllerBuffer_)
    {
        controllerFile_ << row;
    }

    controllerFile_.flush();
    controllerBuffer_.clear();
}

std::int64_t LoggerNode::stampNs(const builtin_interfaces::msg::Time &stamp)
{
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
           static_cast<std::int64_t>(stamp.nanosec);
}

std::string LoggerNode::makeFileName(const std::string &suffix)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time, &tm);

    std::ostringstream name;
    name << std::put_time(&tm, "%Y%m%d_%H%M%S")
         << '_' << suffix << ".csv";
    return name.str();
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LoggerNode>());
    rclcpp::shutdown();
    return 0;
}
