#include "PipelineTimingCollector.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <rclcpp/qos.hpp>

namespace precision_land
{
void PipelineTimingCollector::setEnabled(bool enable)
{
    enabled_ = enable;

    if (!enabled_)
    {
        close();
    }
}

void PipelineTimingCollector::startSession(rclcpp::Node &node, const std::string &sessionStamp)
{
    if (!enabled_)
    {
        return;
    }

    try
    {
        close();

        node_ = &node;
        sessionStamp_ = sessionStamp;
        resolvedCsvPath_ = buildSessionCsvPath(sessionStamp_);
        sessionStarted_ = true;

        openFileIfNeeded();
        writeHeaderIfNeeded();
        createRosInterfaces();
    }
    catch (...)
    {
        disableOnError();
        throw;
    }
}

void PipelineTimingCollector::flush()
{
    if (!enabled_ || !fileOpened_)
    {
        return;
    }

    try
    {
        if (!csvLineBuffer_.empty())
        {
            for (const std::string &line : csvLineBuffer_)
            {
                csvFile_ << line << '\n';
            }

            csvFile_.flush();
            csvLineBuffer_.clear();
        }
    }
    catch (...)
    {
        disableOnError();
        throw;
    }
}

void PipelineTimingCollector::close()
{
    try
    {
        flush();
    }
    catch (...)
    {
    }

    flushTimer_.reset();
    arucoSub_.reset();
    kalmanSub_.reset();
    precisionLandSub_.reset();

    records_.clear();
    csvLineBuffer_.clear();

    if (csvFile_.is_open())
    {
        csvFile_.close();
    }

    node_ = nullptr;
    sessionStarted_ = false;
    fileOpened_ = false;
    headerWritten_ = false;
    sessionStamp_.clear();
    resolvedCsvPath_.clear();
}

void PipelineTimingCollector::createRosInterfaces()
{
    if (node_ == nullptr)
    {
        throw std::runtime_error("node_ nullptr trong createRosInterfaces");
    }

    rclcpp::QoS qos(rclcpp::KeepLast(50));
    qos.best_effort();

    arucoSub_ = node_->create_subscription<std_msgs::msg::String>(
        "/debug_dt/aruco",
        qos,
        std::bind(&PipelineTimingCollector::arucoCallback, this, std::placeholders::_1));

    kalmanSub_ = node_->create_subscription<std_msgs::msg::String>(
        "/debug_dt/kalman",
        qos,
        std::bind(&PipelineTimingCollector::kalmanCallback, this, std::placeholders::_1));

    precisionLandSub_ = node_->create_subscription<std_msgs::msg::String>(
        "/debug_dt/precision_land",
        qos,
        std::bind(&PipelineTimingCollector::precisionLandCallback, this, std::placeholders::_1));

    flushTimer_ = node_->create_wall_timer(
        std::chrono::duration<double>(kFlushPeriodSec),
        std::bind(&PipelineTimingCollector::flushTimerCallback, this));
}

void PipelineTimingCollector::openFileIfNeeded()
{
    if (fileOpened_)
    {
        return;
    }

    if (resolvedCsvPath_.empty())
    {
        throw std::runtime_error("resolvedCsvPath rong, chua startSession");
    }

    csvFile_.open(resolvedCsvPath_, std::ios::out | std::ios::trunc);
    if (!csvFile_.is_open())
    {
        throw std::runtime_error("Khong mo duoc file pipeline timing CSV: " + resolvedCsvPath_);
    }

    fileOpened_ = true;
}

void PipelineTimingCollector::writeHeaderIfNeeded()
{
    if (!fileOpened_ || headerWritten_)
    {
        return;
    }

    csvFile_
        << "image_stamp,"
        << "aruco_found,"
        << "aruco_rx_now,aruco_proc_start,aruco_proc_end,aruco_pub_now,"
        << "aruco_rx_wait_dt,aruco_queue_before_proc_dt,aruco_processing_dt,aruco_send_dt,aruco_total_node_dt,"
        << "kalman_rx_now,kalman_proc_start,kalman_proc_end,kalman_pub_now,"
        << "kalman_processing_dt,kalman_send_dt,kalman_measurement_dt,kalman_predict_dt,"
        << "pl_pose_stamp,pl_vel_stamp,pl_pose_rx_now,pl_vel_rx_now,"
        << "pl_ctrl_start_now,pl_ctrl_end_now,pl_cmd_pub_now,"
        << "pl_pose_wait_dt,pl_vel_wait_dt,pl_control_processing_dt,pl_send_cmd_dt,"
        << "dt_aruco_to_kalman,dt_kalman_to_pl_pose,dt_kalman_to_pl_vel,dt_total_image_to_cmd"
        << '\n';

    csvFile_.flush();
    headerWritten_ = true;
}

void PipelineTimingCollector::disableOnError()
{
    enabled_ = false;
    close();
}

void PipelineTimingCollector::arucoCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (!enabled_ || !sessionStarted_ || msg == nullptr)
    {
        return;
    }

    std::unordered_map<std::string, std::string> data;
    if (!parseJson(msg->data, data))
    {
        return;
    }

    const double imageStamp = getFloat(data, "image_stamp", -1.0);
    if (imageStamp < 0.0)
    {
        return;
    }

    const std::string key = makeKey(imageStamp);
    PipelineTimingRecord &record = records_[key];

    record.recordKeyStamp = imageStamp;
    record.imageStamp = imageStamp;

    record.arucoRxNow = getFloat(data, "rx_now");
    record.arucoProcStart = getFloat(data, "proc_start");
    record.arucoProcEnd = getFloat(data, "proc_end");
    record.arucoPubNow = getFloat(data, "pub_now");
    record.arucoRxWaitDt = getFloat(data, "rx_wait_dt");
    record.arucoQueueBeforeProcDt = getFloat(data, "queue_before_proc_dt");
    record.arucoProcessingDt = getFloat(data, "processing_dt");
    record.arucoSendDt = getFloat(data, "send_dt");
    record.arucoTotalNodeDt = getFloat(data, "total_node_dt");
    record.arucoFound = getInt(data, "found", -1);

    updateLastSeen(record);
}

void PipelineTimingCollector::kalmanCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (!enabled_ || !sessionStarted_ || msg == nullptr)
    {
        return;
    }

    std::unordered_map<std::string, std::string> data;
    if (!parseJson(msg->data, data))
    {
        return;
    }

    const double imageStamp = getFloat(data, "image_stamp", -1.0);
    if (imageStamp < 0.0)
    {
        return;
    }

    if (getString(data, "stage", "") != "PUB")
    {
        return;
    }

    const std::string key = makeKey(imageStamp);
    PipelineTimingRecord &record = records_[key];

    record.recordKeyStamp = imageStamp;
    record.imageStamp = imageStamp;

    record.kalmanRxNow = getFloat(data, "rx_now");
    record.kalmanProcStart = getFloat(data, "proc_start");
    record.kalmanProcEnd = getFloat(data, "proc_end");
    record.kalmanPubNow = getFloat(data, "pub_now");
    record.kalmanProcessingDt = getFloat(data, "processing_dt");
    record.kalmanSendDt = getFloat(data, "send_dt");
    record.kalmanMeasurementDt = getFloat(data, "measurement_dt");
    record.kalmanPredictDt = getFloat(data, "predict_dt");

    updateLastSeen(record);
}

void PipelineTimingCollector::precisionLandCallback(const std_msgs::msg::String::SharedPtr msg)
{
    if (!enabled_ || !sessionStarted_ || msg == nullptr)
    {
        return;
    }

    std::unordered_map<std::string, std::string> data;
    if (!parseJson(msg->data, data))
    {
        return;
    }

    const double joinStamp = getFirstValidStamp(data, {"image_stamp", "state_stamp", "pose_stamp"});
    if (joinStamp < 0.0)
    {
        return;
    }

    const std::string key = makeKey(joinStamp);
    PipelineTimingRecord &record = records_[key];

    record.recordKeyStamp = joinStamp;
    record.imageStamp = joinStamp;

    record.plPoseStamp = getFloat(data, "pose_stamp");
    record.plVelStamp = getFloat(data, "vel_stamp");
    record.plPoseRxNow = getFloat(data, "pose_rx_now");
    record.plVelRxNow = getFloat(data, "vel_rx_now");
    record.plCtrlStartNow = getFloat(data, "ctrl_start_now");
    record.plCtrlEndNow = getFloat(data, "ctrl_end_now");
    record.plCmdPubNow = getFloat(data, "cmd_pub_now");
    record.plPoseWaitDt = getFloat(data, "pose_wait_dt");
    record.plVelWaitDt = getFloat(data, "vel_wait_dt");
    record.plControlProcessingDt = getFloat(data, "control_processing_dt");
    record.plSendCmdDt = getFloat(data, "send_cmd_dt");
    record.dtTotalImageToCmd = getFirstValidStamp(data, {"total_image_to_cmd_dt", "total_state_to_cmd_dt"});

    updateLastSeen(record);
}

void PipelineTimingCollector::flushTimerCallback()
{
    if (!enabled_ || !sessionStarted_ || node_ == nullptr)
    {
        return;
    }

    try
    {
        const double nowSec = static_cast<double>(node_->get_clock()->now().nanoseconds()) / 1e9;

        std::vector<std::string> completeKeys;
        std::vector<std::string> staleKeys;

        completeKeys.reserve(records_.size());
        staleKeys.reserve(records_.size());

        for (const auto &[key, record] : records_)
        {
            if (hasCompleteRecord(record))
            {
                completeKeys.push_back(key);
            }
            else if (isStaleRecord(record, nowSec))
            {
                staleKeys.push_back(key);
            }
        }

        if (completeKeys.empty() && staleKeys.empty())
        {
            return;
        }

        std::sort(
            completeKeys.begin(),
            completeKeys.end(),
            [](const std::string &a, const std::string &b)
            {
                return std::stod(a) < std::stod(b);
            });

        for (const std::string &key : completeKeys)
        {
            csvLineBuffer_.push_back(buildCsvRow(records_.at(key)));
        }

        if (csvLineBuffer_.size() >= kFlushBatchSize)
        {
            flush();
        }

        for (const std::string &key : completeKeys)
        {
            records_.erase(key);
        }

        for (const std::string &key : staleKeys)
        {
            records_.erase(key);
        }
    }
    catch (...)
    {
        disableOnError();
        throw;
    }
}

bool PipelineTimingCollector::parseJson(
    const std::string &raw,
    std::unordered_map<std::string, std::string> &flatMap) const
{
    flatMap.clear();

    std::string text = raw;
    text.erase(std::remove_if(text.begin(), text.end(), ::isspace), text.end());

    if (text.size() < 2 || text.front() != '{' || text.back() != '}')
    {
        return false;
    }

    text = text.substr(1, text.size() - 2);

    std::size_t start = 0;
    while (start < text.size())
    {
        std::size_t commaPos = start;
        bool inQuotes = false;

        while (commaPos < text.size())
        {
            if (text[commaPos] == '"' && (commaPos == 0 || text[commaPos - 1] != '\\'))
            {
                inQuotes = !inQuotes;
            }

            if (!inQuotes && text[commaPos] == ',')
            {
                break;
            }

            ++commaPos;
        }

        const std::string token = text.substr(start, commaPos - start);
        const std::size_t colonPos = token.find(':');

        if (colonPos != std::string::npos)
        {
            std::string key = token.substr(0, colonPos);
            std::string value = token.substr(colonPos + 1);

            if (!key.empty() && key.front() == '"' && key.back() == '"')
            {
                key = key.substr(1, key.size() - 2);
            }

            if (!value.empty() && value.front() == '"' && value.back() == '"')
            {
                value = value.substr(1, value.size() - 2);
            }

            flatMap[key] = value;
        }

        start = commaPos + 1;
    }

    return !flatMap.empty();
}

double PipelineTimingCollector::getFloat(
    const std::unordered_map<std::string, std::string> &data,
    const std::string &key,
    double defaultValue) const
{
    const auto it = data.find(key);
    if (it == data.end())
    {
        return defaultValue;
    }

    try
    {
        return std::stod(it->second);
    }
    catch (...)
    {
        return defaultValue;
    }
}

int PipelineTimingCollector::getInt(
    const std::unordered_map<std::string, std::string> &data,
    const std::string &key,
    int defaultValue) const
{
    const auto it = data.find(key);
    if (it == data.end())
    {
        return defaultValue;
    }

    try
    {
        return std::stoi(it->second);
    }
    catch (...)
    {
        return defaultValue;
    }
}

std::string PipelineTimingCollector::getString(
    const std::unordered_map<std::string, std::string> &data,
    const std::string &key,
    const std::string &defaultValue) const
{
    const auto it = data.find(key);
    if (it == data.end())
    {
        return defaultValue;
    }

    return it->second;
}

double PipelineTimingCollector::getFirstValidStamp(
    const std::unordered_map<std::string, std::string> &data,
    const std::vector<std::string> &keys) const
{
    for (const std::string &key : keys)
    {
        const double value = getFloat(data, key, -1.0);
        if (value >= 0.0)
        {
            return value;
        }
    }

    return -1.0;
}

std::string PipelineTimingCollector::makeKey(double stamp) const
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << stamp;
    return ss.str();
}

void PipelineTimingCollector::updateLastSeen(PipelineTimingRecord &record)
{
    if (node_ == nullptr)
    {
        return;
    }

    record.collectorLastSeen = static_cast<double>(node_->get_clock()->now().nanoseconds()) / 1e9;
}

bool PipelineTimingCollector::hasCompleteRecord(const PipelineTimingRecord &record) const
{
    return
        record.recordKeyStamp >= 0.0 &&
        record.arucoPubNow >= 0.0 &&
        record.kalmanPubNow >= 0.0 &&
        record.plCtrlStartNow >= 0.0 &&
        record.plCmdPubNow >= 0.0;
}

bool PipelineTimingCollector::isStaleRecord(const PipelineTimingRecord &record, double nowSec) const
{
    return (nowSec - record.collectorLastSeen) > kStaleTimeoutSec;
}

std::string PipelineTimingCollector::buildSessionCsvPath(const std::string &sessionStamp) const
{
    namespace fs = std::filesystem;

    const fs::path logDir(kLogDirectory);
    fs::create_directories(logDir);

    return (logDir / (sessionStamp + "_pipeline_timing.csv")).string();
}

std::string PipelineTimingCollector::buildCsvRow(const PipelineTimingRecord &record) const
{
    const double dtArucoToKalman =
        (record.arucoPubNow >= 0.0 && record.kalmanRxNow >= 0.0)
            ? std::max(0.0, record.kalmanRxNow - record.arucoPubNow)
            : -1.0;

    const double dtKalmanToPlPose =
        (record.kalmanPubNow >= 0.0 && record.plPoseRxNow >= 0.0)
            ? std::max(0.0, record.plPoseRxNow - record.kalmanPubNow)
            : -1.0;

    const double dtKalmanToPlVel =
        (record.kalmanPubNow >= 0.0 && record.plVelRxNow >= 0.0)
            ? std::max(0.0, record.plVelRxNow - record.kalmanPubNow)
            : -1.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);

    ss
        << record.imageStamp << ','
        << record.arucoFound << ','

        << record.arucoRxNow << ','
        << record.arucoProcStart << ','
        << record.arucoProcEnd << ','
        << record.arucoPubNow << ','
        << record.arucoRxWaitDt << ','
        << record.arucoQueueBeforeProcDt << ','
        << record.arucoProcessingDt << ','
        << record.arucoSendDt << ','
        << record.arucoTotalNodeDt << ','

        << record.kalmanRxNow << ','
        << record.kalmanProcStart << ','
        << record.kalmanProcEnd << ','
        << record.kalmanPubNow << ','
        << record.kalmanProcessingDt << ','
        << record.kalmanSendDt << ','
        << record.kalmanMeasurementDt << ','
        << record.kalmanPredictDt << ','

        << record.plPoseStamp << ','
        << record.plVelStamp << ','
        << record.plPoseRxNow << ','
        << record.plVelRxNow << ','
        << record.plCtrlStartNow << ','
        << record.plCtrlEndNow << ','
        << record.plCmdPubNow << ','
        << record.plPoseWaitDt << ','
        << record.plVelWaitDt << ','
        << record.plControlProcessingDt << ','
        << record.plSendCmdDt << ','

        << dtArucoToKalman << ','
        << dtKalmanToPlPose << ','
        << dtKalmanToPlVel << ','
        << record.dtTotalImageToCmd;

    return ss.str();
}
} // namespace precision_land