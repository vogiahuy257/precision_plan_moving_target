#pragma once

#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace precision_land
{
struct PipelineTimingRecord
{
    double recordKeyStamp = -1.0;
    double imageStamp = -1.0;

    int arucoFound = -1;

    double arucoRxNow = -1.0;
    double arucoProcStart = -1.0;
    double arucoProcEnd = -1.0;
    double arucoPubNow = -1.0;
    double arucoRxWaitDt = -1.0;
    double arucoQueueBeforeProcDt = -1.0;
    double arucoProcessingDt = -1.0;
    double arucoSendDt = -1.0;
    double arucoTotalNodeDt = -1.0;

    double kalmanRxNow = -1.0;
    double kalmanProcStart = -1.0;
    double kalmanProcEnd = -1.0;
    double kalmanPubNow = -1.0;
    double kalmanProcessingDt = -1.0;
    double kalmanSendDt = -1.0;
    double kalmanMeasurementDt = -1.0;
    double kalmanPredictDt = -1.0;

    double plPoseStamp = -1.0;
    double plVelStamp = -1.0;
    double plPoseRxNow = -1.0;
    double plVelRxNow = -1.0;
    double plCtrlStartNow = -1.0;
    double plCtrlEndNow = -1.0;
    double plCmdPubNow = -1.0;
    double plPoseWaitDt = -1.0;
    double plVelWaitDt = -1.0;
    double plControlProcessingDt = -1.0;
    double plSendCmdDt = -1.0;
    double dtTotalImageToCmd = -1.0;

    double collectorLastSeen = 0.0;
};

class PipelineTimingCollector
{
public:
    void setEnabled(bool enable);

    void startSession(rclcpp::Node &node, const std::string &sessionStamp);

    void flush();
    void close();

    bool isEnabled() const
    {
        return enabled_;
    }

private:
    void createRosInterfaces();
    void openFileIfNeeded();
    void writeHeaderIfNeeded();
    void disableOnError();

    void arucoCallback(const std_msgs::msg::String::SharedPtr msg);
    void kalmanCallback(const std_msgs::msg::String::SharedPtr msg);
    void precisionLandCallback(const std_msgs::msg::String::SharedPtr msg);

    void flushTimerCallback();

    bool parseJson(const std::string &raw, std::unordered_map<std::string, std::string> &flatMap) const;
    double getFloat(const std::unordered_map<std::string, std::string> &data, const std::string &key, double defaultValue = -1.0) const;
    int getInt(const std::unordered_map<std::string, std::string> &data, const std::string &key, int defaultValue = -1) const;
    std::string getString(const std::unordered_map<std::string, std::string> &data, const std::string &key, const std::string &defaultValue = "") const;

    double getFirstValidStamp(const std::unordered_map<std::string, std::string> &data, const std::vector<std::string> &keys) const;
    std::string makeKey(double stamp) const;

    void updateLastSeen(PipelineTimingRecord &record);
    bool hasCompleteRecord(const PipelineTimingRecord &record) const;
    bool isStaleRecord(const PipelineTimingRecord &record, double nowSec) const;

    std::string buildSessionCsvPath(const std::string &sessionStamp) const;
    std::string buildCsvRow(const PipelineTimingRecord &record) const;

private:
    static constexpr std::size_t kFlushBatchSize = 100;
    static constexpr double kFlushPeriodSec = 0.5;
    static constexpr double kStaleTimeoutSec = 5.0;
    static constexpr const char *kLogDirectory = "precisionland_logs/pipeline_timing";

    bool enabled_ = false;
    bool sessionStarted_ = false;
    bool fileOpened_ = false;
    bool headerWritten_ = false;

    rclcpp::Node *node_ = nullptr;
    std::string sessionStamp_;
    std::string resolvedCsvPath_;
    std::ofstream csvFile_;

    std::unordered_map<std::string, PipelineTimingRecord> records_;
    std::vector<std::string> csvLineBuffer_;

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr arucoSub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr kalmanSub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr precisionLandSub_;
    rclcpp::TimerBase::SharedPtr flushTimer_;
};
} // namespace precision_land