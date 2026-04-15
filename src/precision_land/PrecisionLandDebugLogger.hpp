#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

#include "ControlTypes.hpp"

namespace precision_land
{
class PrecisionLandDebugLogger
{
public:
    void setEnabled(bool enable);

    void startSession();

    void logSample(const PrecisionLandDebugSample &sample);

    void flush();
    void close();

    bool isEnabled() const
    {
        return enabled_;
    }

    const std::string &getSessionStamp() const
    {
        return sessionStamp_;
    }

private:
    void openFileIfNeeded();
    void writeHeaderIfNeeded();
    void disableOnError();

    std::string makeCurrentTimeString() const;
    std::string buildSessionCsvPath() const;
    std::string sampleToCsvLine(const PrecisionLandDebugSample &sample) const;

private:
    static constexpr std::size_t kFlushBatchSize = 100;
    static constexpr const char *kLogDirectory = "precisionland_logs/controller";

    bool enabled_ = false;
    bool fileOpened_ = false;
    bool headerWritten_ = false;
    bool sessionStarted_ = false;

    std::string sessionStamp_;
    std::string resolvedCsvPath_;
    std::ofstream csvFile_;
    std::vector<std::string> lineBuffer_;
};
} // namespace precision_land