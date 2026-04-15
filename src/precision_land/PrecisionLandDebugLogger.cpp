#include "PrecisionLandDebugLogger.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace precision_land
{
void PrecisionLandDebugLogger::setEnabled(bool enable)
{
    enabled_ = enable;

    if (!enabled_)
    {
        close();
    }
}

void PrecisionLandDebugLogger::startSession()
{
    if (!enabled_)
    {
        return;
    }

    try
    {
        close();

        sessionStamp_ = makeCurrentTimeString();
        resolvedCsvPath_ = buildSessionCsvPath();
        sessionStarted_ = true;

        openFileIfNeeded();
        writeHeaderIfNeeded();
    }
    catch (...)
    {
        disableOnError();
        throw;
    }
}

void PrecisionLandDebugLogger::logSample(const PrecisionLandDebugSample &sample)
{
    if (!enabled_ || !sessionStarted_)
    {
        return;
    }

    try
    {
        openFileIfNeeded();
        writeHeaderIfNeeded();

        lineBuffer_.push_back(sampleToCsvLine(sample));

        if (lineBuffer_.size() >= kFlushBatchSize)
        {
            flush();
        }
    }
    catch (...)
    {
        disableOnError();
        throw;
    }
}

void PrecisionLandDebugLogger::flush()
{
    if (!enabled_ || !fileOpened_ || lineBuffer_.empty())
    {
        return;
    }

    try
    {
        for (const std::string &line : lineBuffer_)
        {
            csvFile_ << line << '\n';
        }

        csvFile_.flush();
        lineBuffer_.clear();
    }
    catch (...)
    {
        disableOnError();
        throw;
    }
}

void PrecisionLandDebugLogger::close()
{
    try
    {
        if (fileOpened_ && !lineBuffer_.empty())
        {
            for (const std::string &line : lineBuffer_)
            {
                csvFile_ << line << '\n';
            }

            csvFile_.flush();
        }
    }
    catch (...)
    {
    }

    lineBuffer_.clear();

    if (csvFile_.is_open())
    {
        csvFile_.close();
    }

    fileOpened_ = false;
    headerWritten_ = false;
    sessionStarted_ = false;
    sessionStamp_.clear();
    resolvedCsvPath_.clear();
}

void PrecisionLandDebugLogger::openFileIfNeeded()
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
        throw std::runtime_error("Khong mo duoc file CSV debug: " + resolvedCsvPath_);
    }

    fileOpened_ = true;
}

void PrecisionLandDebugLogger::writeHeaderIfNeeded()
{
    if (!fileOpened_ || headerWritten_)
    {
        return;
    }

    csvFile_
        << "time,state,"
        << "drone_pos_x,drone_pos_y,drone_pos_z,"
        << "drone_vel_x,drone_vel_y,drone_vel_z,"
        << "target_raw_x,target_raw_y,target_raw_z,"
        << "target_est_x,target_est_y,target_est_z,"
        << "target_pred_x,target_pred_y,target_pred_z,"
        << "target_vel_x,target_vel_y,target_vel_z,"
        << "error_x,error_y,"
        << "future_error_x,future_error_y,"
        << "error_xy_norm,future_error_xy_norm,"
        << "pid_out_x,pid_out_y,"
        << "ff_x,ff_y,"
        << "final_sp_x,final_sp_y,final_sp_z,"
        << "altitude_abs,dist_bottom,"
        << "should_disarm,land_detected,"
        << "pose_wait_dt,vel_wait_dt,control_processing_dt,send_cmd_dt,total_image_to_cmd_dt"
        << '\n';

    csvFile_.flush();
    headerWritten_ = true;
}

void PrecisionLandDebugLogger::disableOnError()
{
    enabled_ = false;
    close();
}

std::string PrecisionLandDebugLogger::makeCurrentTimeString() const
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTimeT = std::chrono::system_clock::to_time_t(now);

    std::tm localTm{};
#if defined(_WIN32)
    localtime_s(&localTm, &nowTimeT);
#else
    localtime_r(&nowTimeT, &localTm);
#endif

    std::ostringstream ss;
    ss << std::put_time(&localTm, "%H%M_%d_%m_%y");
    return ss.str();
}

std::string PrecisionLandDebugLogger::buildSessionCsvPath() const
{
    namespace fs = std::filesystem;

    const fs::path logDir(kLogDirectory);
    fs::create_directories(logDir);

    return (logDir / (sessionStamp_ + "_controller.csv")).string();
}

std::string PrecisionLandDebugLogger::sampleToCsvLine(const PrecisionLandDebugSample &sample) const
{
    const float errorNorm = sample.errorXY.norm();
    const float futureErrorNorm = sample.futureErrorXY.norm();

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);

    ss
        << sample.timeSec << ','
        << sample.state << ','

        << sample.dronePos.x() << ',' << sample.dronePos.y() << ',' << sample.dronePos.z() << ','
        << sample.droneVel.x() << ',' << sample.droneVel.y() << ',' << sample.droneVel.z() << ','

        << sample.targetRaw.x() << ',' << sample.targetRaw.y() << ',' << sample.targetRaw.z() << ','
        << sample.targetEst.x() << ',' << sample.targetEst.y() << ',' << sample.targetEst.z() << ','
        << sample.targetPred.x() << ',' << sample.targetPred.y() << ',' << sample.targetPred.z() << ','
        << sample.targetVel.x() << ',' << sample.targetVel.y() << ',' << sample.targetVel.z() << ','

        << sample.errorXY.x() << ',' << sample.errorXY.y() << ','
        << sample.futureErrorXY.x() << ',' << sample.futureErrorXY.y() << ','
        << errorNorm << ',' << futureErrorNorm << ','

        << sample.pidOutXY.x() << ',' << sample.pidOutXY.y() << ','
        << sample.ffXY.x() << ',' << sample.ffXY.y() << ','

        << sample.finalSp.x() << ',' << sample.finalSp.y() << ',' << sample.finalSp.z() << ','

        << sample.altitudeAbs << ',' << sample.distBottom << ','
        << static_cast<int>(sample.shouldDisarm) << ','
        << static_cast<int>(sample.landDetected) << ','

        << sample.timing.poseWaitDt << ','
        << sample.timing.velWaitDt << ','
        << sample.timing.controlProcessingDt << ','
        << sample.timing.sendCmdDt << ','
        << sample.timing.totalImageToCmdDt;

    return ss.str();
}
} // namespace precision_land