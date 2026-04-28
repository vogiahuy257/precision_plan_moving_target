#include "DebugLogger.hpp"

#include <filesystem>
#include <iomanip>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <sstream>

DebugLogger::DebugLogger()
{
    try
    {
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger constructor failed: %s",
            exception.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger constructor failed: unknown exception");
    }
}

DebugLogger::~DebugLogger()
{
    try
    {
        close();
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger destructor failed: %s",
            exception.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger destructor failed: unknown exception");
    }
}

/**
 * Cau hinh logger theo param debug.
 *
 * Input:
 *     logger: logger cua node de in log ROS
 *     debugEnabled: bat/tat che do ghi CSV
 *     csvPath: duong dan file CSV can ghi
 *
 * Logic:
 *     - Neu debugEnabled=false thi dong file cu neu co va khong mo file moi
 *     - Neu csvPath la "", "auto" hoac la thu muc nhu "kalman_logs/"
 *       thi tu sinh ten file theo ngay
 *     - Neu thu muc cha chua ton tai thi tu tao
 *     - Mo file va ghi header
 *
 * Output:
 *     Logger san sang de ghi log neu duoc bat.
 */
void DebugLogger::configure(
    const rclcpp::Logger &logger,
    bool debugEnabled,
    const std::string &csvPath)
{
    try
    {
        logger_ = logger;
        close();

        enabled_ = debugEnabled;
        rowCounter_ = 0;

        if (!enabled_)
        {
            RCLCPP_INFO(logger_, "DebugLogger disabled");
            return;
        }

        csvPath_ = resolveCsvPath(csvPath);

        const std::filesystem::path outputPath(csvPath_);
        const std::filesystem::path parentPath = outputPath.parent_path();

        if (!parentPath.empty())
        {
            std::filesystem::create_directories(parentPath);
        }

        file_.open(csvPath_, std::ios::out | std::ios::trunc);

        if (!file_.is_open())
        {
            enabled_ = false;
            throw std::runtime_error("Cannot open debug csv file: " + csvPath_);
        }

        file_ << std::fixed << std::setprecision(6);
        writeHeader();

        RCLCPP_INFO(
            logger_,
            "DebugLogger enabled | csv_path=%s",
            csvPath_.c_str());
    }
    catch (const std::exception &exception)
    {
        enabled_ = false;

        try
        {
            close();
        }
        catch (...)
        {
        }

        RCLCPP_ERROR(
            logger_,
            "DebugLogger configure failed: %s",
            exception.what());
    }
    catch (...)
    {
        enabled_ = false;

        try
        {
            close();
        }
        catch (...)
        {
        }

        RCLCPP_ERROR(
            logger_,
            "DebugLogger configure failed: unknown exception");
    }
}

/**
 * Tao ten file log mac dinh theo ngay hien tai.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Sinh ten file theo mau dd_mm_yyyy_kalman_log.csv
 *
 * Output:
 *     Tra ve ten file CSV.
 */
std::string DebugLogger::buildDefaultCsvFileName() const
{
    try
    {
        // Lấy thời gian hệ thống hiện tại
        const auto nowTimePoint = std::chrono::system_clock::now();
        const std::time_t nowTimeT = std::chrono::system_clock::to_time_t(nowTimePoint);

        // Chuyển sang thời gian local để tách giờ, phút, ngày, tháng, năm
        std::tm localTime{};
#if defined(_WIN32)
        localtime_s(&localTime, &nowTimeT);
#else
        localtime_r(&nowTimeT, &localTime);
#endif

        // Tạo tên file theo format:
        // hh_mm_dd_mm_yyyy_kalman_log.csv
        // Ví dụ: 14_32_21_04_2026_kalman_log.csv
        std::ostringstream oss;
        oss << std::setfill('0')
            << std::setw(2) << localTime.tm_hour << "_"
            << std::setw(2) << localTime.tm_min << "_"
            << std::setw(2) << localTime.tm_mday << "_"
            << std::setw(2) << (localTime.tm_mon + 1) << "_"
            << (localTime.tm_year + 1900)
            << "_kalman_log.csv";

        return oss.str();
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("DebugLogger::buildDefaultCsvFileName failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "DebugLogger::buildDefaultCsvFileName failed: unknown exception");
    }
}

/**
 * Xay dung duong dan CSV mac dinh.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Dat file log vao thu muc kalman_logs/ voi ten file theo ngay.
 *
 * Output:
 *     Tra ve duong dan day du cho file CSV mac dinh.
 */
std::string DebugLogger::buildDefaultCsvPath() const
{
    try
    {
        return (std::filesystem::path("kalman_logs") / buildDefaultCsvFileName()).string();
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("DebugLogger::buildDefaultCsvPath failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("DebugLogger::buildDefaultCsvPath failed: unknown exception");
    }
}

/**
 * Xu ly csvPath do user cung cap.
 *
 * Input:
 *     csvPath: chuoi path doc tu param
 *
 * Logic:
 *     - Neu la "" hoac "auto" -> tao path mac dinh
 *     - Neu la thu muc, vi du "kalman_logs/" -> tao file mac dinh trong thu muc do
 *     - Neu la duong dan file hop le -> giu nguyen
 *
 * Output:
 *     Tra ve duong dan file CSV cuoi cung se duoc mo.
 */
std::string DebugLogger::resolveCsvPath(const std::string &csvPath) const
{
    try
    {
        if (csvPath.empty() || csvPath == "auto")
        {
            return buildDefaultCsvPath();
        }

        const std::filesystem::path inputPath(csvPath);

        bool looksLikeDirectory = false;

        if (!csvPath.empty())
        {
            const char lastChar = csvPath.back();
            if (lastChar == '/' || lastChar == '\\')
            {
                looksLikeDirectory = true;
            }
        }

        if (!looksLikeDirectory &&
            std::filesystem::exists(inputPath) &&
            std::filesystem::is_directory(inputPath))
        {
            looksLikeDirectory = true;
        }

        if (looksLikeDirectory)
        {
            return (inputPath / buildDefaultCsvFileName()).string();
        }

        return csvPath;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(
            std::string("DebugLogger::resolveCsvPath failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("DebugLogger::resolveCsvPath failed: unknown exception");
    }
}

/**
 * Ghi 1 dong log tu SystemData.
 *
 * Input:
 *     data: du lieu tong hien tai cua node
 *     stamp: moc thoi gian cua dong log
 *
 * Logic:
 *     Neu logger tat hoac file chua mo thi bo qua.
 *     Neu hop le thi build row va ghi CSV.
 *
 * Output:
 *     File CSV duoc them 1 dong.
 */
void DebugLogger::log(
    const kalman_filter_data::SystemData &data,
    const rclcpp::Time &stamp)
{
    try
    {
        if (!enabled_ || !file_.is_open())
        {
            return;
        }

        const kalman_filter_data::DebugLogRow row = buildRow(data, stamp);
        writeRow(row);

        rowCounter_++;

        if ((rowCounter_ % 20U) == 0U)
        {
            file_.flush();
        }
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger log failed: %s",
            exception.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger log failed: unknown exception");
    }
}

/**
 * Dong file log neu dang mo.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Dong file va reset trang thai noi bo.
 *
 * Output:
 *     Logger tro ve trang thai dong.
 */
void DebugLogger::close()
{
    try
    {
        if (file_.is_open())
        {
            file_.flush();
            file_.close();
        }

        headerWritten_ = false;
    }
    catch (const std::exception &exception)
    {
        enabled_ = false;

        RCLCPP_ERROR(
            logger_,
            "DebugLogger close failed: %s",
            exception.what());
    }
    catch (...)
    {
        enabled_ = false;

        RCLCPP_ERROR(
            logger_,
            "DebugLogger close failed: unknown exception");
    }
}

/**
 * Kiem tra logger co dang bat hay khong.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Tra ve co dang enable khong.
 *
 * Output:
 *     true neu dang bat.
 */
bool DebugLogger::isEnabled() const
{
    try
    {
        return enabled_;
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger isEnabled failed: %s",
            exception.what());
        return false;
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger isEnabled failed: unknown exception");
        return false;
    }
}

/**
 * Ghi header CSV.
 *
 * Input:
 *     Khong co.
 *
 * Logic:
 *     Ghi danh sach ten cot 1 lan duy nhat.
 *
 * Output:
 *     File CSV co dong header.
 */
void DebugLogger::writeHeader()
{
    try
    {
        if (!file_.is_open() || headerWritten_)
        {
            return;
        }

        file_
            << "stamp_sec,"
            << "initialized,"
            << "force_zero,"
            << "target_valid,"
            << "vehicle_odom_valid,"
            << "vehicle_local_pos_valid,"
            << "vehicle_pos_x,"
            << "vehicle_pos_y,"
            << "vehicle_pos_z,"
            << "vehicle_vel_x,"
            << "vehicle_vel_y,"
            << "vehicle_vel_z,"
            << "meas_opt_x,"
            << "meas_opt_y,"
            << "meas_opt_z,"
            << "meas_world_x,"
            << "meas_world_y,"
            << "meas_world_z,"
            << "est_pos_x,"
            << "est_pos_y,"
            << "est_pos_z,"
            << "est_vel_x,"
            << "est_vel_y,"
            << "est_vel_z,"
            << "predict_dt,"
            << "predict_count,"
            << "mount_mode,"
            << "last_reset_command"
            << "\n";

        headerWritten_ = true;
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger writeHeader failed: %s",
            exception.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger writeHeader failed: unknown exception");
    }
}

/**
 * Chuyen du lieu tong sang 1 dong log phang.
 *
 * Input:
 *     data: du lieu tong cua node
 *     stamp: moc thoi gian log
 *
 * Logic:
 *     Copy cac field quan trong tu SystemData sang DebugLogRow.
 *
 * Output:
 *     Tra ve DebugLogRow san sang ghi CSV.
 */
kalman_filter_data::DebugLogRow DebugLogger::buildRow(
    const kalman_filter_data::SystemData &data,
    const rclcpp::Time &stamp) const
{
    try
    {
        kalman_filter_data::DebugLogRow row;

        row.stampSec = stamp.seconds();

        row.initialized = data.runtime.initialized;
        row.forceZero = data.runtime.forceZero;
        row.targetValid = data.runtime.targetValid;
        row.vehicleOdomValid = data.runtime.vehicleOdomValid;
        row.vehicleLocalPosValid = data.runtime.vehicleLocalPosValid;

        row.vehiclePosX = data.vehicle.positionWorld.x();
        row.vehiclePosY = data.vehicle.positionWorld.y();
        row.vehiclePosZ = data.vehicle.positionWorld.z();

        row.vehicleVelX = data.vehicle.velocityWorld.x();
        row.vehicleVelY = data.vehicle.velocityWorld.y();
        row.vehicleVelZ = data.vehicle.velocityWorld.z();

        row.measOptX = data.targetMeasurement.positionOptical.x();
        row.measOptY = data.targetMeasurement.positionOptical.y();
        row.measOptZ = data.targetMeasurement.positionOptical.z();

        row.measWorldX = data.targetMeasurement.positionWorld.x();
        row.measWorldY = data.targetMeasurement.positionWorld.y();
        row.measWorldZ = data.targetMeasurement.positionWorld.z();

        row.estPosX = data.kalman.estimatedPositionWorld.x();
        row.estPosY = data.kalman.estimatedPositionWorld.y();
        row.estPosZ = data.kalman.estimatedPositionWorld.z();

        row.estVelX = data.kalman.estimatedVelocityWorld.x();
        row.estVelY = data.kalman.estimatedVelocityWorld.y();
        row.estVelZ = data.kalman.estimatedVelocityWorld.z();

        row.predictDt = data.kalman.predictDt;
        row.predictCount = data.kalman.predictCount;

        row.mountMode = data.config.transform.mountModeString;
        row.lastResetCommand = data.runtime.lastResetCommand;

        return row;
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger buildRow failed: %s",
            exception.what());
        return kalman_filter_data::DebugLogRow{};
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger buildRow failed: unknown exception");
        return kalman_filter_data::DebugLogRow{};
    }
}

/**
 * Ghi 1 dong CSV tu DebugLogRow.
 *
 * Input:
 *     row: dong log da duoc flatten
 *
 * Logic:
 *     Ghi tat ca gia tri theo cung thu tu voi header.
 *
 * Output:
 *     Them 1 dong vao file CSV.
 */
void DebugLogger::writeRow(const kalman_filter_data::DebugLogRow &row)
{
    try
    {
        if (!file_.is_open())
        {
            return;
        }

        file_
            << row.stampSec << ","
            << static_cast<int>(row.initialized) << ","
            << static_cast<int>(row.forceZero) << ","
            << static_cast<int>(row.targetValid) << ","
            << static_cast<int>(row.vehicleOdomValid) << ","
            << static_cast<int>(row.vehicleLocalPosValid) << ","
            << row.vehiclePosX << ","
            << row.vehiclePosY << ","
            << row.vehiclePosZ << ","
            << row.vehicleVelX << ","
            << row.vehicleVelY << ","
            << row.vehicleVelZ << ","
            << row.measOptX << ","
            << row.measOptY << ","
            << row.measOptZ << ","
            << row.measWorldX << ","
            << row.measWorldY << ","
            << row.measWorldZ << ","
            << row.estPosX << ","
            << row.estPosY << ","
            << row.estPosZ << ","
            << row.estVelX << ","
            << row.estVelY << ","
            << row.estVelZ << ","
            << row.predictDt << ","
            << row.predictCount << ","
            << row.mountMode << ","
            << row.lastResetCommand
            << "\n";
    }
    catch (const std::exception &exception)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger writeRow failed: %s",
            exception.what());
    }
    catch (...)
    {
        RCLCPP_ERROR(
            logger_,
            "DebugLogger writeRow failed: unknown exception");
    }
}