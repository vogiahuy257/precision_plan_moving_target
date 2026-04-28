#pragma once

#include <fstream>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "DataStructs.hpp"

class DebugLogger
{
public:
    DebugLogger();
    ~DebugLogger();

    /**
     * Cau hinh logger theo param debug.
     *
     * Input:
     *     logger: logger cua node de in log ROS
     *     debugEnabled: bat/tat che do ghi CSV
     *     csvPath: duong dan file CSV can ghi
     *
     * Logic:
     *     Neu debugEnabled=false thi tat logger.
     *     Neu debugEnabled=true thi mo file, tao thu muc neu can va ghi header.
     *
     * Output:
     *     Logger san sang de ghi CSV hoac o trang thai disable.
     */
    void configure(
        const rclcpp::Logger &logger,
        bool debugEnabled,
        const std::string &csvPath);

    /**
     * Ghi 1 dong log tu SystemData.
     *
     * Input:
     *     data: du lieu tong hien tai cua node
     *     stamp: moc thoi gian cua dong log
     *
     * Logic:
     *     Neu logger dang tat thi bo qua.
     *     Neu dang bat thi convert SystemData -> DebugLogRow va ghi CSV.
     *
     * Output:
     *     Them 1 dong vao file CSV khi debug bat.
     */
    void log(
        const kalman_filter_data::SystemData &data,
        const rclcpp::Time &stamp);

    /**
     * Dong file log neu dang mo.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Dong stream file va reset trang thai noi bo.
     *
     * Output:
     *     Logger tro ve trang thai dong.
     */
    void close();

    /**
     * Kiem tra logger co dang duoc bat hay khong.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Tra ve co dang duoc enable hay khong.
     *
     * Output:
     *     true neu dang bat debug log.
     */
    bool isEnabled() const;

private:

    /**
    * Xay dung duong dan file CSV mac dinh neu nguoi dung khong cung cap.
    *
    * Input:
    *     Khong co.
    *
    * Logic:
    *     Tao thu muc logs neu chua ton tai, sau do tao ten file theo mau kalman_filter_debug_YYYYMMDD_HHMMSS.csv
    *
    * Output:
    *     Duong dan file CSV mac dinh da duoc tao san sang de ghi.
    */
    std::string buildDefaultCsvPath() const;

    /*
    * Xay dung ten file CSV mac dinh.
    *
    * Input:
    *     Khong co.
    *
    * Logic:
    *     Tao ten file theo mau kalman_filter_debug_YYYYMMDD_HHMMSS.csv
    *
    * Output:
    *     Ten file CSV mac dinh da duoc tao san sang de ghi.
    */
    std::string buildDefaultCsvFileName() const;
    
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
    std::string resolveCsvPath(const std::string &csvPath) const;

    /**
     * Ghi dong header cho file CSV.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     Ghi ten cac cot CSV theo thu tu co dinh.
     *
     * Output:
     *     Header duoc ghi 1 lan duy nhat.
     */
    void writeHeader();

    /**
     * Chuyen du lieu tong sang 1 dong log phang.
     *
     * Input:
     *     data: du lieu tong cua node
     *     stamp: thoi gian log
     *
     * Logic:
     *     Copy cac field can theo doi tu SystemData sang DebugLogRow.
     *
     * Output:
     *     1 dong log da san sang de ghi CSV.
     */
    kalman_filter_data::DebugLogRow buildRow(
        const kalman_filter_data::SystemData &data,
        const rclcpp::Time &stamp) const;

    /**
     * Ghi 1 dong CSV tu DebugLogRow.
     *
     * Input:
     *     row: du lieu da duoc flatten
     *
     * Logic:
     *     Ghi toan bo field vao file theo dung thu tu cot.
     *
     * Output:
     *     Them 1 dong du lieu vao file.
     */
    void writeRow(const kalman_filter_data::DebugLogRow &row);

private:
    bool enabled_{false};
    bool headerWritten_{false};
    std::size_t rowCounter_{0};

    std::string csvPath_{};
    std::ofstream file_{};
    rclcpp::Logger logger_{rclcpp::get_logger("DebugLogger")};
};