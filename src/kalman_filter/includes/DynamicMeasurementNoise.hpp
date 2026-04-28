#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "DataStructs.hpp"

/**
 * Ket qua tinh toan ma tran nhieu do dong R_k.
 */
struct DynamicMeasurementNoiseResult
{
    double rx{0.0};
    double ry{0.0};
    double rz{0.0};
    double extraRxy{0.0};
    double rangeToTarget{0.0};
    double nearRangeError{0.0};
};

/**
 * Lop uoc luong measurement noise R dong theo khoang cach camera-target.
 *
 * Logic chinh:
 *     - R_base la nhieu do camera/ArUco co dinh.
 *     - Khi UAV ha xuong qua gan marker, pose ArUco/PnP co the dao dong hon.
 *     - Phan tang them cho R_xy duoc tinh theo do gan marker.
 *     - Khong su dung gyro/attitude rate trong class nay.
 */
class DynamicMeasurementNoiseEstimator
{
public:
    /**
     * Tinh R dong tu config noise va vi tri target trong optical frame.
     *
     * Input:
     *     noiseConfig           : cau hinh noise cua Kalman.
     *     targetPositionOptical : vi tri target trong he optical camera [m].
     *
     * Logic:
     *     - Neu dynamic R tat thi tra ve R co dinh.
     *     - Lay |z_optical| lam xap xi khoang cach camera -> target.
     *     - Neu khoang cach nho hon nearRange thi tang R_x va R_y.
     *     - Dat nearNoiseGain = 0.0 neu muon tat tac dong tang R khi gan marker.
     *
     * Output:
     *     DynamicMeasurementNoiseResult chua R_x, R_y, R_z va cac gia tri debug.
     */
    DynamicMeasurementNoiseResult estimate(
        const kalman_filter_data::NoiseConfig &noiseConfig,
        const Eigen::Vector3d &targetPositionOptical) const
    {
        validateNoiseConfig(noiseConfig);

        DynamicMeasurementNoiseResult result{};
        result.rx = noiseConfig.rPosX;
        result.ry = noiseConfig.rPosY;
        result.rz = noiseConfig.rPosZ;

        if (!noiseConfig.dynamicREnabled)
        {
            return result;
        }

        result.rangeToTarget = std::max(
            noiseConfig.minDynamicRange,
            std::abs(targetPositionOptical.z()));

        result.nearRangeError = std::max(
            0.0,
            noiseConfig.nearRange - result.rangeToTarget);

        result.extraRxy = noiseConfig.nearNoiseGain *
                          result.nearRangeError *
                          result.nearRangeError;

        result.extraRxy = std::clamp(
            result.extraRxy,
            0.0,
            noiseConfig.maxExtraRxy);

        result.rx += result.extraRxy;
        result.ry += result.extraRxy;

        return result;
    }

    /**
     * Ghi ket qua R dong vao ma tran measurementNoiseCov cua OpenCV KalmanFilter.
     *
     * Input:
     *     measurementNoiseCov : ma tran R cua OpenCV KalmanFilter, kich thuoc 3x3.
     *     result              : ket qua da tinh tu ham estimate().
     *
     * Logic:
     *     - Gan R_x, R_y, R_z vao duong cheo.
     *     - Dua cac phan tu ngoai duong cheo ve 0 de giu gia thiet noise doc lap.
     *
     * Output:
     *     measurementNoiseCov duoc cap nhat truc tiep.
     */
    void applyToMeasurementNoiseCov(
        cv::Mat &measurementNoiseCov,
        const DynamicMeasurementNoiseResult &result) const
    {
        if (measurementNoiseCov.rows != 3 || measurementNoiseCov.cols != 3)
        {
            throw std::runtime_error("measurementNoiseCov phai co kich thuoc 3x3");
        }

        measurementNoiseCov.at<double>(0, 0) = result.rx;
        measurementNoiseCov.at<double>(1, 1) = result.ry;
        measurementNoiseCov.at<double>(2, 2) = result.rz;

        measurementNoiseCov.at<double>(0, 1) = 0.0;
        measurementNoiseCov.at<double>(0, 2) = 0.0;
        measurementNoiseCov.at<double>(1, 0) = 0.0;
        measurementNoiseCov.at<double>(1, 2) = 0.0;
        measurementNoiseCov.at<double>(2, 0) = 0.0;
        measurementNoiseCov.at<double>(2, 1) = 0.0;
    }

private:
    /**
     * Kiem tra cac tham so noise truoc khi tinh dynamic R.
     *
     * Input:
     *     noiseConfig: cau hinh noise can kiem tra.
     *
     * Logic:
     *     Bao loi neu param NaN/Inf hoac am tai cac truong phuong sai/gain/gioi han.
     *
     * Output:
     *     Khong co. Nem exception neu config khong hop le.
     */
    void validateNoiseConfig(const kalman_filter_data::NoiseConfig &noiseConfig) const
    {
        const bool valid =
            std::isfinite(noiseConfig.rPosX) &&
            std::isfinite(noiseConfig.rPosY) &&
            std::isfinite(noiseConfig.rPosZ) &&
            std::isfinite(noiseConfig.nearRange) &&
            std::isfinite(noiseConfig.nearNoiseGain) &&
            std::isfinite(noiseConfig.maxExtraRxy) &&
            std::isfinite(noiseConfig.minDynamicRange) &&
            noiseConfig.rPosX >= 0.0 &&
            noiseConfig.rPosY >= 0.0 &&
            noiseConfig.rPosZ >= 0.0 &&
            noiseConfig.nearRange > 0.0 &&
            noiseConfig.nearNoiseGain >= 0.0 &&
            noiseConfig.maxExtraRxy >= 0.0 &&
            noiseConfig.minDynamicRange > 0.0;

        if (!valid)
        {
            throw std::runtime_error("NoiseConfig dynamic R khong hop le");
        }
    }
};
