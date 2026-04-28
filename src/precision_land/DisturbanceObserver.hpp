#pragma once

#include "ControlTypes.hpp"

#include <Eigen/Core>

namespace precision_land
{
/**
 * Uoc luong nhieu tuong duong tren truc van toc XY.
 *
 * Logic:
 *     - Lay sai khac giua velocity setpoint da gui va velocity UAV do duoc.
 *     - Loc low-pass de lay thanh phan nhieu cham do gio/he dong hoc.
 *     - Xuat compensationXY de cong bu vao velocity command.
 */
class DisturbanceObserver
{
public:
    /**
     * Cau hinh tham so DOB.
     *
     * Input:
     *     params: tham so enabled, tauSec, gain, maxBias va deadband.
     *
     * Logic:
     *     - Kiem tra NaN/Inf va gioi han hop le.
     *     - Luu tham so de dung trong update().
     *
     * Output:
     *     Khong tra ve gia tri. Neu tham so loi thi throw runtime_error.
     */
    void configure(const DisturbanceObserverParams &params);

    /**
     * Reset trang thai uoc luong nhieu.
     *
     * Input:
     *     Khong co.
     *
     * Logic:
     *     - Dua disturbanceHatXY ve 0.
     *
     * Output:
     *     Khong tra ve gia tri.
     */
    void reset();

    /**
     * Cap nhat DOB theo moi chu ky dieu khien.
     *
     * Input:
     *     input: velocity tham chieu, velocity UAV do duoc, targetValid va dtSec.
     *
     * Logic:
     *     - Neu DOB tat thi reset nhieu ve 0.
     *     - Neu target khong hop le thi decay nhieu ve 0.
     *     - Neu target hop le thi loc residual velocity de uoc luong nhieu.
     *
     * Output:
     *     DisturbanceObserverOutput gom estimatedDisturbanceXY va compensationXY.
     */
    DisturbanceObserverOutput update(const DisturbanceObserverInput &input);

private:
    Eigen::Vector2f clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const;
    Eigen::Vector2f applyVectorDeadband(const Eigen::Vector2f &value, float deadband) const;

private:
    DisturbanceObserverParams params_{};
    Eigen::Vector2f disturbanceHatXY_{Eigen::Vector2f::Zero()};
};
} // namespace precision_land