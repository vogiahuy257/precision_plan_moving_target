#pragma once

#include "ControlTypes.hpp"
#include "DisturbanceObserver.hpp"

#include <Eigen/Core>

namespace precision_land
{
/**
 * Bộ điều khiển velocity XY.
 *
 * Logic:
 *     - PID chạy trên sai số tương lai.
 *     - Có cộng feedforward vận tốc target.
 *     - Có cộng bù nhiễu từ Disturbance Observer.
 *     - Có clamp biên tốc độ theo chuẩn vector norm.
 *     - Có slew limiter ở đầu ra cuối cùng.
 */
class XYVelocityController
{
public:
    XYVelocityController() = default;

    /**
     * Cấu hình tham số cho bộ điều khiển XY.
     *
     * Input:
     *     params: bộ tham số PID, giới hạn vận tốc, slew và DOB.
     *
     * Logic:
     *     - Kiểm tra NaN/Inf và giới hạn cơ bản.
     *     - Lưu tham số vào controller.
     *     - Cấu hình DisturbanceObserver.
     *
     * Output:
     *     Không trả về giá trị. Nếu tham số lỗi thì throw runtime_error.
     */
    void configure(const XYControllerParams &params);

    /**
     * Reset toàn bộ trạng thái nội bộ của controller.
     *
     * Input:
     *     Không có.
     *
     * Logic:
     *     - Reset tích phân PID.
     *     - Reset sai số cũ của D.
     *     - Reset slew output.
     *     - Reset DisturbanceObserver.
     *
     * Output:
     *     Không trả về giá trị.
     */
    void reset();

    /**
     * Tính velocity setpoint XY cho UAV.
     *
     * Input:
     *     input: sai số tương lai, vận tốc target, vận tốc UAV và dt.
     *
     * Logic:
     *     - Tính PID feedback từ futureErrorXY.
     *     - Cộng feedforward vận tốc target nếu được cho phép.
     *     - Dùng DOB để ước lượng nhiễu tương đương và cộng bù.
     *     - Clamp vector vận tốc và áp dụng slew limiter.
     *
     * Output:
     *     XYControllerOutput chứa các thành phần debug và velocitySpXY cuối cùng.
     */
    XYControllerOutput update(const XYControllerInput &input);

private:
    /**
     * Giới hạn tốc độ thay đổi vận tốc lệnh theo gia tốc tối đa.
     */
    float applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const;

    /**
     * Giới hạn vector theo norm để tổng vận tốc XY không vượt maxNorm.
     */
    Eigen::Vector2f clampVectorNorm(const Eigen::Vector2f &value, float maxNorm) const;

private:
    XYControllerParams params_{};

    float velXIntegral_{0.0f};
    float velYIntegral_{0.0f};

    float prevErrX_{0.0f};
    float prevErrY_{0.0f};
    bool prevErrValid_{false};

    float vxFilt_{0.0f};
    float vyFilt_{0.0f};

    DisturbanceObserver disturbanceObserver_{};
    Eigen::Vector2f prevVelocitySpXY_{Eigen::Vector2f::Zero()};
    bool prevVelocitySpValid_{false};
};
} // namespace precision_land