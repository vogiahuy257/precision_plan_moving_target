#pragma once

#include "ControlTypes.hpp"

namespace precision_land
{
/**
 * Bộ điều khiển velocity XY.
 *
 * Logic:
 * - PID chạy trên sai số tương lai.
 * - Có cộng feedforward vận tốc target.
 * - Có clamp biên tốc độ.
 * - Có slew limiter ở đầu ra cuối cùng.
 */
class XYVelocityController
{
public:
    XYVelocityController() = default;

    void configure(const XYControllerParams &params);
    void reset();
    XYControllerOutput update(const XYControllerInput &input);

private:
    float applySlew(float commandVelocity, float previousVelocity, float accelLimit, float dtSec) const;

private:
    XYControllerParams params_{};

    float velXIntegral_{0.0f};
    float velYIntegral_{0.0f};

    float prevErrX_{0.0f};
    float prevErrY_{0.0f};
    bool prevErrValid_{false};

    float vxFilt_{0.0f};
    float vyFilt_{0.0f};
};
} // namespace precision_land
