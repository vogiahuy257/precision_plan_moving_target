#include "DescentZController.hpp"

#include <algorithm>
#include <cmath>

namespace precision_land
{
void DescentZController::configure(const ZControllerParams &params)
{
    params_ = params;
}

ZControllerOutput DescentZController::computeCommand(const ZControllerInput &input) const
{
    ZControllerOutput output{};

    const float lateralError = std::sqrt(
        input.futureErrorXY.x() * input.futureErrorXY.x() +
        input.futureErrorXY.y() * input.futureErrorXY.y());

    // Khi rất gần đất và sai số ngang đủ nhỏ thì yêu cầu disarm
    // if (input.vehicleAltitudeAbs < params_.disarmHeight)
    // {
    //     output.vzCommand = 0.0f;
    //     output.shouldDisarm = true;
    //     return output;
    // }

    // Khi vào vùng gần đất thì tiếp tục hạ với vận tốc cố định
    if (input.vehicleAltitudeAbs < params_.landZoneZ)
    {
        output.vzCommand = std::abs(params_.descentVel);
        output.shouldDisarm = false;
        return output;
    }

    // Nếu còn lệch ngang lớn thì chưa cho hạ
    if (lateralError >= params_.descentGateRadius)
    {
        output.vzCommand = 0.0f;
        output.shouldDisarm = false;
        return output;
    }

    // Trong gate thì scale vận tốc hạ theo sai số ngang
    const float scale = 1.0f - lateralError / std::max(params_.descentGateRadius, 1e-6f);
    const float scaleClamped = std::clamp(scale, 0.0f, 1.0f);

    output.vzCommand = params_.vmin + (params_.vmax - params_.vmin) * scaleClamped;
    output.shouldDisarm = false;
    return output;
}
} // namespace precision_land