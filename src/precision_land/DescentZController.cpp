#include "DescentZController.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace precision_land
{
void DescentZController::configure(const ZControllerParams &params)
{
    // Kiem tra nhanh param de tranh cau hinh loi ngay tu dau
    if (!std::isfinite(params.landZoneZ) ||
        !std::isfinite(params.descentVel) ||
        !std::isfinite(params.descentGateRadius) ||
        !std::isfinite(params.vmin) ||
        !std::isfinite(params.vmax))
    {
        throw std::runtime_error("ZControllerParams chua gia tri NaN/Inf");
    }

    if (params.landZoneZ < 0.0f)
    {
        throw std::runtime_error("landZoneZ phai >= 0");
    }

    if (params.descentGateRadius < 0.0f)
    {
        throw std::runtime_error("descentGateRadius phai >= 0");
    }

    if (params.vmin < 0.0f || params.vmax < 0.0f)
    {
        throw std::runtime_error("vmin va vmax phai >= 0");
    }

    if (params.vmin > params.vmax)
    {
        throw std::runtime_error("vmin khong duoc lon hon vmax");
    }

    params_ = params;
}

ZControllerOutput DescentZController::computeCommand(const ZControllerInput &input) const
{
    ZControllerOutput output{};
    output.vzCommand = 0.0f;

    try
    {
        if (!std::isfinite(input.futureErrorXY.x()) ||
            !std::isfinite(input.futureErrorXY.y()) ||
            !std::isfinite(input.vehicleAltitudeAbs))
        {
            throw std::runtime_error("ZControllerInput chua gia tri NaN/Inf");
        }

        if (input.vehicleAltitudeAbs < 0.0f)
        {
            throw std::runtime_error("vehicleAltitudeAbs phai >= 0");
        }

        const float lateralError = std::sqrt(
            input.futureErrorXY.x() * input.futureErrorXY.x() +
            input.futureErrorXY.y() * input.futureErrorXY.y());

        if (!std::isfinite(lateralError))
        {
            throw std::runtime_error("lateralError khong hop le");
        }

        // Khi vao vung gan dat thi tiep tuc ha voi van toc co dinh
        if (input.vehicleAltitudeAbs < params_.landZoneZ)
        {
            output.vzCommand = std::abs(params_.descentVel);
            return output;
        }

        // Neu con lech ngang lon thi chua cho ha
        if (lateralError >= params_.descentGateRadius)
        {
            output.vzCommand = 0.0f;
            return output;
        }

        // Trong gate thi scale van toc ha theo sai so ngang
        const float denominator = std::max(params_.descentGateRadius, 1e-6f);
        const float scale = 1.0f - lateralError / denominator;
        const float scaleClamped = std::clamp(scale, 0.0f, 1.0f);

        output.vzCommand = params_.vmin + (params_.vmax - params_.vmin) * scaleClamped;

        if (!std::isfinite(output.vzCommand))
        {
            throw std::runtime_error("vzCommand tinh ra khong hop le");
        }

        return output;
    }
    catch (const std::exception &e)
    {
        // Fallback an toan: khong ha nua neu co loi
        output.vzCommand = 0.0f;
        throw std::runtime_error(std::string("DescentZController::computeCommand loi: ") + e.what());
    }
    catch (...)
    {
        output.vzCommand = 0.0f;
        throw std::runtime_error("DescentZController::computeCommand gap loi khong xac dinh");
    }
}
} // namespace precision_land