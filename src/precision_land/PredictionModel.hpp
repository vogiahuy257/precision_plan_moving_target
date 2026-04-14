#pragma once

#include "ControlTypes.hpp"

namespace precision_land
{
/**
 * Mô hình dự đoán tương lai cho target và UAV.
 *
 * Logic:
 * - Target: mô hình vận tốc không đổi.
 * - UAV: mô hình vận tốc hiện tại + thành phần gia tốc XY tùy chọn.
 * - Sai số điều khiển lấy tại thời điểm tương lai.
 */
class PredictionModel
{
public:
    PredictionOutput predict(const PredictionInput &input) const;
};
} // namespace precision_land
