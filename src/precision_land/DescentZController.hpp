#pragma once

#include "ControlTypes.hpp"

namespace precision_land
{
/**
 * Bộ điều khiển trục Z cho pha hạ cánh.
 *
 * Logic:
 * - Khi rất gần đất và sai số ngang nhỏ: yêu cầu disarm.
 * - Khi gần đất: hạ với tốc độ cố định.
 * - Khi lệch ngang lớn: chưa cho hạ.
 * - Khi vào vùng gate: hạ theo profile tuyến tính.
 */
class DescentZController
{
public:
    /**
     * Cấu hình tham số cho bộ điều khiển Z.
     *
     * Input:
     *     params: bộ tham số điều khiển Z
     *
     * Logic:
     *     lưu toàn bộ tham số để dùng khi tính lệnh hạ
     *
     * Output:
     *     không có
     */
    void configure(const ZControllerParams &params);

    /**
     * Tính lệnh vận tốc hạ và cờ disarm.
     *
     * Input:
     *     input.vehicleAltitudeAbs: độ cao hiện tại so với mặt đất
     *     input.futureErrorXY: sai số XY dự đoán
     *
     * Logic:
     *     - nếu rất gần đất và lệch ngang nhỏ thì yêu cầu disarm
     *     - nếu vào vùng land zone thì hạ với vận tốc cố định
     *     - nếu ngoài gate thì giữ vz = 0
     *     - nếu trong gate thì scale vận tốc hạ theo sai số ngang
     *
     * Output:
     *     ZControllerOutput gồm vzCommand và shouldDisarm
     */
    ZControllerOutput computeCommand(const ZControllerInput &input) const;

private:
    ZControllerParams params_{};
};
} // namespace precision_land