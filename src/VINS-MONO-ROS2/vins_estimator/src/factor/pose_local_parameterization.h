#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include "../utility/utility.h"

// Ceres >= 2.x replaced LocalParameterization with Manifold.
// This class preserves the old VINS pose update logic:
//   p_plus = p + dp
//   q_plus = q * deltaQ(dtheta)
// with pose memory layout [px, py, pz, qx, qy, qz, qw]
// because Eigen::Map<Eigen::Quaterniond> expects coeffs as [x, y, z, w].
class PoseLocalParameterization : public ceres::Manifold
{
public:
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override;
    bool PlusJacobian(const double *x, double *jacobian) const override;
    bool RightMultiplyByPlusJacobian(const double *x,
                                     const int num_rows,
                                     const double *ambient_matrix,
                                     double *tangent_matrix) const override;
    bool Minus(const double *y, const double *x, double *y_minus_x) const override;
    bool MinusJacobian(const double *x, double *jacobian) const override;

    int AmbientSize() const override { return 7; }
    int TangentSize() const override { return 6; }

    // Kept for compatibility with the old implementation name.
    bool ComputeJacobian(const double *x, double *jacobian) const;
};
