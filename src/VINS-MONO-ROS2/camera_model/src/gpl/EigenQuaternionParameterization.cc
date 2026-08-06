#include "camodocal/gpl/EigenQuaternionParameterization.h"

#include <cmath>
#include <vector>

namespace camodocal
{

bool
EigenQuaternionParameterization::Plus(const double* x,
                                      const double* delta,
                                      double* x_plus_delta) const
{
    const double norm_delta =
        sqrt(delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]);
    if (norm_delta > 0.0)
    {
        const double sin_delta_by_delta = (sin(norm_delta) / norm_delta);
        double q_delta[4];
        q_delta[0] = sin_delta_by_delta * delta[0];
        q_delta[1] = sin_delta_by_delta * delta[1];
        q_delta[2] = sin_delta_by_delta * delta[2];
        q_delta[3] = cos(norm_delta);
        EigenQuaternionProduct(q_delta, x, x_plus_delta);
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            x_plus_delta[i] = x[i];
        }
    }
    return true;
}

namespace
{

bool
ComputeEigenQuaternionJacobian(const double* x, double* jacobian)
{
    jacobian[0] =  x[3]; jacobian[1]  =  x[2]; jacobian[2]  = -x[1];  // NOLINT
    jacobian[3] = -x[2]; jacobian[4]  =  x[3]; jacobian[5]  =  x[0];  // NOLINT
    jacobian[6] =  x[1]; jacobian[7]  = -x[0]; jacobian[8]  =  x[3];  // NOLINT
    jacobian[9] = -x[0]; jacobian[10] = -x[1]; jacobian[11] = -x[2];  // NOLINT
    return true;
}

} // namespace

#if CAMODOCAL_CERES_HAS_LOCAL_PARAMETERIZATION

bool
EigenQuaternionParameterization::ComputeJacobian(const double* x,
                                                 double* jacobian) const
{
    return ComputeEigenQuaternionJacobian(x, jacobian);
}

#else

bool
EigenQuaternionParameterization::PlusJacobian(const double* x,
                                              double* jacobian) const
{
    return ComputeEigenQuaternionJacobian(x, jacobian);
}

bool
EigenQuaternionParameterization::RightMultiplyByPlusJacobian(const double* x,
                                                             int num_rows,
                                                             const double* ambient_matrix,
                                                             double* tangent_matrix) const
{
    double plus_jacobian[12];
    if (!PlusJacobian(x, plus_jacobian))
    {
        return false;
    }

    for (int r = 0; r < num_rows; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k)
            {
                sum += ambient_matrix[r * 4 + k] * plus_jacobian[k * 3 + c];
            }
            tangent_matrix[r * 3 + c] = sum;
        }
    }

    return true;
}

bool
EigenQuaternionParameterization::Minus(const double* y,
                                       const double* x,
                                       double* y_minus_x) const
{
    // Inverse operation for the original Plus(): y = q_delta * x.
    // Quaternion coefficient order is Eigen convention: [x, y, z, w].
    const double x_inv[4] = {-x[0], -x[1], -x[2], x[3]};
    double q_delta[4];
    EigenQuaternionProduct(y, x_inv, q_delta);

    const double vec_norm = sqrt(q_delta[0] * q_delta[0] +
                                 q_delta[1] * q_delta[1] +
                                 q_delta[2] * q_delta[2]);

    if (vec_norm > 0.0)
    {
        const double angle = atan2(vec_norm, q_delta[3]);
        const double scale = angle / vec_norm;
        y_minus_x[0] = scale * q_delta[0];
        y_minus_x[1] = scale * q_delta[1];
        y_minus_x[2] = scale * q_delta[2];
    }
    else
    {
        y_minus_x[0] = 0.0;
        y_minus_x[1] = 0.0;
        y_minus_x[2] = 0.0;
    }

    return true;
}

bool
EigenQuaternionParameterization::MinusJacobian(const double* x,
                                               double* jacobian) const
{
    // The old PlusJacobian() has orthonormal columns for unit quaternions,
    // so its transpose is the local inverse Jacobian at x.
    double plus_jacobian[12];
    if (!PlusJacobian(x, plus_jacobian))
    {
        return false;
    }

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            jacobian[r * 4 + c] = plus_jacobian[c * 3 + r];
        }
    }

    return true;
}

#endif

}
