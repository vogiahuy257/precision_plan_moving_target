#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    (void)x;

    // Same Jacobian as the old ceres::LocalParameterization implementation.
    // Matrix size is AmbientSize x TangentSize = 7 x 6, row-major.
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}

bool PoseLocalParameterization::PlusJacobian(const double *x, double *jacobian) const
{
    return ComputeJacobian(x, jacobian);
}

bool PoseLocalParameterization::RightMultiplyByPlusJacobian(const double *x,
                                                            const int num_rows,
                                                            const double *ambient_matrix,
                                                            double *tangent_matrix) const
{
    Eigen::Matrix<double, 7, 6, Eigen::RowMajor> plus_jacobian;
    if (!PlusJacobian(x, plus_jacobian.data()))
    {
        return false;
    }

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 7, Eigen::RowMajor>> A(ambient_matrix, num_rows, 7);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>> B(tangent_matrix, num_rows, 6);
    B.noalias() = A * plus_jacobian;

    return true;
}

bool PoseLocalParameterization::Minus(const double *y, const double *x, double *y_minus_x) const
{
    Eigen::Map<const Eigen::Vector3d> p_y(y);
    Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);

    Eigen::Map<const Eigen::Vector3d> p_x(x);
    Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);

    Eigen::Map<Eigen::Vector3d> dp(y_minus_x);
    Eigen::Map<Eigen::Vector3d> dtheta(y_minus_x + 3);

    dp = p_y - p_x;

    Eigen::Quaterniond dq = q_x.conjugate() * q_y;
    if (dq.w() < 0.0)
    {
        dq.coeffs() *= -1.0;
    }

    // Inverse of Utility::deltaQ(theta) under the small-angle convention used by VINS.
    dtheta = 2.0 * dq.vec();

    return true;
}

bool PoseLocalParameterization::MinusJacobian(const double *x, double *jacobian) const
{
    (void)x;

    // Matrix size is TangentSize x AmbientSize = 6 x 7, row-major.
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(jacobian);
    j.setZero();
    j.topLeftCorner<3, 3>().setIdentity();
    j.block<3, 3>(3, 3).setIdentity();

    return true;
}
