#ifndef EIGENQUATERNIONPARAMETERIZATION_H
#define EIGENQUATERNIONPARAMETERIZATION_H

// Compatibility note:
// - Old VINS/camodocal code was written for Ceres LocalParameterization.
// - New Ceres versions, used on Ubuntu 24.04 / ROS 2 Jazzy, removed
//   ceres/local_parameterization.h and use ceres::Manifold instead.
//
// This file keeps the original Plus() and ComputeJacobian() logic unchanged,
// and only adapts the base class/API so the same code can build with both
// old and new Ceres.
#if __has_include(<ceres/local_parameterization.h>)
  #include <ceres/local_parameterization.h>
  #define CAMODOCAL_CERES_HAS_LOCAL_PARAMETERIZATION 1
#else
  #include <ceres/manifold.h>
  #define CAMODOCAL_CERES_HAS_LOCAL_PARAMETERIZATION 0
#endif

namespace camodocal
{

class EigenQuaternionParameterization : public
#if CAMODOCAL_CERES_HAS_LOCAL_PARAMETERIZATION
    ceres::LocalParameterization
#else
    ceres::Manifold
#endif
{
public:
    ~EigenQuaternionParameterization() override {}

    // Original camodocal/VINS local update. Do not change this logic because
    // other calibration/model code relies on the same quaternion convention.
    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const override;

#if CAMODOCAL_CERES_HAS_LOCAL_PARAMETERIZATION
    bool ComputeJacobian(const double* x,
                         double* jacobian) const override;

    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 3; }
#else
    // Ceres Manifold API wrappers. They preserve the old global/local sizes
    // and route PlusJacobian() to the original ComputeJacobian() implementation.
    int AmbientSize() const override { return 4; }
    int TangentSize() const override { return 3; }

    bool PlusJacobian(const double* x,
                      double* jacobian) const override;

    bool RightMultiplyByPlusJacobian(const double* x,
                                     int num_rows,
                                     const double* ambient_matrix,
                                     double* tangent_matrix) const override;

    bool Minus(const double* y,
               const double* x,
               double* y_minus_x) const override;

    bool MinusJacobian(const double* x,
                       double* jacobian) const override;

    // Keep these helper names for old source compatibility.
    int GlobalSize() const { return 4; }
    int LocalSize() const { return 3; }
#endif

private:
    template<typename T>
    void EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const;
};


template<typename T>
void
EigenQuaternionParameterization::EigenQuaternionProduct(const T z[4], const T w[4], T zw[4]) const
{
    zw[0] = z[3] * w[0] + z[0] * w[3] + z[1] * w[2] - z[2] * w[1];
    zw[1] = z[3] * w[1] - z[0] * w[2] + z[1] * w[3] + z[2] * w[0];
    zw[2] = z[3] * w[2] + z[0] * w[1] - z[1] * w[0] + z[2] * w[3];
    zw[3] = z[3] * w[3] - z[0] * w[0] - z[1] * w[1] - z[2] * w[2];
}

}

#endif
