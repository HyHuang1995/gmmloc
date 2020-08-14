#include "gmmloc/utils/math_utils.h"

namespace gmmloc {
Matrix3d MathUtils::skew(const Vector3d &v) {
  Matrix3d m;
  m.fill(0.);
  m(0, 1) = -v(2);
  m(0, 2) = v(1);
  m(1, 2) = -v(0);
  m(1, 0) = v(2);
  m(2, 0) = -v(1);
  m(2, 1) = v(0);
  return m;
}

Matrix3d MathUtils::computeEssentialMatrix(const SE3Quat &Tcw1,
                                           const SE3Quat &Tcw2

) {

  Quaterniond rot_c1_w = Tcw1.rotation();
  Vector3d t_c1_w = Tcw1.translation();
  Quaterniond rot_c2_w = Tcw2.rotation();
  Vector3d t_c2_w = Tcw2.translation();

  Quaterniond rot_c1_c2 = rot_c1_w * rot_c2_w.conjugate();
  Vector3d t_c1_c2 = -(rot_c1_c2 * t_c2_w) + t_c1_w;

  Matrix3d t12_skew = skew(t_c1_c2);

  return t12_skew * rot_c1_c2.toRotationMatrix();
}

Matrix3d MathUtils::computeFundamentalMatrix(const SE3Quat &Tcw1,
                                             const Matrix3d &K1,
                                             const SE3Quat &Tcw2,
                                             const Matrix3d &K2

) {
  Matrix3d essential_mat = computeEssentialMatrix(Tcw1, Tcw2);

  return K1.transpose().inverse() * essential_mat * K2.inverse();
}

} // namespace gmmloc
