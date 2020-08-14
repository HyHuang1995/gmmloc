#include "gmmloc/gmm/factors.h"

namespace g2o {

void EdgePt2Gaussian::computeError() {
  const VertexSBAPointXYZ *v1 =
      static_cast<const VertexSBAPointXYZ *>(_vertices[0]);

  // Vector3 obs(_measurement);
  _error = comp_->sqrt_info_.transpose() * (v1->estimate() - comp_->mean_);
}

void EdgePt2Gaussian::linearizeOplus() {
  // _jacobianOplusXi = normal_.transpose();

  _jacobianOplusXi = comp_->sqrt_info_.transpose();
}

void EdgeSE3QuatPrior::computeError() {
  const VertexSE3Expmap *v1 =
      static_cast<const VertexSE3Expmap *>(_vertices[0]);

  const SE3Quat &Tj = v1->estimate();

  SE3Quat delta = _inverseMeasurement * Tj;

  _error = delta.log();
}

void EdgeSE3QuatPrior::linearizeOplus() {
  // _jacobianOplusXi = normal_.transpose();
  const VertexSE3Expmap *v1 =
      static_cast<const VertexSE3Expmap *>(_vertices[0]);

  const SE3Quat &Tj = v1->estimate();

  SE3Quat delta = _inverseMeasurement * Tj;

  Vector6 dvec = delta.log();

  Eigen::Matrix<double, 6, 6> Jr = Eigen::Matrix<double, 6, 6>::Zero();
  const Eigen::Matrix3d &&phi_s = skew(dvec.head<3>());
  const Eigen::Matrix3d &&luo_s = skew(dvec.tail<3>());
  Jr.block(0, 0, 3, 3) = phi_s;
  Jr.block(3, 3, 3, 3) = phi_s;
  Jr.block(0, 3, 3, 3) = luo_s;
  Jr *= 0.5;
  Jr = Eigen::Matrix<double, 6, 6>::Identity() + Jr;

  _jacobianOplusXi = Jr * Tj.inverse().adj();

  // Jr.block(0, 0, )
}

void EdgePt2GaussianDeg::computeError() {
  const VertexSBAPointXYZ *v1 =
      static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
  // Vector3 obs(_measurement);
  _error = normal_.transpose() * (v1->estimate() - mean_);
}

void EdgePt2GaussianDeg::linearizeOplus() {
  _jacobianOplusXi = normal_.transpose();
}

inline Vector2 project2d(const Vector3 &v) {
  return Vector2(v(0) / v(2), v(1) / v(2));
}

Vector2 EdgeProjectXYZOnly::cam_project(const Vector3 &trans_xyz) const {
  Vector2 proj = project2d(trans_xyz);

  return Vector2(proj[0] * fx + cx, proj[1] * fy + cy);
}

void EdgeProjectXYZOnly::computeError() {
  const VertexSBAPointXYZ *v1 =
      static_cast<const VertexSBAPointXYZ *>(_vertices[0]);

  Vector2 obs(_measurement);
  _error = obs - cam_project(rot_c_w_ * v1->estimate() + t_c_w_);
}

void EdgeProjectXYZOnly::linearizeOplus() {
  // VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  // SE3Quat T(vj->estimate());
  VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
  Vector3 xyz = vi->estimate();

  Vector3 xyz_trans = rot_c_w_ * xyz + t_c_w_;

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z * z;

  Eigen::Matrix<double, 2, 3> tmp;
  tmp(0, 0) = fx;
  tmp(0, 1) = 0;
  tmp(0, 2) = -x / z * fx;

  tmp(1, 0) = 0;
  tmp(1, 1) = fy;
  tmp(1, 2) = -y / z * fy;

  _jacobianOplusXi = -1. / z * tmp * rot_c_w_.toRotationMatrix();
}

// Vector2 EdgeProjectXYZOnlyStereo::cam_project(const Vector3 &trans_xyz) const
// {
//   Vector2 proj = project2d(trans_xyz);

//   return Vector2(proj[0] * fx + cx, proj[1] * fy + cy);
// }

Vector3 EdgeProjectXYZOnlyStereo::cam_project(const Vector3 &trans_xyz) const {
  const number_t invz = 1.0f / trans_xyz[2];
  Vector3 res;
  res[0] = trans_xyz[0] * invz * fx + cx;
  res[1] = trans_xyz[1] * invz * fy + cy;
  res[2] = res[0] - bf * invz;
  return res;
}

void EdgeProjectXYZOnlyStereo::computeError() {
  const VertexSBAPointXYZ *v1 =
      static_cast<const VertexSBAPointXYZ *>(_vertices[0]);

  Vector3 obs(_measurement);
  _error = obs - cam_project(rot_c_w_ * v1->estimate() + t_c_w_);
}

void EdgeProjectXYZOnlyStereo::linearizeOplus() {
  // VertexSE3Expmap *vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
  // SE3Quat T(vj->estimate());
  VertexSBAPointXYZ *vi = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
  Vector3 xyz = vi->estimate();

  Vector3 xyz_trans = rot_c_w_ * xyz + t_c_w_;

  double x = xyz_trans[0];
  double y = xyz_trans[1];
  double z = xyz_trans[2];
  double z_2 = z * z;

  // Eigen::Matrix<double, 2, 3> tmp;
  // tmp(0, 0) = fx;
  // tmp(0, 1) = 0;
  // tmp(0, 2) = -x / z * fx;

  // tmp(1, 0) = 0;
  // tmp(1, 1) = fy;
  // tmp(1, 2) = -y / z * fy;

  Eigen::Matrix3d R = rot_c_w_.toRotationMatrix();

  _jacobianOplusXi(0, 0) = -fx * R(0, 0) / z + fx * x * R(2, 0) / z_2;
  _jacobianOplusXi(0, 1) = -fx * R(0, 1) / z + fx * x * R(2, 1) / z_2;
  _jacobianOplusXi(0, 2) = -fx * R(0, 2) / z + fx * x * R(2, 2) / z_2;

  _jacobianOplusXi(1, 0) = -fy * R(1, 0) / z + fy * y * R(2, 0) / z_2;
  _jacobianOplusXi(1, 1) = -fy * R(1, 1) / z + fy * y * R(2, 1) / z_2;
  _jacobianOplusXi(1, 2) = -fy * R(1, 2) / z + fy * y * R(2, 2) / z_2;

  _jacobianOplusXi(2, 0) = _jacobianOplusXi(0, 0) - bf * R(2, 0) / z_2;
  _jacobianOplusXi(2, 1) = _jacobianOplusXi(0, 1) - bf * R(2, 1) / z_2;
  _jacobianOplusXi(2, 2) = _jacobianOplusXi(0, 2) - bf * R(2, 2) / z_2;
}

} // namespace g2o