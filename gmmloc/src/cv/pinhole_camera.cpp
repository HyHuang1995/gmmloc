#include "gmmloc/cv/pinhole_camera.h"

#include "gmmloc/common/common.h"

namespace gmmloc {

const double PinholeCamera::kMinimumDepth = 0.0;

PinholeCamera::PinholeCamera(const Eigen::Vector4d &intrinsics,
                             uint32_t image_width, uint32_t image_height)
    : intrinsics_(intrinsics), width_(image_width), height_(image_height) {}

PinholeCamera::PinholeCamera(double focallength_cols, double focallength_rows,
                             double imagecenter_cols, double imagecenter_rows,
                             uint32_t image_width, uint32_t image_height)
    : intrinsics_(focallength_cols, focallength_rows, imagecenter_cols,
                  imagecenter_rows),
      width_(image_width), height_(image_height) {}

void PinholeCamera::unproject3(const Eigen::Vector2d &uv, const double &z,
                               Eigen::Vector3d *pt3d) const {
  CHECK_NOTNULL(pt3d);
  CHECK_EQ(intrinsics_.size(), kNumOfParams) << "intrinsics: invalid size!";

  const auto &fu = intrinsics_[0];
  const auto &fv = intrinsics_[1];
  const auto &cu = intrinsics_[2];
  const auto &cv = intrinsics_[3];

  (*pt3d)[0] = z * (uv[0] - cu) / fu;
  (*pt3d)[1] = z * (uv[1] - cv) / fv;
  (*pt3d)[2] = z;
}

void PinholeCamera::unproject3(const Eigen::Quaterniond &rot_w_c,
                               const Eigen::Vector3d &t_w_c,
                               const Eigen::Vector2d &uv, double z,
                               Eigen::Vector3d *pt3d) const {
  Eigen::Vector3d ptc;
  unproject3(uv, z, &ptc);

  (*pt3d) = rot_w_c * ptc + t_w_c;
}

const ProjectionResult
PinholeCamera::project3(const Eigen::Vector3d &point_3d,
                        Eigen::Vector2d *out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);
  CHECK_EQ(intrinsics_.size(), kNumOfParams) << "intrinsics: invalid size!";

  const auto &fu = intrinsics_[0];
  const auto &fv = intrinsics_[1];
  const auto &cu = intrinsics_[2];
  const auto &cv = intrinsics_[3];

  auto rz = static_cast<double>(1.0) / point_3d[2];

  Eigen::Vector2d keypoint;
  keypoint[0] = point_3d[0] * rz;
  keypoint[1] = point_3d[1] * rz;

  (*out_keypoint)[0] = fu * keypoint[0] + cu;
  (*out_keypoint)[1] = fv * keypoint[1] + cv;

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

const ProjectionResult PinholeCamera::project3(
    const Eigen::Vector3d &point_3d, Eigen::Vector2d *out_keypoint,
    Eigen::Matrix<double, 2, 3> *out_jacobian_point,
    Eigen::Matrix<double, 2, 4> *out_jacobian_intrinsics) const {

  CHECK_NOTNULL(out_keypoint);

  const double &fu = intrinsics_[0];
  const double &fv = intrinsics_[1];
  const double &cu = intrinsics_[2];
  const double &cv = intrinsics_[3];

  // Project the point.
  const double &x = point_3d[0];
  const double &y = point_3d[1];
  const double &z = point_3d[2];

  const double rz = 1.0 / z;
  (*out_keypoint)[0] = x * rz;
  (*out_keypoint)[1] = y * rz;

  // jacobian point
  if (out_jacobian_point) {
    // Jacobian including distortion
    const double rz2 = rz * rz;

    const double duf_dx = fu * rz;
    // const double duf_dy = 0.0;
    const double duf_dz = -fu * x * rz2;
    // const double dvf_dx = 0.0;
    const double dvf_dy = fv * rz;
    const double dvf_dz = -fv * y * rz2;

    (*out_jacobian_point) << duf_dx, 0.0, duf_dz, 0.0, dvf_dy, dvf_dz;
  }

  // Calculate the Jacobian w.r.t to the intrinsic parameters, if requested.
  if (out_jacobian_intrinsics) {
    out_jacobian_intrinsics->resize(2, kNumOfParams);
    const double duf_dfu = (*out_keypoint)[0];
    const double duf_dfv = 0.0;
    const double duf_dcu = 1.0;
    const double duf_dcv = 0.0;
    const double dvf_dfu = 0.0;
    const double dvf_dfv = (*out_keypoint)[1];
    const double dvf_dcu = 0.0;
    const double dvf_dcv = 1.0;

    (*out_jacobian_intrinsics) << duf_dfu, duf_dfv, duf_dcu, duf_dcv, dvf_dfu,
        dvf_dfv, dvf_dcu, dvf_dcv;
  }

  // Normalized image plane to camera plane.
  (*out_keypoint)[0] = fu * (*out_keypoint)[0] + cu;
  (*out_keypoint)[1] = fv * (*out_keypoint)[1] + cv;

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

inline const ProjectionResult
PinholeCamera::evaluateProjectionResult(const Eigen::Vector2d &keypoint,
                                        const Eigen::Vector3d &point_3d) const {

  //   Eigen::Matrix<typename DerivedKeyPoint::Scalar, 2, 1> kp = keypoint;
  const bool visibility = isKeypointVisible(keypoint);

  if (visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionResult(ProjectionResult::Status::KEYPOINT_VISIBLE);
  else if (!visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionResult(
        ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX);
  else if (point_3d[2] < 0.0)
    return ProjectionResult(ProjectionResult::Status::POINT_BEHIND_CAMERA);
  else
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}

bool PinholeCamera::isKeypointVisible(const Eigen::Vector2d &keypoint) const {
  return keypoint[0] >= static_cast<double>(0.0) &&
         keypoint[1] >= static_cast<double>(0.0) &&
         keypoint[0] < static_cast<double>(width_) &&
         keypoint[1] < static_cast<double>(height_);
}
} // namespace gmmloc