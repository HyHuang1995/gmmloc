#pragma once

#include "../common/common.h"

namespace gmmloc {

struct Feature {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Feature(const cv::KeyPoint &kp, const cv::Mat &_desc)
      : uv(kp.pt.x, kp.pt.y), desc(_desc.clone()), size(kp.size),
        responce(kp.response), angle(kp.angle), octave(kp.octave), depth(-1.0f),
        u_right(-1.0f)

  {}

  inline double error(const Vector2d &obs) const {
    return (uv - obs).squaredNorm();
  }

  inline double error(const Vector3d &obs) const {
    if (u_right < 0.0f) {
      return error(Vector2d(obs.x(), obs.y()));
    } else {
      Vector3d uvr(uv.x(), uv.y(), u_right);
      return (uvr - obs).squaredNorm();
    }
  }

  Vector2d uv;

  cv::Mat desc;

  float size = 0.0f;
  float responce = 0.0f;
  float angle = -1.0f;
  int octave = 0;

  float depth = -1.0f;
  float u_right = -1.0f;
};

} // namespace gmmloc
