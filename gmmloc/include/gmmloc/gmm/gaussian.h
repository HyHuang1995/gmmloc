#pragma once

#include <vector>

#include <Eigen/Dense>

#include "../common/common.h"

#include "../cv/pinhole_camera.h"

namespace gmmloc {
constexpr int Dim = 3;

class GaussianComponent {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = GaussianComponent *;
  using ConstPtr = const GaussianComponent *;

  struct NeighbourInfo {
    double dist;
    double idx;
    Ptr ptr;
  };

  using Vec = Eigen::Matrix<double, Dim, 1>;
  using Mat = Eigen::Matrix<double, Dim, Dim>;

  GaussianComponent(const Vec &mean, const Mat &cov) : mean_(mean), cov_(cov) {
    cov_inv_ = cov_.inverse();

    det_ = cov_.determinant();

    det_sqrt_ = sqrt(det_);
    det_sqrt_inv_ = 1.0 / det_sqrt_;

    decompose();
  }

  ~GaussianComponent() = default;

  void decompose();

  double predict(const Vec &feature);

  double predictLog(const Vec &feature);

  double predictFull(const Vec &feature);

  double chi2(const Vec &feature);

  inline double MDist2(const Eigen::Vector3d &centre) {
    Eigen::Vector3d delta = centre - mean_;
    return delta.transpose() * cov_inv_ * delta;
  }

  const Mat &cov() const { return cov_; }

  const Mat &cov_inv() const { return cov_inv_; }

  const Vec &mean() const { return mean_; }

  const double &det() const { return det_; }

  friend std::ostream &operator<<(std::ostream &os,
                                  const GaussianComponent &hh);

public:
  std::vector<NeighbourInfo> nbs_;

  Vec mean_;

  Mat cov_, cov_inv_;

  Matrix3d sqrt_info_;

  Vec scale_;

  Mat axis_;

  Quaterniond rot_;

  double det_;

  double det_sqrt_;
  double det_sqrt_inv_;

  bool is_degenerated = false;
  bool is_salient = false;
};

class GaussianComponent2d {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<GaussianComponent2d>;

  using ConstPtr = std::shared_ptr<const GaussianComponent2d>;

  GaussianComponent2d() = delete;

  GaussianComponent2d(const GaussianComponent2d &g) = delete;

  GaussianComponent2d &operator=(const GaussianComponent2d &g) = delete;

  GaussianComponent2d(const Vector2d &mean, const Matrix2d &cov)
      : mean_(mean), cov_(cov) {
    cov_inv_ = cov_.inverse();

    det_ = cov_.determinant();

    det_sqrt_ = sqrt(det_);
    det_sqrt_inv_ = 1.0 / det_sqrt_;

    decompose();
  }

  ~GaussianComponent2d() = default;

  void decompose();

  double MDist2(Eigen::Vector2d centre) {
    Eigen::Vector2d delta = centre - mean_;
    return delta.transpose() * cov_inv_ * delta;
  }

  const Matrix2d &cov() const { return cov_; }

  const Matrix2d &cov_inv() const { return cov_inv_; }

  const Vector2d &mean() const { return mean_; }

  const Vector2d &scale() const { return scale_; }

  const double &theta() const { return theta_; }

  const double &det() const { return det_; }

public:
  uint32_t id_ = 0;

  GaussianComponent *parent_ = nullptr;

  double proj_d_ = 0.0;

  Vector2d mean_;

  Matrix2d cov_, cov_inv_;

  Vector2d scale_;

  Matrix2d axis_;

  // Quaterniond rot_;
  double theta_;

  double det_;

  double det_sqrt_;
  double det_sqrt_inv_;
};

// TODO: wrapper for dynamic memory allocation
using GaussianMixture = std::vector<GaussianComponent::Ptr>;

} // namespace gmmloc

// #include "gmm.hpp"
