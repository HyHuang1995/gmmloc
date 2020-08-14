#pragma once

#include "gaussian.h"
#include "gaussian_mixture.h"

namespace gmmloc {

class GMMUtility {
public:
  static double KLDivergence(const GaussianComponent &g0,
                             const GaussianComponent &g1);

  static bool loadGMMModel(const std::string &file_path, GMM::Ptr &model);

  static bool saveGMMModel(const std::string &file_path, GMM::Ptr &model,
                           bool clear_file = true);

  template <typename T> static double BHCoefficient(const T &g0, const T &g1);

  static GaussianComponent2d::Ptr
  projectGaussian(const GaussianComponent &comp, PinholeCamera::Ptr cam,
                  const Eigen::Quaterniond &rot_c_w,
                  const Eigen::Vector3d &t_c_w);

  // GaussianComponent2d::Ptr KLDivergence(const GaussianComponent &g0,
  //                            const GaussianComponent &g1);
};

// TODO: dimensionality?
template <typename T>
double GMMUtility::BHCoefficient(const T &g0, const T &g1) {

  auto &&cov0 = g0.cov();
  auto &&cov1 = g1.cov();
  // auto &&cov1_inv = g1.cov_inv();

  auto cov = (cov0 + cov1) / 2.0;

  auto &&mu0 = g0.mean();
  auto &&mu1 = g1.mean();
  decltype(mu0) delta = mu1 - mu0;

  double d0 = delta.transpose() * cov.inverse() * delta;
  d0 /= 8.0;
  double d1 = log(cov.determinant() / sqrt(g0.det() * g1.det())) / 2.0;

  // auto d2 = log(cov1.determinant() / cov0.determinant());

  double dist = d0 + d1;

  return dist;
}

} // namespace gmmloc
