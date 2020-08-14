#include "gmmloc/gmm/gaussian.h"

namespace gmmloc {

using namespace std;

const double pi3_sqrt = sqrt((M_PI * 2.0) * (M_PI * 2.0) * (M_PI * 2.0));

const double pi3_sqrt_inv = 1.0 / pi3_sqrt;

std::ostream &operator<<(std::ostream &os, const GaussianComponent &gc) {
  os << "mean: " << gc.mean_.transpose() << endl
     << "is degenerated: " << gc.is_degenerated << endl;
  return os;
}

void GaussianComponent2d::decompose() {
  // Eigen::Matrix3d aa;
  // cov_.determinant();
  auto eigen_solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(cov_);

  axis_ = eigen_solver.eigenvectors();
  scale_ = eigen_solver.eigenvalues();

  // Matrix2d rot_mat;
  // rot_mat.col(0) = axis_.col(0).normalized();
  // rot_mat.col(1) = axis_.col(1).normalized();
  // rot_mat.col(2) = axis_.col(0).cross(axis_.col(1)).normalized();

  // rot_ = Quaterniond(rot_mat);
  theta_ = atan(axis_(1, 0) / axis_(0, 0));
}

// template <typename Dim>
// void GaussianComponent<Dim>::decompose()
void GaussianComponent::decompose() {
  // Eigen::Matrix3d aa;
  // cov_.determinant();
  auto eigen_solver = Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(cov_);

  axis_ = eigen_solver.eigenvectors();
  scale_ = eigen_solver.eigenvalues();
  // cout << scale_.transpose() << endl;
  if (scale_.x() < 1e-4) {
    is_degenerated = true;

    sqrt_info_ = cov_inv_.llt().matrixL();
  }
  sqrt_info_ = cov_inv_.llt().matrixL();

  const double scale_thresh = 0.2;
  if (scale_.y() > scale_thresh && scale_.z() > scale_thresh) {
    is_salient = true;
    // cout << scale_.transpose() << endl;
  }

  Matrix3d rot_mat;
  rot_mat.col(0) = axis_.col(0).normalized();
  rot_mat.col(1) = axis_.col(1).normalized();
  rot_mat.col(2) = axis_.col(0).cross(axis_.col(1)).normalized();

  rot_ = Quaterniond(rot_mat);
}

double GaussianComponent::chi2(const Vec &feature) {
  // return feature-mean_
  Vec delta = feature - mean_;

  return delta.transpose() * cov_inv_ * delta;
}

double GaussianComponent::predict(const Vec &feature) {
  Vec delta = feature - mean_;

  return pi3_sqrt_inv * det_sqrt_inv_ *
         exp(-0.5 * delta.transpose() * cov_inv_ * delta);
}

} // namespace gmmloc