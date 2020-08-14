#pragma once

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/types/sba/types_sba.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/se3_ops.h"

#include <Eigen/Geometry>

#include "gaussian.h"

namespace g2o {

// Projection using focal_length in x and y directions
class EdgeSE3QuatPrior : public BaseUnaryEdge<6, SE3Quat, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeSE3QuatPrior() = default;

  virtual void setMeasurement(const SE3Quat &m) {
    _measurement = m;
    _inverseMeasurement = m.inverse();
  }

  void computeError();

  virtual void linearizeOplus();

  bool read(std::istream &is) {}

  bool write(std::ostream &os) const {}

  SE3Quat _inverseMeasurement;
};

// Projection using focal_length in x and y directions
class EdgePt2Gaussian : public BaseUnaryEdge<3, Vector3, VertexSBAPointXYZ> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgePt2Gaussian() = default;

  void computeError();

  virtual void linearizeOplus();

  bool read(std::istream &is) {}

  bool write(std::ostream &os) const {}

  const gmmloc::GaussianComponent *comp_;
};

// Projection using focal_length in x and y directions
class EdgePt2GaussianDeg : public BaseUnaryEdge<1, double, VertexSBAPointXYZ> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgePt2GaussianDeg() = default;

  void computeError();

  virtual void linearizeOplus();

  bool read(std::istream &is) {}

  bool write(std::ostream &os) const {}

  Vector3 normal_, mean_;
};

// Projection using focal_length in x and y directions
class EdgeProjectXYZOnly
    : public BaseUnaryEdge<2, Eigen::Vector2d, VertexSBAPointXYZ> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectXYZOnly() = default;

  void computeError();

  bool isDepthPositive() {
    // const VertexSE3Expmap *v1 =
    //     static_cast<const VertexSE3Expmap *>(_vertices[1]);
    // const VertexSBAPointXYZ *v2 =
    //     static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    const VertexSBAPointXYZ *v2 =
        static_cast<const VertexSBAPointXYZ *>(_vertices[0]);

    return (rot_c_w_ * v2->estimate() + t_c_w_)(2) > 0.0;
    // return (v1->estimate().map(v2->estimate()))(2) > 0.0;
  }

  bool read(std::istream &is) {}

  bool write(std::ostream &os) const {}

  virtual void linearizeOplus();

  inline Vector2 cam_project(const Vector3 &trans_xyz) const;

  Eigen::Quaterniond rot_c_w_;
  Eigen::Vector3d t_c_w_;

  double fx, fy, cx, cy;
};

// Projection using focal_length in x and y directions
class EdgeProjectXYZOnlyStereo
    : public BaseUnaryEdge<3, Eigen::Vector3d, VertexSBAPointXYZ> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeProjectXYZOnlyStereo() = default;

  void computeError();

  bool isDepthPositive() {
    const VertexSBAPointXYZ *v2 =
        static_cast<const VertexSBAPointXYZ *>(_vertices[0]);

    return (rot_c_w_ * v2->estimate() + t_c_w_)(2) > 0.0;
  }

  bool read(std::istream &is) {}

  bool write(std::ostream &os) const {}

  virtual void linearizeOplus();

  inline Vector3 cam_project(const Vector3 &trans_xyz) const;

  Eigen::Quaterniond rot_c_w_;
  Eigen::Vector3d t_c_w_;

  double fx, fy, cx, cy, bf;
};

} // namespace g2o
