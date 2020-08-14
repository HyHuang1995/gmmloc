#pragma once

#include "../types/feature.h"

#include "gaussian.h"

#include "../utils/nanoflann.hpp"

namespace gmmloc {

using GaussianComponents = std::vector<GaussianComponent::Ptr>;

using GaussianComponents2d = std::vector<GaussianComponent2d::Ptr>;

struct FLANNPoints3d {

  struct Point3 {
    double x, y, z;
  };
  std::vector<Point3> pts;
  inline FLANNPoints3d(const GaussianComponents &comp_) {
    Point3 pt;
    for (size_t i = 0; i < comp_.size(); i++) {
      pt.x = comp_[i]->mean().x();
      pt.y = comp_[i]->mean().y();
      pt.z = comp_[i]->mean().z();
      pts.push_back(pt);
    }
  }

  inline size_t kdtree_get_point_count() const { return pts.size(); }

  inline double kdtree_distance(const double *p1, const size_t idx_p2,
                                size_t /*size*/) const {
    const double d0 = p1[0] - pts[idx_p2].x;
    const double d1 = p1[1] - pts[idx_p2].y;
    const double d2 = p1[2] - pts[idx_p2].z;
    return d0 * d0 + d1 * d1 + d2 * d2;
  }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    else if (dim == 1)
      return pts[idx].y;
    else
      return pts[idx].z;
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

struct FLANNPoints2d {

  struct Point2 {
    double x, y;
  };
  std::vector<Point2> pts;
  inline FLANNPoints2d(const GaussianComponents2d &comp_) {
    Point2 pt;
    for (size_t i = 0; i < comp_.size(); i++) {
      pt.x = comp_[i]->mean().x();
      pt.y = comp_[i]->mean().y();
      pts.push_back(pt);
    }
  }

  inline size_t kdtree_get_point_count() const { return pts.size(); }

  inline double kdtree_distance(const double *p1, const size_t idx_p2,
                                size_t /*size*/) const {
    const double d0 = p1[0] - pts[idx_p2].x;
    const double d1 = p1[1] - pts[idx_p2].y;
    return d0 * d0 + d1 * d1;
  }

  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx].x;
    // else if (dim == 1)
    //   return pts[idx].y;
    else
      return pts[idx].y;
  }

  template <class BBOX> bool kdtree_get_bbox(BBOX & /* bb */) const {
    return false;
  }
};

using KDTreePoints2d = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, FLANNPoints2d>, FLANNPoints2d, 2>;
using KDTreePoints3d = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, FLANNPoints3d>, FLANNPoints3d, 3>;

class GMM {

public:
  using Ptr = std::shared_ptr<GMM>;

  using ConstPtr = std::shared_ptr<const GMM>;

  // using Ptr = GMM *;

  // using ConstPtr = const GMM *;

public:
  explicit GMM(const GaussianComponents &components);

  GMM(/* args */) = delete;

  GMM(const GMM &other) = delete;

  void operator=(const GMM &) = delete;

  ~GMM();

  void renderViewProb(const Quaterniond &rot_c_w, const Vector3d &t_c_w);

  void renderView(const Quaterniond &rot_c_w, const Vector3d &t_c_w);

  void renderViewCovarianceCheck(const Quaterniond &rot_c_w,
                                 const Vector3d &t_c_w);

  void renderView(const Quaterniond &rot_c_w, const Vector3d &t_c_w,
                  std::vector<uint32_t> indices, bool sort_by_depth = false);

  std::unique_ptr<KDTreePoints2d> genKDTree2d();

  void searchCorrespondence(const std::vector<Feature> &kpts,
                            GaussianComponents2d &comps);

  void searchCorrespondence(const std::vector<Feature> &kpts,
                            std::vector<GaussianComponents2d> &comps,
                            int num = 5);

  void visualize2d(cv::Mat &viz_img) const;

public:
  // get set functions
  void setCamera(PinholeCamera::Ptr cam) { camera_ = cam; }

  const GaussianComponents &getComponents() const { return components_; }

  GaussianComponent::ConstPtr getComponent3d(size_t idx) const {
    return components_[idx];
  }

  const GaussianComponents2d &getComponents2d() const { return components2d_; }

  size_t countComponents() { return components_.size(); }

private:
  PinholeCamera::Ptr camera_ = nullptr;

  uint32_t num_;
  GaussianComponents components_;

  GaussianComponents2d components2d_;

  KDTreePoints3d *kdtree3d_ = nullptr;
  FLANNPoints3d *pts3d_ = nullptr;

public:
  void buildKDTree();

  void queryPoint(const Eigen::Vector3d &pt, std::vector<int> &res);
};

} // namespace gmmloc
