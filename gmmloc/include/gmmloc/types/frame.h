#pragma once

#include <vector>

#include "feature.h"
#include "keyframe.h"
#include "mappoint.h"

#include "orb_dbow2/dbow2/BowVector.h"
#include "orb_dbow2/dbow2/FeatureVector.h"

#include "../cv/pinhole_camera.h"

#include "../config.h"

#include <opencv2/core.hpp>

namespace gmmloc {

class MapPoint;
class KeyFrame;

class Frame {
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<Frame>;

  using ConstPtr = std::shared_ptr<const Frame>;

  Frame() = delete;
  explicit Frame(int a);

  // Copy constructor.
  Frame(const Frame &frame);

  ~Frame() = default;

  void computeStereoMatches(const std::vector<cv::Mat> &img_pyr_left,
                            const std::vector<cv::Mat> &img_pyr_right);

  void computeStereoMatches(std::vector<cv::KeyPoint> &kps_right,
                            const cv::Mat &desc_right,
                            const std::vector<cv::Mat> &img_pyr_left,
                            const std::vector<cv::Mat> &img_pyr_right);

  std::vector<size_t> getFeaturesInArea(const float &x, const float &y,
                                        const float &r, const int minLevel = -1,
                                        const int maxLevel = -1) const;

  void assignFeaturesToGrid();

public:
  const SE3Quat getTcw() const { return *T_c_w_; }

  const SE3Quat getTwc() const { return *T_w_c_; }

  void setTcw(const SE3Quat &T_c_w) {
    T_c_w_->setRotation(T_c_w.rotation());
    T_c_w_->setTranslation(T_c_w.translation());
    *T_w_c_ = T_c_w_->inverse();
  }

  void setTwc(const SE3Quat &T_w_c) {
    T_w_c_->setRotation(T_w_c.rotation());
    T_w_c_->setTranslation(T_w_c.translation());
    *T_c_w_ = T_w_c_->inverse();
  }

  bool project3(const Vector3d &pt, Vector2d *uv);

  bool project3(const Vector3d &pt, Vector3d *uvr);

  bool unproject3(size_t idx, Vector3d *pt3d);

public:
  static long unsigned int next_idx_; ///< Next Frame id.
  long unsigned int idx_;             ///< Current Frame id.

  bool is_keyframe_ = false;

  // Reference Keyframe.
  KeyFrame *ref_keyframe_ = nullptr;

  double timestamp_;

  float mbf, mb;

  size_t num_feats_;
  std::vector<Feature> features_;

  std::vector<MapPoint *> mappoints_;
  std::vector<bool> is_outlier_;

  DBoW2::BowVector bow_vec_;
  DBoW2::FeatureVector feat_vec_;

  PinholeCamera *camera_ = nullptr;

  std::vector<std::size_t> grid_[frame::grid_cols][frame::grid_rows];

  // protected:
  SE3QuatPtr T_c_w_, T_w_c_, T_c_r_;
};

} // namespace gmmloc
