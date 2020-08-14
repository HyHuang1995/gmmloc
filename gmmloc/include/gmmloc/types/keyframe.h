#pragma once

#include <mutex>
#include <set>

#include "orb_dbow2/dbow2/BowVector.h"
#include "orb_dbow2/dbow2/FeatureVector.h"

#include "frame.h"
#include "mappoint.h"

#include "../gmm/gaussian_mixture.h"

namespace gmmloc {

class MapPoint;
class Frame;
class Map;

struct ErrorResult {
  bool is_projection_valid = true;

  bool is_stereo = false;

  double err = 0.0;
};

class KeyFrame {
  friend class Map;

public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  KeyFrame(Frame &F);

  void updateConnections();

  void addConnection(KeyFrame *kf_ptr, const int &weight);

  void removeConnection(KeyFrame *kf_ptr);

  std::vector<KeyFrame *> getVectorCovisibleKeyFrames();
  std::vector<KeyFrame *> getBestCovisibilityKeyFrames(const int &N);
  std::vector<KeyFrame *> getCovisiblesByWeight(const int &w);

  void updateBestCovisibles();
  KeyFrame *getBestCovisibilityKeyFrame();

  void addObservation(MapPoint *mappt, const size_t &idx);

  void removeObservation(const size_t &idx);

  void removeObservation(MapPoint *mappt);

  void replaceObservation(const size_t &idx, MapPoint *mappt);

  std::vector<MapPoint *> getMapPoints();

  int countMapPoints(const int &minum_obs_);

  MapPoint *getMapPoint(const size_t &idx);

  bool project3(const Vector3d &pt, Vector2d *uv);

  bool project3(const Vector3d &pt, Vector3d *uvr);

  bool unproject3(size_t idx, Vector3d *pt3d);

  ErrorResult projectionError(const Vector3d &pt, size_t idx);

  std::vector<size_t> getFeaturesInArea(const float &x, const float &y,
                                        const float &r) const;

public:
  static long unsigned int next_idx_;
  long unsigned int idx_;
  const long unsigned int frame_idx_;

  const double timestamp_;

  std::atomic_bool not_valid_;

  PinholeCamera *camera_ = nullptr;

  // Variables used by the local mapping
  long unsigned int ba_local_kf_ = 0;
  long unsigned int fixed_kf_idx_ba = 0;

  const float mbf, mb;

  const size_t num_feats_;

  const std::vector<Feature> features_;

  DBoW2::BowVector bow_vec_;
  DBoW2::FeatureVector feat_vec_;

public:
  const SE3Quat getTcw();

  const SE3Quat getTwc();

  void setTcw(const SE3Quat &T_c_w);

  void setTwc(const SE3Quat &T_w_c);

protected:
  SE3QuatPtr T_c_w_, T_w_c_;

  std::vector<MapPoint *> mappoints_;

  std::vector<std::vector<std::vector<size_t>>> grid_;

  std::unordered_map<KeyFrame *, int> map_frame_weights_;
  std::vector<KeyFrame *> ordered_keyframes_;
  std::vector<int> ordered_weights_;

  KeyFrame *best_cov_kf_ = nullptr;

  std::shared_mutex mutex_pose_;
  std::mutex mutex_connections_;
  std::mutex mutex_attr_;

public:
  std::vector<GaussianComponents2d> comps_;
};

} // namespace gmmloc
