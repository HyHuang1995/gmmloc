#pragma once

#include <atomic>
#include <shared_mutex>

#include <opencv2/core/core.hpp>

#include "frame.h"
#include "keyframe.h"
#include "map.h"

#include "../gmm/gaussian_mixture.h"
#include "../gmm/gaussian.h"

namespace gmmloc {

class KeyFrame;
class Map;
class Frame;

struct ProjStat {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Vector3d uvr;

  double dist;

  double view_cos;

  double scale_pred;
};

class MapPoint {
  friend class Map;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum MapPointType {
    FromDepth = 0,
    FromDepthGMM = 1,

    FromTriMono = 2,
    FromTriMonoGMM = 3,

    FromTriStereo = 4,
    FromTriStereoGMM = 5
  };

  MapPoint(const Vector3d &pos, KeyFrame *ref_kf);

  MapPoint(const Vector3d &pos, Frame *frame, const int &idx);

  Vector3d getPosition();

  void setPosition(const Vector3d &pt3d);

  Vector3d getNormal();

  void addObservation(KeyFrame *kf_ptr, size_t idx);

  bool removeObservation(KeyFrame *kf_ptr);

  std::unordered_map<KeyFrame *, size_t> getObservations();

  int countObservations();

  bool checkObservation(KeyFrame *kf_ptr);

  int getIndexInKeyFrame(KeyFrame *kf_ptr);

  void computeDistinctiveDescriptors();

  cv::Mat getDescriptor();

  void updateNormalAndDepth();

  bool checkScaleAndVisible(const Vector3d &t_w_c, ProjStat &stat);

  void replace(MapPoint *mappt);

  MapPoint *getReplaced();

public:
  long unsigned int idx_; ///< Global ID for MapPoint
  static long unsigned int next_idx_;
  const long int ref_idx_ = -1;
  const long int frame_idx_;
  int num_obs_ = 0;

  bool is_in_view_;

  long unsigned int last_visible_idx_ = 0;

  long unsigned int ba_local_kf_ = 0;
  long unsigned int fuse_tgt_kf_ = 0;

  std::atomic_bool not_valid_;

  static std::mutex global_mutex_;
  static std::mutex global_mutex_idx;

  std::atomic_int num_visible_;
  std::atomic_int num_found_;

protected:
  Vector3d pos_;
  Vector3d normal_;

  std::unordered_map<KeyFrame *, size_t> observations_;

  cv::Mat desc_;

  KeyFrame *ref_kf_ = nullptr;

  MapPoint *ptr_replaced_ = nullptr;

  float min_dist_ = 0.0f, max_dist_ = 0.0f;

  std::shared_mutex mutex_pos_;
  std::shared_mutex mutex_attr_;

public:
  // associated component
  std::vector<GaussianComponent::ConstPtr> asscociations_;
  MapPointType type_;
};

} // namespace gmmloc
