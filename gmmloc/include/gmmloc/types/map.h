#pragma once

#include <mutex>
#include <set>

#include "frame.h"

#include "keyframe.h"
#include "mappoint.h"

#include "../gmm/gaussian_mixture.h"

namespace gmmloc {

class MapPoint;
class KeyFrame;
class Frame;

// basic frame information for visualization and analysis
struct FrameInfo {
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp;

  KeyFrame *ref = nullptr;

  SE3QuatPtr Trc;
};

struct StrOptStat {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool res = false;
  double chi2_proj, chi2_str;
  Vector3d pt_est;
};

enum StrOptResult {

};

class Map {
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Map() = default;

  void addKeyFrame(KeyFrame *kf_ptr);

  void addObservation(MapPoint *mappt);

  void insertMapPoint(MapPoint *mappt);

  void insertKeyFrame(KeyFrame *kf);

  void removeMapPoint(MapPoint *mappt);

  void removeKeyFrame(KeyFrame *kf);

  void updateLastFrame(Frame *last_frame);

  void updateFrameInfo(Frame *curr_frame);

  void replaceMapPoint(MapPoint *mp_src, MapPoint *mp_tgt);

  void setGMMMap(GMM::Ptr model) { gmm_map_ = model; }

  // for visualization
  std::vector<KeyFrame *> getAllKeyFrames();
  std::vector<MapPoint *> getAllMapPoints();

  long unsigned int countMapPoints();
  long unsigned countKeyFrames();

  void clear();

  void summarize();

  void setTrajectory(const eigen_aligned_std_vector<Quaterniond> &rot,
                     const eigen_aligned_std_vector<Vector3d> &trans);

protected:
  std::set<MapPoint *> mappoints_;
  std::set<KeyFrame *> keyframes_;

  std::mutex mutex_info_;

  std::vector<FrameInfo *> frame_info_;
  std::vector<SE3QuatConstPtr> gt_traj_;

  long unsigned int max_kf_idx_ = 0;

  std::mutex mutex_map_;

  GMM::Ptr gmm_map_ = nullptr;
};

} // namespace gmmloc
