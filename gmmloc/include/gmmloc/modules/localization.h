#pragma once

#include <atomic>
#include <mutex>

#include "../types/keyframe.h"
#include "../types/map.h"
#include "../types/mappoint.h"

#include "../gmm/gaussian_mixture.h"

namespace gmmloc {

class Localization {
public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Localization();

  void spinOnce();

  void spin();

  void insertKeyFrame(KeyFrame *kf_ptr);

  void interruptBA();

  void setModel(const GMM::Ptr &gmm) { gmm_model_ = gmm; }

  void setMap(Map *map) { map_ = map; }

  int countKFsInQueue();

public:
  void stop() { shutdown_ = true; }

  std::atomic_bool shutdown_;
  std::atomic_bool is_finished_;
  std::atomic_bool is_idle_;

protected:
  bool checkNewKeyFrames();

  void processNewKeyFrame();

  void removeMapPoints();

  void searchInNeighbors();

  int fuseObservations(KeyFrame *kf, const std::vector<MapPoint *> &mappts,
                       const float th = 3.0f);

  void removeKeyFrames();

protected:
  void createMapPoints();

  GaussianComponent *optimizeTriangulationVec(Vector3d &x3d, KeyFrame *kf_ptr,
                                              size_t idx1, size_t idx2);

  void jointOptimization(KeyFrame *kf_ptr, bool *pbStopFlag, Map *pMap);

protected:
  std::list<KeyFrame *> keyframe_queue_;

  KeyFrame *curr_kf_;

  std::list<MapPoint *> candidate_mappts_;

  std::mutex mutex_kf_insertion_;

  bool flag_abort_ba_ = false;

protected:
  GMM::Ptr gmm_model_;

  Map *map_ = nullptr;
};

} // namespace gmmloc
