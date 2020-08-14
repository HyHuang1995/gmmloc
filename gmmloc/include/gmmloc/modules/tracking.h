#pragma once

#include <opencv2/core/core.hpp>
#include <mutex>

#include "../cv/orb_extractor.h"
#include "../cv/orb_vocabulary.h"

#include "../types/frame.h"
#include "../utils/dataloader.h"

#include "localization.h"

namespace gmmloc {

struct TrackStat {
  bool res;
  int num_match_inliers;
  int num_ref_matches;
  float ratio_map;
};

class Tracking {

public:
  Tracking();

  ~Tracking() = default;

  void initialize(Frame *init_frame);

  TrackStat track(Frame *frame);

public:
  void updateLastFrame();

  void createTemporalPoints();

  void clearTemporalPoints();

  int trackWithMotionModel();

  int trackKeyFrame();

  void updateLocalMap();

  void searchLocalPoints();

  int trackLocalMap();

  int optimizeCurrentPose();

public:
  TrackStat stat_;

  Frame *curr_frame_ = nullptr, *last_frame_ = nullptr;

  // for temporal tracking
  std::list<MapPoint *> temp_pts_;

  KeyFrame *ref_keyframe_ = nullptr;

  std::vector<KeyFrame *> local_keyframes_;
  std::vector<MapPoint *> local_mappoints_;

  std::vector<ProjStat> mappoints_proj_stat;
};

} // namespace gmmloc
