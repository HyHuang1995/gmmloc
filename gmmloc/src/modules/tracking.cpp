#include "gmmloc/modules/tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <cmath>
#include <iostream>
#include <mutex>

#include <unordered_set>

#include "gmmloc/config.h"
#include "gmmloc/global.h"

#include "gmmloc/types/map.h"

#include "gmmloc/cv/orb_matcher.h"

#include "gmmloc/utils/timing.h"

namespace gmmloc {

using namespace std;

Tracking::Tracking() {}

void Tracking::initialize(Frame *init_frame) {

  last_frame_ = init_frame;

  ref_keyframe_ = init_frame->ref_keyframe_;
  local_keyframes_.push_back(ref_keyframe_);
}

TrackStat Tracking::track(Frame *frame) {
  CHECK_NOTNULL(last_frame_);

  curr_frame_ = frame;

  LOG(INFO) << last_frame_->num_feats_;

  updateLastFrame();

  if (!last_frame_->is_keyframe_) {
    createTemporalPoints();
  }

  // track with motion model
  stat_.res = true;
  {
    int num_matches = trackWithMotionModel();

    if (num_matches < 10) {
      stat_.num_match_inliers = 10;
      stat_.res = false;

      LOG(ERROR) << "tracking failure..";
    }
  }

  // track with keyframe
  if (!stat_.res) {
    int num_matches = trackKeyFrame();

    if (num_matches < 10) {
      stat_.num_match_inliers = 10;
      stat_.res = false;

      LOG(ERROR) << "tracking failure..";
      return stat_;
    }
  }

  curr_frame_->ref_keyframe_ = ref_keyframe_;
  {
    // Update Local KeyFrames and Local Points
    updateLocalMap();

    searchLocalPoints();

    int num_matches = trackLocalMap();
    stat_.num_match_inliers = num_matches;
  }

  // process statistics
  {

    int num_map = 0;
    int num_total = 0;
    for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
      if (curr_frame_->features_[i].depth > 0 &&
          curr_frame_->features_[i].depth < frame::th_depth) {
        num_total++;
        if (curr_frame_->mappoints_[i]) {
          if (curr_frame_->mappoints_[i]->countObservations() > 0) {
            num_map++;
          }
        }
      }
    }

    stat_.ratio_map = (float)num_map / (float)(std::max(1, num_total));
  }

  clearTemporalPoints();

  for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
    if (curr_frame_->mappoints_[i] && curr_frame_->is_outlier_[i]) {
      curr_frame_->mappoints_[i] = nullptr;
    }
  }

  last_frame_ = curr_frame_;
  LOG(INFO) << last_frame_ << endl;

  return stat_;
}

void Tracking::updateLocalMap() {
  LOG(INFO) << "update local map";

  // update local keyframes
  {
    unordered_set<KeyFrame *> set_local_kfs;
    unordered_map<KeyFrame *, int> kf_counter;

    for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
      if (curr_frame_->mappoints_[i]) {
        MapPoint *mappt = curr_frame_->mappoints_[i];
        if (!mappt->not_valid_) {
          const unordered_map<KeyFrame *, size_t> observations =
              mappt->getObservations();
          for (unordered_map<KeyFrame *, size_t>::const_iterator
                   it = observations.begin(),
                   itend = observations.end();
               it != itend; it++)
            kf_counter[it->first]++;
        } else {
          curr_frame_->mappoints_[i] = nullptr;
        }
      }
    }

    if (kf_counter.empty())
      return;

    int max = 0;
    KeyFrame *kf_max = nullptr;

    local_keyframes_.clear();

    for (auto &&it : kf_counter) {
      KeyFrame *kf_ptr = it.first;

      if (kf_ptr->not_valid_)
        continue;

      if (it.second > max) {
        max = it.second;
        kf_max = kf_ptr;
      }

      set_local_kfs.emplace(it.first);
    }

    for (auto &&kf_ptr : set_local_kfs) {
      if (set_local_kfs.size() > 80)
        break;

      const vector<KeyFrame *> neigh_kfs =
          kf_ptr->getBestCovisibilityKeyFrames(10);
      for (auto &&neigh_kf : neigh_kfs) {
        if (!neigh_kf->not_valid_) {
          if (set_local_kfs.count(neigh_kf)) {
            set_local_kfs.emplace(neigh_kf);
            break;
          }
        }
      }
    }

    if (kf_max) {
      ref_keyframe_ = kf_max;
      curr_frame_->ref_keyframe_ = ref_keyframe_;
    }

    local_keyframes_.assign(set_local_kfs.begin(), set_local_kfs.end());
  }

  // update local mappoints
  {

    unordered_set<MapPoint *> set_local_mps;
    for (auto &&kf_ptr : local_keyframes_) {
      // KeyFrame *kf_ptr = *itKF;
      const vector<MapPoint *> mappts = kf_ptr->getMapPoints();

      for (auto &&mappt : mappts) {
        if (mappt && !mappt->not_valid_ && !set_local_mps.count(mappt)) {
          set_local_mps.emplace(mappt);
        }
      }
    }

    local_mappoints_.clear();
    local_mappoints_.assign(set_local_mps.begin(), set_local_mps.end());
  }
}

void Tracking::searchLocalPoints() {
  LOG(INFO) << "search local poinst";

  for (vector<MapPoint *>::iterator vit = curr_frame_->mappoints_.begin(),
                                    vend = curr_frame_->mappoints_.end();
       vit != vend; vit++) {
    MapPoint *mappt = *vit;
    if (mappt) {
      if (mappt->not_valid_) {
        *vit = nullptr;
      } else {
        mappt->num_visible_++;
        mappt->last_visible_idx_ = curr_frame_->idx_;
        mappt->is_in_view_ = false;
      }
    }
  }

  int num_to_match = 0;

  Vector3d t_w_c = curr_frame_->T_w_c_->translation();

  // Project points in frame and check its visibility
  mappoints_proj_stat.resize(local_mappoints_.size());
  for (size_t i = 0; i < local_mappoints_.size(); i++) {
    auto &&mappt = local_mappoints_[i];
    auto &proj_stat = mappoints_proj_stat[i];

    if (mappt->last_visible_idx_ == curr_frame_->idx_)
      continue;
    if (mappt->not_valid_)
      continue;

    Vector3d uvr;
    mappt->is_in_view_ = false;
    if (!curr_frame_->project3(mappt->getPosition(), &uvr)) {
      continue;
    }

    if (mappt->checkScaleAndVisible(t_w_c, proj_stat)) {
      mappt->is_in_view_ = true;
      proj_stat.uvr = uvr;

      mappt->num_visible_++;
      num_to_match++;
    }
  }

  if (num_to_match > 0) {
    ORBmatcher matcher(0.8);
    int th = 3;
    if (curr_frame_->idx_ < 2)
      th = 5;

    matcher.searchByProjection(*curr_frame_, local_mappoints_,
                               mappoints_proj_stat, th);
  }
}

int Tracking::trackLocalMap() {

  LOG(INFO) << "track local map";

  // Optimize Pose
  optimizeCurrentPose();

  int num_match_inliers = 0;

  // Update MapPoints Statistics
  for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
    if (curr_frame_->mappoints_[i]) {

      if (!curr_frame_->is_outlier_[i]) {
        curr_frame_->mappoints_[i]->num_found_++;

        if (curr_frame_->mappoints_[i]->countObservations() > 0) {
          num_match_inliers++;
        }
      } else {
        curr_frame_->mappoints_[i] = nullptr;
      }
    }
  }

  return num_match_inliers;
}

int Tracking::trackKeyFrame() {
  ORBVocabulary::transform(curr_frame_);

  ORBmatcher matcher(0.7, true);
  vector<MapPoint *> mappts;

  int nmatches = matcher.searchByBoW(ref_keyframe_, *curr_frame_, mappts);

  if (nmatches < 15) {
    LOG(ERROR) << "not enough match";
  }

  curr_frame_->mappoints_ = mappts;
  curr_frame_->setTcw(last_frame_->getTcw());

  optimizeCurrentPose();

  // Discard outliers
  int num_matches_map = 0;
  for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
    if (curr_frame_->mappoints_[i]) {
      if (curr_frame_->is_outlier_[i]) {
        MapPoint *mappt = curr_frame_->mappoints_[i];

        curr_frame_->mappoints_[i] = nullptr;
        curr_frame_->is_outlier_[i] = false;
        mappt->is_in_view_ = false;
        mappt->last_visible_idx_ = curr_frame_->idx_;
        nmatches--;
      } else if (curr_frame_->mappoints_[i]->countObservations() > 0)
        num_matches_map++;
    }
  }

  return num_matches_map;
}

int Tracking::trackWithMotionModel() {
  LOG(INFO) << "track with motion model";
  ORBmatcher matcher(0.9, true);

  const int th = 7;
  int nmatches =
      matcher.searchByProjection(*curr_frame_, *last_frame_, th, false);

  LOG(INFO) << "search done. #matches: " << nmatches;

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    fill(curr_frame_->mappoints_.begin(), curr_frame_->mappoints_.end(),
         nullptr);
    nmatches = matcher.searchByProjection(*curr_frame_, *last_frame_, 2 * th,
                                          false); // 2*th
  }

  if (nmatches < 20)
    return false;

  // LOG(INFO) << curr_frame_->getTcw();
  optimizeCurrentPose();

  LOG(INFO) << "opt done.";
  int num_matches_map = 0;
  for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
    if (curr_frame_->mappoints_[i]) {
      if (curr_frame_->is_outlier_[i]) {
        MapPoint *mappt = curr_frame_->mappoints_[i];

        curr_frame_->mappoints_[i] = nullptr;
        curr_frame_->is_outlier_[i] = false;
        mappt->is_in_view_ = false;
        mappt->last_visible_idx_ = curr_frame_->idx_;
        nmatches--;
      } else if (curr_frame_->mappoints_[i]->countObservations() > 0)
        num_matches_map++;
    }
  }

  LOG(INFO) << "motion model num matches: " << num_matches_map;
  return num_matches_map;
}

void Tracking::clearTemporalPoints() {

  for (size_t i = 0; i < curr_frame_->num_feats_; i++) {
    MapPoint *mappt = curr_frame_->mappoints_[i];
    if (mappt)
      if (mappt->countObservations() < 1) {
        curr_frame_->is_outlier_[i] = false;
        curr_frame_->mappoints_[i] = nullptr;
      }
  }

  for (MapPoint *mp : temp_pts_) {
    delete mp;
  }

  temp_pts_.clear();
}

void Tracking::updateLastFrame() {
  for (size_t i = 0; i < last_frame_->num_feats_; i++) {
    MapPoint *mp = last_frame_->mappoints_[i];

    if (mp) {
      MapPoint *rep = mp->getReplaced();
      if (rep) {
        last_frame_->mappoints_[i] = rep;
      }
    }
  }
}

// TODO: 1. update pose according to reference keyframe
void Tracking::createTemporalPoints() {
  LOG(INFO) << "create temproal points";

  if (last_frame_->is_keyframe_) {
    return;
  }

  vector<pair<float, int>> depth_indices;

  for (size_t i = 0; i < last_frame_->num_feats_; i++) {
    float z = last_frame_->features_[i].depth;
    if (z > 0) {
      depth_indices.push_back(make_pair(z, i));
    }
  }

  if (depth_indices.empty())
    return;

  std::sort(depth_indices.begin(), depth_indices.end());

  int num_pts = 0;
  for (size_t j = 0; j < depth_indices.size(); j++) {
    int i = depth_indices[j].second;

    bool create_new = false;

    MapPoint *mappt = last_frame_->mappoints_[i];
    if (!mappt)
      create_new = true;
    else if (mappt->countObservations() < 1) {
      create_new = true;
    }

    if (create_new) {
      const auto &feat = last_frame_->features_[i];
      if (feat.depth > 0.0f) {
        Vector3d pt3d, ptc;
        last_frame_->camera_->unproject3(feat.uv, feat.depth, &ptc);
        pt3d = last_frame_->getTwc().map(ptc);
        MapPoint *new_mappt = new MapPoint(pt3d, last_frame_, i);

        last_frame_->mappoints_[i] = new_mappt;

        temp_pts_.push_back(new_mappt);
      }
      num_pts++;
    } else {
      num_pts++;
    }

    if (depth_indices[j].first > frame::th_depth && num_pts > 100)
      break;
  }
}

} // namespace gmmloc
