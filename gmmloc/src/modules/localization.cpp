#include "gmmloc/modules/localization.h"

#include <chrono>
#include <mutex>

#include "gmmloc/global.h"

#include "gmmloc/cv/orb_matcher.h"

#include "gmmloc/config.h"
#include "gmmloc/utils/timing.h"

namespace gmmloc {

using namespace std;

Localization::Localization() {
  shutdown_ = false;
  is_finished_ = true;
  is_idle_ = true;
}

void Localization::spin() {

  is_finished_ = false;

  while (true) {

    bool has_kf = checkNewKeyFrames();
    if (shutdown_ && !has_kf) {
      break;
    }

    if (has_kf) {
      is_idle_ = false;

      processNewKeyFrame();

      removeMapPoints();

      createMapPoints();

      if (!checkNewKeyFrames()) {
        searchInNeighbors();
      }

      flag_abort_ba_ = false;

      if (!checkNewKeyFrames()) {
        if (map_->countKeyFrames() > 2) {
          jointOptimization(curr_kf_, &flag_abort_ba_, map_);
        }

        removeKeyFrames();
      }
      is_idle_ = true;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }

  is_finished_ = true;
}

void Localization::spinOnce() {

  LOG(INFO) << "spinOnce";

  vector<float> vTimesTrack;
  bool timer_start = false;
  static timing::Timer timer_loc("loc", true);

  is_idle_ = false;

  if (checkNewKeyFrames()) {

    if (map_->countKeyFrames() > 2) {
      timer_start = true;
      timer_loc.Start();
    }

    timing::Timer timer_miscs("loc/miscs");

    processNewKeyFrame();

    removeMapPoints();

    createMapPoints();

    if (!checkNewKeyFrames()) {
      searchInNeighbors();
    }

    flag_abort_ba_ = false;
    timer_miscs.Stop();

    if (!checkNewKeyFrames()) {
      if (map_->countKeyFrames() > 2) {
        if (map_->countKeyFrames() > 2) {

          timing::Timer timer_ba("loc/ba");
          jointOptimization(curr_kf_, &flag_abort_ba_, map_);
          timer_ba.Stop();
        }
      }

      removeKeyFrames();

      if (timer_start) {
        timer_start = false;
        timer_loc.Stop();
      }

      if (common::verbose) {
        timing::Timing::Print(std::cout);
      }
    }
  }

  is_idle_ = true;

  std::this_thread::sleep_for(std::chrono::milliseconds(3));

  return;
}

void Localization::removeMapPoints() {
  // check recent mappoints
  list<MapPoint *>::iterator lit = candidate_mappts_.begin();
  const unsigned long int curr_idx = curr_kf_->idx_;

  const int cnThObs = 3;

  while (lit != candidate_mappts_.end()) {
    MapPoint *mappt = *lit;
    float found_ratio =
        static_cast<float>(mappt->num_found_) / mappt->num_visible_;
    if (mappt->not_valid_) {
      lit = candidate_mappts_.erase(lit);
    } else if (found_ratio < 0.25f) {
      map_->removeMapPoint(mappt);
      lit = candidate_mappts_.erase(lit);
    } else if (((int)curr_idx - (int)mappt->ref_idx_) >= 2 &&
               mappt->countObservations() <= cnThObs) {
      map_->removeMapPoint(mappt);
      lit = candidate_mappts_.erase(lit);
    } else if (((int)curr_idx - (int)mappt->ref_idx_) >= 3)
      lit = candidate_mappts_.erase(lit);
    else
      lit++;
  }
}

void Localization::searchInNeighbors() {
  const int nn = 10;
  const vector<KeyFrame *> neigh_kfs =
      curr_kf_->getBestCovisibilityKeyFrames(nn);

  unordered_set<KeyFrame *> tgt_kfs;
  {
    for (auto &&kf1 : neigh_kfs) {
      if (kf1->not_valid_ || tgt_kfs.count(kf1)) {
        continue;
      }

      tgt_kfs.emplace(kf1);

      // Extend to some second neighbors
      const vector<KeyFrame *> neigh_kfs_2nd =
          kf1->getBestCovisibilityKeyFrames(5);
      for (auto &&kf2 : neigh_kfs_2nd) {
        if (kf2->not_valid_ || tgt_kfs.count(kf2) || kf2 == curr_kf_) {
          continue;
        }

        tgt_kfs.emplace(kf2);
      }
    }
  }

  // Search matches by projection from current KF in target KFs
  vector<MapPoint *> curr_mappts = curr_kf_->getMapPoints();
  for (auto &&kf : tgt_kfs) {
    fuseObservations(kf, curr_mappts);
  }

  // Search matches by projection from target KFs in current KF
  vector<MapPoint *> fuse_candidates;
  {

    // TODO: set?
    // set<MapPoint *> set_fuse_candidates;
    for (auto &&kf : tgt_kfs) {
      vector<MapPoint *> mappts_kf = kf->getMapPoints();

      for (auto &&mappt : mappts_kf) {
        if (!mappt || mappt->not_valid_ ||
            mappt->fuse_tgt_kf_ == curr_kf_->idx_) {
          continue;
        }

        mappt->fuse_tgt_kf_ = curr_kf_->idx_;
        fuse_candidates.push_back(mappt);
      }
    }
  }
  LOG(INFO) << "#tgt_kfs: " << tgt_kfs.size()
            << " #fuse_candidates: " << fuse_candidates.size();

  fuseObservations(curr_kf_, fuse_candidates);

  // update points
  auto mappts = curr_kf_->getMapPoints();
  for (auto &&mappt : mappts) {
    if (mappt && !mappt->not_valid_) {
      mappt->computeDistinctiveDescriptors();

      mappt->updateNormalAndDepth();
    }
  }

  // Update connections in covisibility graph
  curr_kf_->updateConnections();
}

int Localization::fuseObservations(KeyFrame *kf,
                                   const std::vector<MapPoint *> &mappts,
                                   const float th) {
  // auto Twc = kf->getTwc();
  Vector3d t_w_c = kf->getTwc().translation();

  int num_fused = 0;

  const int nMPs = mappts.size();

  for (auto &&mappt : mappts) {
    if (!mappt)
      continue;

    if (mappt->not_valid_ || mappt->checkObservation(kf))
      continue;

    Vector3d pos = mappt->getPosition();

    Vector3d uvr;

    if (!kf->project3(pos, &uvr)) {
      continue;
    }

    ProjStat stat;
    if (!mappt->checkScaleAndVisible(t_w_c, stat)) {
      continue;
    }
    int lvl_pred = stat.scale_pred;

    // Search in a radius
    const float radius = th * frame::scale_factors[lvl_pred];

    const vector<size_t> indices =
        kf->getFeaturesInArea(uvr.x(), uvr.y(), radius);

    if (indices.empty())
      continue;

    // Match to the most similar keypoint in the radius
    const cv::Mat dMP = mappt->getDescriptor();

    int best_dist = 256;
    int best_idx = -1;
    for (auto &&idx : indices) {
      const auto &kp = kf->features_[idx];

      const int &kpLevel = kp.octave;

      if (kpLevel < lvl_pred - 1 || kpLevel > lvl_pred)
        continue;

      double err = kp.error(uvr);
      err *= frame::sigma2_inv[kpLevel];

      double thresh = kp.u_right >= 0 ? 7.8 : 5.99;

      if (err > thresh) {
        continue;
      }

      const cv::Mat &dKF = kp.desc;

      const int dist = ORBmatcher::DescriptorDistance(dMP, dKF);

      if (dist < best_dist) // 找MapPoint在该区域最佳匹配的特征点
      {
        best_dist = dist;
        best_idx = idx;
      }
    }

    // If there is already a MapPoint replace otherwise add new measurement
    if (best_dist <= ORBmatcher::TH_LOW) {
      MapPoint *mappt_kf = kf->getMapPoint(best_idx);
      if (mappt_kf) {
        if (!mappt_kf->not_valid_) // 如果这个MapPoint不是bad，选择哪一个呢？
        {
          if (mappt_kf->countObservations() > mappt->countObservations()) {
            // mappt->replace(mappt_kf);

            map_->replaceMapPoint(mappt, mappt_kf);
          } else {
            // mappt_kf->replace(mappt);

            map_->replaceMapPoint(mappt_kf, mappt);
          }
        }
      } else {
        mappt->addObservation(kf, best_idx);
        kf->addObservation(mappt, best_idx);
      }

      num_fused++;
    }
  }

  return num_fused;
}

int Localization::countKFsInQueue() {
  unique_lock<std::mutex> lock(mutex_kf_insertion_);
  return keyframe_queue_.size();
}

void Localization::interruptBA() { flag_abort_ba_ = true; }

void Localization::removeKeyFrames() {
  // Check redundant keyframes (only local keyframes)
  // A keyframe is considered redundant if the 90% of the MapPoints it sees, are
  // seen in at least other 3 keyframes (in the same or finer scale) We only
  // consider close stereo points

  LOG(INFO) << "enter keyframe culling.";
  vector<KeyFrame *> local_kfs = curr_kf_->getVectorCovisibleKeyFrames();

  for (auto &&kf_ptr : local_kfs) {
    if (kf_ptr->idx_ == 0)
      continue;

    const vector<MapPoint *> mappts_kf = kf_ptr->getMapPoints();

    int num_obs_ = 3;
    const int th_obs = num_obs_;
    int num_redundant_obs = 0;
    int num_mps = 0;

    for (size_t i = 0, iend = mappts_kf.size(); i < iend; i++) {
      MapPoint *mappt = mappts_kf[i];
      if (mappt && !mappt->not_valid_) {
        // stereo depth only
        if (kf_ptr->features_[i].depth > frame::th_depth ||
            kf_ptr->features_[i].depth < 0) {

          continue;
        }

        num_mps++;

        if (mappt->countObservations() > th_obs) {
          const int &scale_lvl = kf_ptr->features_[i].octave;
          const unordered_map<KeyFrame *, size_t> observations =
              mappt->getObservations();

          int num_obs_ = 0;
          for (auto &&info : observations) {

            KeyFrame *kfi_ptr = info.first;
            if (kfi_ptr == kf_ptr)
              continue;

            const int &scale_lvli = kfi_ptr->features_[info.second].octave;

            // Scale Condition
            if (scale_lvli <= scale_lvl + 1) {
              num_obs_++;
              if (num_obs_ >= th_obs)
                break;
            }
          }
          if (num_obs_ >= th_obs) {
            num_redundant_obs++;
          }
        }
      }
    }

    if (num_redundant_obs > 0.9 * num_mps) {
      map_->removeKeyFrame(kf_ptr);
    }
  }
  LOG(INFO) << "kf culling done.";
}

void Localization::insertKeyFrame(KeyFrame *kf_ptr) {
  unique_lock<mutex> lock(mutex_kf_insertion_);
  keyframe_queue_.push_back(kf_ptr);
  flag_abort_ba_ = true;
}

bool Localization::checkNewKeyFrames() {
  unique_lock<mutex> lock(mutex_kf_insertion_);
  return (!keyframe_queue_.empty());
}

void Localization::processNewKeyFrame() {
  {
    unique_lock<mutex> lock(mutex_kf_insertion_);

    curr_kf_ = keyframe_queue_.front();
    keyframe_queue_.pop_front();
  }

  // Associate MapPoints to the new keyframe and update normal and descriptor
  const vector<MapPoint *> mappts_ = curr_kf_->getMapPoints();

  // TODO: do it at insersion
  for (size_t i = 0; i < mappts_.size(); i++) {
    MapPoint *mappt = mappts_[i];
    if (mappt) {
      if (!mappt->not_valid_) {
        if (!mappt->checkObservation(curr_kf_)) {
          mappt->addObservation(curr_kf_, i);
          mappt->updateNormalAndDepth();
          mappt->computeDistinctiveDescriptors();
        } else {
          candidate_mappts_.push_back(mappt);
        }
      }
    }
  }

  // Update links in the Covisibility Graph
  curr_kf_->updateConnections();

  // Insert Keyframe in Map
  map_->addKeyFrame(curr_kf_);
}

} // namespace gmmloc
