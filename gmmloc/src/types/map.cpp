#include "gmmloc/types/map.h"

#include <mutex>

#include "gmmloc/common/common.h"

namespace gmmloc {

using namespace std;

void Map::addKeyFrame(KeyFrame *kf_ptr) {
  unique_lock<mutex> lock(mutex_map_);
  keyframes_.insert(kf_ptr);
  if (kf_ptr->idx_ > max_kf_idx_)
    max_kf_idx_ = kf_ptr->idx_;
}

void Map::addObservation(MapPoint *mappt) {
  unique_lock<mutex> lock(mutex_map_);
  mappoints_.insert(mappt);
}

void Map::updateFrameInfo(Frame *curr_frame) {
  LOG(INFO) << "update frame info.";

  FrameInfo *info = new FrameInfo;
  info->Trc = SE3QuatPtr(new SE3Quat);

  (*info->Trc) = curr_frame->ref_keyframe_->getTcw() * curr_frame->getTwc();
  info->timestamp = curr_frame->timestamp_;
  info->ref = curr_frame->ref_keyframe_;

  (*curr_frame->T_c_r_) = info->Trc->inverse();

  unique_lock<mutex> lock(mutex_info_);

  frame_info_.push_back(info);
}

void Map::removeMapPoint(MapPoint *mappt) {
  unordered_map<KeyFrame *, size_t> observations;
  {
    unique_lock<shared_mutex> lock1(mappt->mutex_attr_);
    unique_lock<shared_mutex> lock2(mappt->mutex_pos_);
    mappt->not_valid_ = true;
    observations = mappt->observations_;
    mappt->observations_.clear();
  }
  for (auto &&obs : observations) {
    KeyFrame *kf_ptr = obs.first;
    kf_ptr->removeObservation(obs.second);
  }

  {
    unique_lock<mutex> lock(mutex_map_);
    mappoints_.erase(mappt);
  }
}

void Map::removeKeyFrame(KeyFrame *kf_ptr) {
  {
    unique_lock<mutex> lock(kf_ptr->mutex_connections_);
    if (kf_ptr->idx_ == 0)
      return;
  }

  for (auto &&info : kf_ptr->map_frame_weights_)
    info.first->removeConnection(kf_ptr);

  for (auto &&mp : kf_ptr->mappoints_) {
    if (mp) {
      if (mp->removeObservation(kf_ptr)) {
        this->removeMapPoint(mp);
      }
    }
  }

  {
    unique_lock<mutex> lock(kf_ptr->mutex_connections_);
    unique_lock<mutex> lock1(kf_ptr->mutex_attr_);

    kf_ptr->best_cov_kf_ = kf_ptr->ordered_keyframes_[0];
    kf_ptr->map_frame_weights_.clear();
    kf_ptr->ordered_keyframes_.clear();

    kf_ptr->not_valid_ = true;
  }

  // update map information
  {
    auto kf_tgt = kf_ptr->getBestCovisibilityKeyFrame();
    CHECK(!kf_tgt->not_valid_);
    CHECK_NOTNULL(kf_tgt);

    SE3Quat Ttr = kf_tgt->getTcw() * kf_ptr->getTwc();

    unique_lock<mutex> lock(mutex_info_);
    for_each(frame_info_.begin(), frame_info_.end(), [&](FrameInfo *info) {
      if (info->ref == kf_ptr) {
        info->ref = kf_tgt;
        (*info->Trc) = Ttr * (*info->Trc);
      }
    });
  }

  {
    unique_lock<mutex> lock(mutex_map_);
    keyframes_.erase(kf_ptr);
  }
}

void Map::replaceMapPoint(MapPoint *src, MapPoint *tgt) {
  if (tgt->idx_ == src->idx_)
    return;

  int nvisible, nfound;
  unordered_map<KeyFrame *, size_t> obs;
  {
    unique_lock<shared_mutex> lock1(src->mutex_attr_);
    unique_lock<shared_mutex> lock2(src->mutex_pos_);
    obs = src->observations_;
    src->observations_.clear();
    src->not_valid_ = true;
    nvisible = src->num_visible_;
    nfound = src->num_found_;
    src->ptr_replaced_ = tgt;
  }

  for (unordered_map<KeyFrame *, size_t>::iterator mit = obs.begin(),
                                                   mend = obs.end();
       mit != mend; mit++) {
    KeyFrame *kf_ptr = mit->first;

    if (!tgt->checkObservation(kf_ptr)) {
      kf_ptr->replaceObservation(mit->second, tgt);
      tgt->addObservation(kf_ptr, mit->second);
    } else {
      kf_ptr->removeObservation(mit->second);
    }
  }

  tgt->num_visible_ += nvisible;
  tgt->num_found_ += nfound;
  tgt->computeDistinctiveDescriptors();

  {
    unique_lock<mutex> lock(mutex_map_);
    mappoints_.erase(src);
  }
}

void Map::setTrajectory(const eigen_aligned_std_vector<Quaterniond> &rot,
                        const eigen_aligned_std_vector<Vector3d> &trans) {

  CHECK_EQ(rot.size(), trans.size());

  for (size_t i = 0; i < rot.size(); i++) {
    gt_traj_.push_back(SE3QuatConstPtr(new SE3Quat(rot[i], trans[i])));
  }
}

void Map::summarize() {
  std::string file_name = common::output_path + "/traj_est.txt";
  ofstream fs(file_name);

  eigen_aligned_std_vector<SE3Quat> traj_;
  fs << std::fixed;
  for (auto &&info : frame_info_) {
    /* code */
    CHECK_NOTNULL(info->ref);

    const auto &timestamp = info->timestamp;

    const SE3Quat Twc = info->ref->getTwc() * (*info->Trc);
    const auto &rot = Twc.rotation();
    const auto &trans = Twc.translation();

    traj_.push_back(Twc);

    fs << setprecision(6) << timestamp << " " << setprecision(9) << trans.x()
       << ' ' << trans.y() << ' ' << trans.z() << ' ' << rot.x() << ' '
       << rot.y() << ' ' << rot.z() << ' ' << rot.w() << endl;
  }
  fs.close();

  LOG(WARNING) << "trajectory saved to " << file_name;

}

vector<KeyFrame *> Map::getAllKeyFrames() {
  unique_lock<mutex> lock(mutex_map_);
  return vector<KeyFrame *>(keyframes_.begin(), keyframes_.end());
}

vector<MapPoint *> Map::getAllMapPoints() {
  unique_lock<mutex> lock(mutex_map_);
  return vector<MapPoint *>(mappoints_.begin(), mappoints_.end());
}

long unsigned int Map::countMapPoints() {
  unique_lock<mutex> lock(mutex_map_);
  return mappoints_.size();
}

long unsigned int Map::countKeyFrames() {
  unique_lock<mutex> lock(mutex_map_);
  return keyframes_.size();
}

void Map::clear() {
  for (set<MapPoint *>::iterator sit = mappoints_.begin(),
                                 send = mappoints_.end();
       sit != send; sit++)
    delete *sit;

  for (set<KeyFrame *>::iterator sit = keyframes_.begin(),
                                 send = keyframes_.end();
       sit != send; sit++)
    delete *sit;

  mappoints_.clear();
  keyframes_.clear();
  max_kf_idx_ = 0;
}

} // namespace gmmloc
