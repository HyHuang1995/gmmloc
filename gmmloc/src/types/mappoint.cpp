#include "gmmloc/types/mappoint.h"

#include <mutex>

#include "gmmloc/config.h"
#include "gmmloc/global.h"

#include "gmmloc/cv/orb_matcher.h"

namespace gmmloc {

using namespace std;

long unsigned int MapPoint::next_idx_ = 0;
mutex MapPoint::global_mutex_, MapPoint::global_mutex_idx;

MapPoint::MapPoint(const Vector3d &Pos, KeyFrame *ref_kf)
    : ref_idx_(ref_kf->idx_), frame_idx_(ref_kf->frame_idx_), ref_kf_(ref_kf),
      not_valid_(false), num_visible_(1), num_found_(1) {
  type_ = FromTriMono;

  pos_ = Pos;
  normal_.setZero();

  unique_lock<mutex> lock(global_mutex_idx);
  idx_ = next_idx_++;
}

MapPoint::MapPoint(const Vector3d &Pos, Frame *frame, const int &idxF)
    : frame_idx_(frame->idx_), not_valid_(false), num_visible_(1),
      num_found_(1) {
  type_ = FromTriMono;

  pos_ = Pos;

  Vector3d Ow = frame->getTwc().translation();
  normal_ = pos_ - Ow;
  normal_.normalize();

  Vector3d PC = pos_ - Ow;
  const float dist = PC.norm();
  const int level = frame->features_[idxF].octave;
  const float levelScaleFactor = frame::scale_factors[level];
  const int nLevels = frame::num_levels;

  max_dist_ = dist * levelScaleFactor;
  min_dist_ = max_dist_ / frame::scale_factors[nLevels - 1];

  frame->features_[idxF].desc.copyTo(desc_);

  unique_lock<mutex> lock(global_mutex_idx);
  idx_ = next_idx_++;
}

void MapPoint::setPosition(const Vector3d &pos) {
  unique_lock<mutex> lock2(global_mutex_);
  unique_lock<shared_mutex> lock(mutex_pos_);
  pos_ = pos;
}

Vector3d MapPoint::getPosition() {
  shared_lock<shared_mutex> lock(mutex_pos_);

  return pos_;
}

Vector3d MapPoint::getNormal() {
  shared_lock<shared_mutex> lock(mutex_pos_);
  return normal_;
}

void MapPoint::addObservation(KeyFrame *kf_ptr, size_t idx) {
  unique_lock<shared_mutex> lock(mutex_attr_);
  if (observations_.count(kf_ptr))
    return;
  observations_[kf_ptr] = idx;

  if (kf_ptr->features_[idx].u_right >= 0)
    num_obs_ += 2;
  else
    num_obs_++;
}

unordered_map<KeyFrame *, size_t> MapPoint::getObservations() {
  shared_lock<shared_mutex> lock(mutex_attr_);
  return observations_;
}

int MapPoint::countObservations() {
  shared_lock<shared_mutex> lock(mutex_attr_);
  return num_obs_;
}

bool MapPoint::removeObservation(KeyFrame *kf_ptr) {
  bool bBad = false;
  {
    unique_lock<shared_mutex> lock(mutex_attr_);

    if (observations_.count(kf_ptr)) {
      size_t idx = observations_[kf_ptr];
      if (kf_ptr->features_[idx].u_right >= 0)
        num_obs_ -= 2;
      else
        num_obs_--;

      observations_.erase(kf_ptr);

      if (num_obs_ > 0 && ref_kf_ == kf_ptr) {
        ref_kf_ = observations_.begin()->first;
      }

      if (num_obs_ <= 2)
        bBad = true;
    }
  }

  return bBad;
}

MapPoint *MapPoint::getReplaced() {
  unique_lock<shared_mutex> lock1(mutex_attr_);
  unique_lock<shared_mutex> lock2(mutex_pos_);
  return ptr_replaced_;
}

void MapPoint::computeDistinctiveDescriptors() {
  vector<cv::Mat> vDescriptors;

  unordered_map<KeyFrame *, size_t> observations;

  {
    unique_lock<shared_mutex> lock1(mutex_attr_);
    if (not_valid_)
      return;

    observations = observations_;
  }

  if (observations.empty())
    return;

  vDescriptors.reserve(observations.size());

  for (unordered_map<KeyFrame *, size_t>::iterator mit = observations.begin(),
                                                   mend = observations.end();
       mit != mend; mit++) {
    KeyFrame *kf_ptr = mit->first;

    if (!kf_ptr->not_valid_)
      vDescriptors.push_back(kf_ptr->features_[mit->second].desc);
  }

  if (vDescriptors.empty())
    return;

  // Compute distances between them
  const size_t N = vDescriptors.size();

  std::vector<std::vector<float>> Distances;
  Distances.resize(N, vector<float>(N, 0));
  for (size_t i = 0; i < N; i++) {
    Distances[i][i] = 0;
    for (size_t j = i + 1; j < N; j++) {
      int distij =
          ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
      Distances[i][j] = distij;
      Distances[j][i] = distij;
    }
  }

  // Take the descriptor with least median distance to the rest
  int BestMedian = INT_MAX;
  int BestIdx = 0;
  for (size_t i = 0; i < N; i++) {
    vector<int> vDists(Distances[i].begin(), Distances[i].end());
    sort(vDists.begin(), vDists.end());

    int median = vDists[0.5 * (N - 1)];

    if (median < BestMedian) {
      BestMedian = median;
      BestIdx = i;
    }
  }

  {
    unique_lock<shared_mutex> lock(mutex_attr_);

    desc_ = vDescriptors[BestIdx].clone();
  }
}

cv::Mat MapPoint::getDescriptor() {
  shared_lock<shared_mutex> lock(mutex_attr_);
  return desc_.clone();
}

int MapPoint::getIndexInKeyFrame(KeyFrame *kf_ptr) {
  shared_lock<shared_mutex> lock(mutex_attr_);
  if (observations_.count(kf_ptr))
    return observations_[kf_ptr];
  else
    return -1;
}

bool MapPoint::checkObservation(KeyFrame *kf_ptr) {
  shared_lock<shared_mutex> lock(mutex_attr_);
  return (observations_.count(kf_ptr));
}

void MapPoint::updateNormalAndDepth() {
  unordered_map<KeyFrame *, size_t> observations;
  KeyFrame *pRefKF;
  Vector3d pos;
  {
    unique_lock<shared_mutex> lock1(mutex_attr_);
    shared_lock<shared_mutex> lock2(mutex_pos_);
    if (not_valid_)
      return;

    observations = observations_;
    pRefKF = ref_kf_;
    pos = pos_;
  }

  if (observations.empty())
    return;

  Vector3d normal = Vector3d::Zero();
  int n = 0;
  for (unordered_map<KeyFrame *, size_t>::iterator mit = observations.begin(),
                                                   mend = observations.end();
       mit != mend; mit++) {
    KeyFrame *kf_ptr = mit->first;
    Vector3d Owi = kf_ptr->getTwc().translation();
    Vector3d normali = pos - Owi;
    normal = normal + normali.normalized();

    n++;
  }

  Vector3d PC = pos - pRefKF->getTwc().translation();

  const float dist = PC.norm();
  const int level = pRefKF->features_[observations[pRefKF]].octave;
  const float levelScaleFactor = frame::scale_factors[level];
  const int nLevels = frame::num_levels;

  {
    shared_lock<shared_mutex> lock3(mutex_pos_);
    max_dist_ = dist * levelScaleFactor;
    min_dist_ = max_dist_ / frame::scale_factors[nLevels - 1];
    normal_ = normal / n;
  }
}

bool MapPoint::checkScaleAndVisible(const Vector3d &t_w_c, ProjStat &stat) {
  float min_dist, max_dist;
  Vector3d vec_pt_c;
  {
    shared_lock<shared_mutex> lock(mutex_pos_);
    max_dist = 1.2f * max_dist_;
    min_dist = 0.8f * min_dist_;

    vec_pt_c = pos_ - t_w_c;
  }

  const float dist = vec_pt_c.norm();

  if (dist < min_dist || dist > max_dist)
    return false;

  float view_cos;
  {
    shared_lock<shared_mutex> lock(mutex_pos_);
    view_cos = vec_pt_c.dot(normal_) / dist;
  }

  const float th_view_cos = 0.5;
  if (view_cos < th_view_cos)
    return false;

  float ratio;
  {
    shared_lock<shared_mutex> lock(mutex_pos_);
    ratio = max_dist_ / dist;
  }

  int lvl_scale = ceil(log(ratio) / frame::scale_factor_log);
  if (lvl_scale < 0)
    lvl_scale = 0;
  else if (lvl_scale >= frame::num_levels)
    lvl_scale = frame::num_levels - 1;

  stat.dist = dist;
  stat.view_cos = view_cos;
  stat.scale_pred = lvl_scale;
  return true;
}

} // namespace gmmloc
