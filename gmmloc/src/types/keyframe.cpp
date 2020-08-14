#include "gmmloc/types/keyframe.h"

#include <mutex>

#include "gmmloc/cv/orb_matcher.h"

#include "gmmloc/config.h"
#include "gmmloc/global.h"

namespace gmmloc {
using namespace std;

long unsigned int KeyFrame::next_idx_ = 0;

bool KeyFrame::unproject3(size_t idx, Vector3d *pt3d) {
  if (features_[idx].depth < 0.0f)
    return false;

  camera_->unproject3(features_[idx].uv, features_[idx].depth, pt3d);

  *pt3d = getTwc().map(*pt3d);

  return true;
}

bool KeyFrame::project3(const Vector3d &pt, Vector2d *uv) {
  CHECK_NOTNULL(uv);
  Vector3d ptc;
  ptc = getTcw().map(pt);

  if (ptc.z() < 0.0) {
    return false;
  }

  auto proj_res = camera_->project3(ptc, uv);
  if (!proj_res.isKeypointVisible()) {
    return false;
  }

  return true;
}

bool KeyFrame::project3(const Vector3d &pt, Vector3d *uvr) {
  CHECK_NOTNULL(uvr);
  Vector2d uv;

  Vector3d ptc = getTcw().map(pt);
  if (ptc.z() < 0.0) {
    return false;
  }

  auto proj_res = camera_->project3(ptc, &uv);
  if (!proj_res.isKeypointVisible()) {
    return false;
  }

  const double ur = uv.x() - mbf / ptc.z();

  (*uvr) << uv, ur;

  return true;
}

ErrorResult KeyFrame::projectionError(const Vector3d &pt, size_t idx) {
  ErrorResult res;

  return res;
}

const SE3Quat KeyFrame::getTcw() {
  shared_lock<shared_mutex> lock(mutex_pose_);
  return SE3Quat(T_c_w_->rotation(), T_c_w_->translation());
};

const SE3Quat KeyFrame::getTwc() {
  shared_lock<shared_mutex> lock(mutex_pose_);

  return SE3Quat(T_w_c_->rotation(), T_w_c_->translation());
}

void KeyFrame::setTcw(const SE3Quat &T_c_w) {
  unique_lock<shared_mutex> lock(mutex_pose_);
  T_c_w_->setTranslation(T_c_w.translation());
  T_c_w_->setRotation(T_c_w.rotation());

  *T_w_c_ = T_c_w_->inverse();
};

void KeyFrame::setTwc(const SE3Quat &T_w_c) {
  unique_lock<shared_mutex> lock(mutex_pose_);
  T_w_c_->setRotation(T_w_c.rotation());
  T_w_c_->setTranslation(T_w_c.translation());
  *T_c_w_ = T_w_c_->inverse();
};

KeyFrame::KeyFrame(Frame &F)
    : frame_idx_(F.idx_), timestamp_(F.timestamp_), not_valid_(false),

      mbf(F.mbf), mb(F.mb), num_feats_(F.num_feats_), camera_(F.camera_),
      features_(F.features_), mappoints_(F.mappoints_) {

  idx_ = next_idx_++;

  grid_.resize(frame::grid_cols);
  for (int i = 0; i < frame::grid_cols; i++) {
    grid_[i].resize(frame::grid_rows);
    for (int j = 0; j < frame::grid_rows; j++)
      grid_[i][j] = F.grid_[i][j];
  }

  T_c_w_ = SE3QuatPtr(new SE3Quat);
  T_w_c_ = SE3QuatPtr(new SE3Quat);
  setTcw(F.getTcw());
}

void KeyFrame::addConnection(KeyFrame *kf_ptr, const int &weight) {
  {
    unique_lock<mutex> lock(mutex_connections_);
    if (!map_frame_weights_.count(kf_ptr))
      map_frame_weights_[kf_ptr] = weight;
    else if (map_frame_weights_[kf_ptr] != weight)
      map_frame_weights_[kf_ptr] = weight;
    else
      return;
  }

  updateBestCovisibles();
}

void KeyFrame::updateBestCovisibles() {
  unique_lock<mutex> lock(mutex_connections_);
  vector<pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(map_frame_weights_.size());
  for (unordered_map<KeyFrame *, int>::iterator
           mit = map_frame_weights_.begin(),
           mend = map_frame_weights_.end();
       mit != mend; mit++)
    vPairs.push_back(make_pair(mit->second, mit->first));

  sort(vPairs.begin(), vPairs.end());
  list<KeyFrame *> lKFs;
  list<int> lWs;
  for (size_t i = 0, iend = vPairs.size(); i < iend; i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  ordered_keyframes_ = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
  ordered_weights_ = vector<int>(lWs.begin(), lWs.end());
}

vector<KeyFrame *> KeyFrame::getVectorCovisibleKeyFrames() {
  unique_lock<mutex> lock(mutex_connections_);
  return ordered_keyframes_;
}

KeyFrame *KeyFrame::getBestCovisibilityKeyFrame() {
  unique_lock<mutex> lock(mutex_connections_);

  return best_cov_kf_;
}

vector<KeyFrame *> KeyFrame::getBestCovisibilityKeyFrames(const int &N) {
  unique_lock<mutex> lock(mutex_connections_);
  if ((int)ordered_keyframes_.size() < N)
    return ordered_keyframes_;
  else
    return vector<KeyFrame *>(ordered_keyframes_.begin(),
                              ordered_keyframes_.begin() + N);
}

vector<KeyFrame *> KeyFrame::getCovisiblesByWeight(const int &w) {
  unique_lock<mutex> lock(mutex_connections_);

  if (ordered_keyframes_.empty())
    return vector<KeyFrame *>();

  vector<int>::iterator it =
      upper_bound(ordered_weights_.begin(), ordered_weights_.end(), w,
                  [](const int &a, const int &b) { return a > b; });
  if (it == ordered_weights_.end() && *ordered_weights_.rbegin() < w)
    return vector<KeyFrame *>();
  else {
    int n = it - ordered_weights_.begin();
    return vector<KeyFrame *>(ordered_keyframes_.begin(),
                              ordered_keyframes_.begin() + n);
  }
}

void KeyFrame::addObservation(MapPoint *mappt, const size_t &idx) {
  unique_lock<mutex> lock(mutex_attr_);
  mappoints_[idx] = mappt;
}

void KeyFrame::removeObservation(const size_t &idx) {
  unique_lock<mutex> lock(mutex_attr_);
  mappoints_[idx] = nullptr;
}

void KeyFrame::removeObservation(MapPoint *mappt) {
  int idx = mappt->getIndexInKeyFrame(this);
  unique_lock<mutex> lock(mutex_attr_);
  if (idx >= 0)
    mappoints_[idx] = nullptr;
}

void KeyFrame::replaceObservation(const size_t &idx, MapPoint *mappt) {
  unique_lock<mutex> lock(mutex_attr_);
  mappoints_[idx] = mappt;
}

int KeyFrame::countMapPoints(const int &minum_obs_) {
  unique_lock<mutex> lock(mutex_attr_);

  int nPoints = 0;
  const bool bCheckObs = minum_obs_ > 0;
  for (int i = 0; i < num_feats_; i++) {
    MapPoint *mappt = mappoints_[i];
    if (mappt) {
      if (!mappt->not_valid_) {
        if (bCheckObs) {
          if (mappoints_[i]->countObservations() >= minum_obs_)
            nPoints++;
        } else
          nPoints++;
      }
    }
  }

  return nPoints;
}

vector<MapPoint *> KeyFrame::getMapPoints() {
  unique_lock<mutex> lock(mutex_attr_);
  return mappoints_;
}

MapPoint *KeyFrame::getMapPoint(const size_t &idx) {
  unique_lock<mutex> lock(mutex_attr_);
  return mappoints_[idx];
}

void KeyFrame::updateConnections() {
  unordered_map<KeyFrame *, int> KFcounter;

  vector<MapPoint *> vpMP;

  {
    unique_lock<mutex> lockMPs(mutex_attr_);
    vpMP = mappoints_;
  }

  for (vector<MapPoint *>::iterator vit = vpMP.begin(), vend = vpMP.end();
       vit != vend; vit++) {
    MapPoint *mappt = *vit;

    if (!mappt)
      continue;

    if (mappt->not_valid_)
      continue;

    unordered_map<KeyFrame *, size_t> observations = mappt->getObservations();

    for (unordered_map<KeyFrame *, size_t>::iterator mit = observations.begin(),
                                                     mend = observations.end();
         mit != mend; mit++) {
      if (mit->first->idx_ == idx_)
        continue;
      KFcounter[mit->first]++;
    }
  }

  // This should not happen
  if (KFcounter.empty())
    return;

  int nmax = 0;
  KeyFrame *pKFmax = nullptr;
  int th = 15;

  vector<pair<int, KeyFrame *>> vPairs;
  vPairs.reserve(KFcounter.size());
  for (unordered_map<KeyFrame *, int>::iterator mit = KFcounter.begin(),
                                                mend = KFcounter.end();
       mit != mend; mit++) {
    if (mit->second > nmax) {
      nmax = mit->second;
      pKFmax = mit->first;
    }
    if (mit->second >= th) {
      vPairs.push_back(make_pair(mit->second, mit->first));
      (mit->first)->addConnection(this, mit->second);
    }
  }

  if (vPairs.empty()) {
    vPairs.push_back(make_pair(nmax, pKFmax));
    pKFmax->addConnection(this, nmax);
  }

  sort(vPairs.begin(), vPairs.end());
  list<KeyFrame *> lKFs;
  list<int> lWs;
  for (size_t i = 0; i < vPairs.size(); i++) {
    lKFs.push_front(vPairs[i].second);
    lWs.push_front(vPairs[i].first);
  }

  {
    unique_lock<mutex> lockCon(mutex_connections_);
    map_frame_weights_ = KFcounter;
    ordered_keyframes_ = vector<KeyFrame *>(lKFs.begin(), lKFs.end());
    ordered_weights_ = vector<int>(lWs.begin(), lWs.end());
  }
}

void KeyFrame::removeConnection(KeyFrame *kf_ptr) {
  bool bUpdate = false;
  {
    unique_lock<mutex> lock(mutex_connections_);
    if (map_frame_weights_.count(kf_ptr)) {
      map_frame_weights_.erase(kf_ptr);
      bUpdate = true;
    }
  }

  if (bUpdate)
    updateBestCovisibles();
}

vector<size_t> KeyFrame::getFeaturesInArea(const float &x, const float &y,
                                           const float &r) const {
  vector<size_t> vIndices;
  vIndices.reserve(num_feats_);

  const int nMinCellX =
      max(0, (int)floor((x - 0.0f - r) * frame::num_grid_col_inv));
  if (nMinCellX >= frame::grid_cols)
    return vIndices;

  const int nMaxCellX =
      min((int)frame::grid_cols - 1,
          (int)ceil((x - 0.0f + r) * frame::num_grid_col_inv));
  if (nMaxCellX < 0)
    return vIndices;

  const int nMinCellY =
      max(0, (int)floor((y - 0.0f - r) * frame::num_grid_row_inv));
  if (nMinCellY >= frame::grid_rows)
    return vIndices;

  const int nMaxCellY =
      min((int)frame::grid_rows - 1,
          (int)ceil((y - 0.0f + r) * frame::num_grid_row_inv));
  if (nMaxCellY < 0)
    return vIndices;

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = grid_[ix][iy];
      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const auto &kpUn = features_[vCell[j]];
        const float distx = kpUn.uv.x() - x;
        const float disty = kpUn.uv.y() - y;

        if (fabs(distx) < r && fabs(disty) < r)
          vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

} // namespace gmmloc
