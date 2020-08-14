#include "gmmloc/types/frame.h"

#include <thread>
#include <vector>

#include "gmmloc/config.h"
#include "gmmloc/cv/orb_matcher.h"

namespace gmmloc {

using namespace std;

long unsigned int Frame::next_idx_ = 0;

Frame::Frame(int a) {

  T_c_w_ = SE3QuatPtr(new SE3Quat);
  T_w_c_ = SE3QuatPtr(new SE3Quat);
  T_c_r_ = SE3QuatPtr(new SE3Quat);

  idx_ = next_idx_++;

  mbf = camera::bf;
  mb = camera::bf / camera::fx;
}

bool Frame::unproject3(size_t idx, Vector3d *pt3d) {
  if (features_[idx].depth < 0.0f)
    return false;

  camera_->unproject3(features_[idx].uv, features_[idx].depth, pt3d);

  *pt3d = getTwc().map(*pt3d);

  return true;
}

Frame::Frame(const Frame &frame)
    : timestamp_(frame.timestamp_), mbf(frame.mbf), mb(frame.mb),
      num_feats_(frame.num_feats_), features_(frame.features_),
      camera_(frame.camera_), idx_(frame.idx_),
      ref_keyframe_(frame.ref_keyframe_) {

  for (int i = 0; i < frame::grid_cols; i++)
    for (int j = 0; j < frame::grid_rows; j++)
      grid_[i][j] = frame.grid_[i][j];

  T_c_w_ = SE3QuatPtr(new SE3Quat);
  T_w_c_ = SE3QuatPtr(new SE3Quat);
  T_c_r_ = SE3QuatPtr(new SE3Quat);
  setTcw(frame.getTcw());
}

void Frame::assignFeaturesToGrid() {

  auto isPosInGrid = [&](const Vector2d &kp, int &posX, int &posY) {
    posX = round((kp.x() - 0.0f) * frame::num_grid_col_inv);
    posY = round((kp.y() - 0.0f) * frame::num_grid_row_inv);

    if (posX < 0 || posX >= frame::grid_cols || posY < 0 ||
        posY >= frame::grid_rows)
      return false;

    return true;
  };

  int nReserve = 0.5f * num_feats_ / (frame::grid_cols * frame::grid_rows);
  for (unsigned int i = 0; i < frame::grid_cols; i++)
    for (unsigned int j = 0; j < frame::grid_rows; j++)
      grid_[i][j].reserve(nReserve);

  for (size_t i = 0; i < num_feats_; i++) {
    const auto &kp = features_[i];

    int nGridPosX, nGridPosY;
    if (isPosInGrid(kp.uv, nGridPosX, nGridPosY))
      grid_[nGridPosX][nGridPosY].push_back(i);
  }
}

bool Frame::project3(const Vector3d &pt, Vector2d *uv) {
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

bool Frame::project3(const Vector3d &pt, Vector3d *uvr) {
  CHECK_NOTNULL(uvr);
  Vector2d uv;

  Vector3d ptc;
  ptc = getTcw().map(pt);

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

vector<size_t> Frame::getFeaturesInArea(const float &x, const float &y,
                                        const float &r, const int minLevel,
                                        const int maxLevel) const {
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

  const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

  for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
    for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
      const vector<size_t> vCell = grid_[ix][iy];
      if (vCell.empty())
        continue;

      for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
        const auto &kpUn = features_[vCell[j]];
        if (bCheckLevels) {
          if (kpUn.octave < minLevel)
            continue;
          if (maxLevel >= 0)
            if (kpUn.octave > maxLevel)
              continue;
        }

        const float distx = kpUn.uv.x() - x;
        const float disty = kpUn.uv.y() - y;

        if (fabs(distx) < r && fabs(disty) < r)
          vIndices.push_back(vCell[j]);
      }
    }
  }

  return vIndices;
}

void Frame::computeStereoMatches(std::vector<cv::KeyPoint> &kps_right,
                                 const cv::Mat &desc_right,
                                 const std::vector<cv::Mat> &img_pyr_left,
                                 const std::vector<cv::Mat> &img_pyr_right) {
  const int th_desc_dist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

  const int nRows = camera::height;

  // Assign keypoints to row table
  vector<vector<size_t>> vRowIndices(nRows, vector<size_t>());

  for (int i = 0; i < nRows; i++)
    vRowIndices[i].reserve(200);

  const int Nr = kps_right.size();

  for (int iR = 0; iR < Nr; iR++) {
    const cv::KeyPoint &kp = kps_right[iR];
    const float &kpY = kp.pt.y;
    const float r = 2.0f * frame::scale_factors[kps_right[iR].octave];
    const int maxr = ceil(kpY + r);
    const int minr = floor(kpY - r);

    for (int yi = minr; yi <= maxr; yi++) {
      vRowIndices[yi].push_back(iR);
    }
  }

  // Set limits for search
  const float minZ = mb;
  const float minD = 0;
  const float maxD = mbf / minZ;

  // For each left keypoint search a match in the right image
  vector<pair<int, int>> vDistIdx;
  vDistIdx.reserve(num_feats_);

  // STEP.2 serach correspondence right
  for (int iL = 0; iL < num_feats_; iL++) {
    const auto &kpL = features_[iL];
    const int &levelL = kpL.octave;
    const float &vL = kpL.uv.y();
    const float &uL = kpL.uv.x();

    const vector<size_t> &vCandidates = vRowIndices[vL];

    if (vCandidates.empty())
      continue;

    const float minU = uL - maxD;
    const float maxU = uL - minD;

    if (maxU < 0)
      continue;

    int bestDist = ORBmatcher::TH_HIGH;
    size_t bestIdxR = 0;

    const cv::Mat &dL = kpL.desc;

    // Compare descriptor to right keypoints
    for (size_t iC = 0; iC < vCandidates.size(); iC++) {
      const size_t iR = vCandidates[iC];
      const cv::KeyPoint &kpR = kps_right[iR];

      if (kpR.octave < levelL - 1 || kpR.octave > levelL + 1)
        continue;

      const float &uR = kpR.pt.x;

      if (uR >= minU && uR <= maxU) {
        const cv::Mat &dR = desc_right.row(iR);
        const int dist = ORBmatcher::DescriptorDistance(dL, dR);

        if (dist < bestDist) {
          bestDist = dist;
          bestIdxR = iR;
        }
      }
    }

    // Subpixel match by correlation
    if (bestDist < th_desc_dist) {
      // coordinates in image pyramid at keypoint scale
      const float uR0 = kps_right[bestIdxR].pt.x;
      const float scaleFactor = frame::scale_factors_inv[kpL.octave];
      const float scaleduL = round(kpL.uv.x() * scaleFactor);
      const float scaledvL = round(kpL.uv.y() * scaleFactor);
      const float scaleduR0 = round(uR0 * scaleFactor);

      // sliding window search
      const int w = 5;
      cv::Mat IL = img_pyr_left[kpL.octave]
                       .rowRange(scaledvL - w, scaledvL + w + 1)
                       .colRange(scaleduL - w, scaleduL + w + 1);
      IL.convertTo(IL, CV_32F);
      IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

      int bestDist = INT_MAX;
      int bestincR = 0;
      const int L = 5;
      vector<float> vDists;
      vDists.resize(2 * L + 1); // 11

      // FIXME:
      const float iniu = scaleduR0 + L - w;
      const float endu = scaleduR0 + L + w + 1;
      if (iniu < 0 || endu >= img_pyr_right[kpL.octave].cols)
        continue;

      for (int incR = -L; incR <= +L; incR++) {
        cv::Mat IR =
            img_pyr_right[kpL.octave]
                .rowRange(scaledvL - w, scaledvL + w + 1)
                .colRange(scaleduR0 + incR - w, scaleduR0 + incR + w + 1);
        IR.convertTo(IR, CV_32F);
        IR = IR - IR.at<float>(w, w) * cv::Mat::ones(IR.rows, IR.cols, CV_32F);

        float dist = cv::norm(IL, IR, cv::NORM_L1);
        if (dist < bestDist) {
          bestDist = dist;
          bestincR = incR;
        }

        vDists[L + incR] = dist;
      }

      if (bestincR == -L || bestincR == L)
        continue;

      const float dist1 = vDists[L + bestincR - 1];
      const float dist2 = vDists[L + bestincR];
      const float dist3 = vDists[L + bestincR + 1];

      const float deltaR =
          (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));

      if (deltaR < -1 || deltaR > 1)
        continue;

      // Re-scaled coordinate
      float bestuR = frame::scale_factors[kpL.octave] *
                     ((float)scaleduR0 + (float)bestincR + deltaR);

      float disparity = (uL - bestuR);

      if (disparity >= minD && disparity < maxD) {
        if (disparity <= 0) {
          disparity = 0.01;
          bestuR = uL - 0.01;
        }
        features_[iL].depth = mbf / disparity;
        features_[iL].u_right = bestuR;
        vDistIdx.push_back(pair<int, int>(bestDist, iL));
      }
    }
  }

  sort(vDistIdx.begin(), vDistIdx.end());
  const float median = vDistIdx[vDistIdx.size() / 2].first;
  const float thDist = 1.5f * 1.4f * median;

  for (int i = vDistIdx.size() - 1; i >= 0; i--) {
    if (vDistIdx[i].first < thDist)
      break;
    else {
      features_[vDistIdx[i].second].depth = -1.0f;
      features_[vDistIdx[i].second].u_right = -1.0f;
    }
  }
}

} // namespace gmmloc
