#include "gmmloc/cv/orb_matcher.h"

#include <limits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "orb_dbow2/dbow2/FeatureVector.h"

#include <stdint.h>

#include "gmmloc/config.h"

#include "gmmloc/utils/math_utils.h"

namespace gmmloc {

using namespace std;

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri)
    : nn_ratio_(nnratio), check_orientation_(checkOri) {}

int ORBmatcher::searchByProjection(Frame &F, const vector<MapPoint *> &mappts,
                                   const vector<ProjStat> &stats,
                                   const float th) {
  int nmatches = 0;

  const bool bFactor = th != 1.0;

  CHECK_EQ(mappts.size(), stats.size());

  for (size_t i = 0; i < mappts.size(); i++) {
    auto &&mappt = mappts[i];
    const auto &stat = stats[i];

    if (!mappt->is_in_view_)
      continue;

    if (mappt->not_valid_)
      continue;

    const int lvl_pred = stat.scale_pred;
    const Vector3d &uvr = stat.uvr;

    float r = computeRadiusByViewingCos(stat.view_cos);

    if (bFactor)
      r *= th;

    const vector<size_t> vIndices = F.getFeaturesInArea(
        uvr.x(), uvr.y(), r * frame::scale_factors[lvl_pred], lvl_pred - 1,
        lvl_pred);

    if (vIndices.empty())
      continue;

    const cv::Mat MPdescriptor = mappt->getDescriptor();

    int bestDist = 256;
    int bestLevel = -1;
    int bestDist2 = 256;
    int bestLevel2 = -1;
    int bestIdx = -1;

    for (vector<size_t>::const_iterator vit = vIndices.begin(),
                                        vend = vIndices.end();
         vit != vend; vit++) {
      const size_t idx = *vit;

      if (F.mappoints_[idx])
        if (F.mappoints_[idx]->countObservations() > 0)
          continue;

      if (F.features_[idx].u_right > 0) {
        const float er = fabs(uvr.z() - F.features_[idx].u_right);
        if (er > r * frame::scale_factors[lvl_pred])
          continue;
      }

      const cv::Mat &d = F.features_[idx].desc;

      const int dist = DescriptorDistance(MPdescriptor, d);

      if (dist < bestDist) {
        bestDist2 = bestDist;
        bestDist = dist;
        bestLevel2 = bestLevel;
        bestLevel = F.features_[idx].octave;
        bestIdx = idx;
      } else if (dist < bestDist2) {
        bestLevel2 = F.features_[idx].octave;
        bestDist2 = dist;
      }
    }

    if (bestDist <= TH_HIGH) {
      if (bestLevel == bestLevel2 && bestDist > nn_ratio_ * bestDist2)
        continue;

      F.mappoints_[bestIdx] = mappt;
      nmatches++;
    }
  }

  return nmatches;
}

float ORBmatcher::computeRadiusByViewingCos(const float &viewCos) {
  if (viewCos > 0.998)
    return 2.5;
  else
    return 4.0;
}

bool ORBmatcher::checkEpipolarDist(const Feature &kp1, const Feature &kp2,
                                   const Matrix3d &fmat,
                                   const KeyFrame *kf2_ptr) {
  const double a =
      kp1.uv.x() * fmat(0, 0) + kp1.uv.y() * fmat(1, 0) + fmat(2, 0);
  const double b =
      kp1.uv.x() * fmat(0, 1) + kp1.uv.y() * fmat(1, 1) + fmat(2, 1);
  const double c =
      kp1.uv.x() * fmat(0, 2) + kp1.uv.y() * fmat(1, 2) + fmat(2, 2);

  const float num = a * kp2.uv.x() + b * kp2.uv.y() + c;

  const float den = a * a + b * b;

  if (den == 0)
    return false;

  const float dsqr = num * num / den;

  return dsqr < 3.84 * frame::sigma2[kp2.octave];
}

int ORBmatcher::searchForTriangulation(
    KeyFrame *kf1_ptr, KeyFrame *kf2_ptr,
    vector<pair<size_t, size_t>> &matched_pairs, const bool bOnlyStereo) {

  const DBoW2::FeatureVector &feat_vec1 = kf1_ptr->feat_vec_;
  const DBoW2::FeatureVector &feat_vec2 = kf2_ptr->feat_vec_;

  SE3Quat Tcw1 = kf1_ptr->getTcw();
  SE3Quat Tcw2 = kf2_ptr->getTcw();

  Matrix3d k1 = kf1_ptr->camera_->getCameraMatrix();
  Matrix3d k2 = kf2_ptr->camera_->getCameraMatrix();

  Matrix3d fmat = MathUtils::computeFundamentalMatrix(Tcw1, k1, Tcw2, k2);

  Vector3d C2 = Tcw2.rotation() * Tcw1.translation() + Tcw2.translation();
  const float invz = 1.0f / C2(2);
  const float ex =
      kf2_ptr->camera_->fx() * C2(0) * invz + kf2_ptr->camera_->cx();
  const float ey =
      kf2_ptr->camera_->fy() * C2(1) * invz + kf2_ptr->camera_->cy();

  int nmatches = 0;
  vector<bool> matched2(kf2_ptr->num_feats_, false);
  vector<int> matches12(kf1_ptr->num_feats_, -1);

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);

  const float factor = HISTO_LENGTH / 360.0f;

  DBoW2::FeatureVector::const_iterator f1it = feat_vec1.begin();
  DBoW2::FeatureVector::const_iterator f2it = feat_vec2.begin();
  DBoW2::FeatureVector::const_iterator f1end = feat_vec1.end();
  DBoW2::FeatureVector::const_iterator f2end = feat_vec2.end();

  while (f1it != f1end && f2it != f2end) {
    if (f1it->first == f2it->first) {
      for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++) {
        const size_t idx1 = f1it->second[i1];

        MapPoint *pMP1 = kf1_ptr->getMapPoint(idx1);

        if (pMP1)
          continue;

        const bool bStereo1 = kf1_ptr->features_[idx1].u_right >= 0;

        if (bOnlyStereo)
          if (!bStereo1)
            continue;

        const auto &kp1 = kf1_ptr->features_[idx1];

        const cv::Mat &d1 = kf1_ptr->features_[idx1].desc;

        int bestDist = TH_LOW;
        int bestIdx2 = -1;

        for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++) {
          size_t idx2 = f2it->second[i2];

          MapPoint *pMP2 = kf2_ptr->getMapPoint(idx2);

          if (matched2[idx2] || pMP2)
            continue;

          const bool bStereo2 = kf2_ptr->features_[idx2].u_right >= 0;

          if (bOnlyStereo)
            if (!bStereo2)
              continue;

          const cv::Mat &d2 = kf2_ptr->features_[idx2].desc;

          const int dist = DescriptorDistance(d1, d2);

          if (dist > TH_LOW || dist > bestDist)
            continue;

          const auto &kp2 = kf2_ptr->features_[idx2];

          if (!bStereo1 && !bStereo2) {
            const float distex = ex - kp2.uv.x();
            const float distey = ey - kp2.uv.y();
            if (distex * distex + distey * distey <
                100 * frame::scale_factors[kp2.octave])
              continue;
          }

          if (checkEpipolarDist(kp1, kp2, fmat, kf2_ptr)) {
            bestIdx2 = idx2;
            bestDist = dist;
          }
        }

        if (bestIdx2 >= 0) {
          const auto &kp2 = kf2_ptr->features_[bestIdx2];
          matches12[idx1] = bestIdx2;
          matched2[bestIdx2] = true;
          nmatches++;

          if (check_orientation_) {
            float rot = kp1.angle - kp2.angle;
            if (rot < 0.0)
              rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
              bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(idx1);
          }
        }
      }

      f1it++;
      f2it++;
    } else if (f1it->first < f2it->first) {
      f1it = feat_vec1.lower_bound(f2it->first);
    } else {
      f2it = feat_vec2.lower_bound(f1it->first);
    }
  }

  if (check_orientation_) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    computeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;
      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        matches12[rotHist[i][j]] = -1;
        nmatches--;
      }
    }
  }

  matched_pairs.clear();
  matched_pairs.reserve(nmatches);

  for (size_t i = 0, iend = matches12.size(); i < iend; i++) {
    if (matches12[i] < 0)
      continue;
    matched_pairs.push_back(make_pair(i, matches12[i]));
  }

  return nmatches;
}

int ORBmatcher::searchByBoW(KeyFrame *pKF, Frame &F,
                            vector<MapPoint *> &matches) {
  const vector<MapPoint *> mappoints = pKF->getMapPoints();

  matches = vector<MapPoint *>(F.num_feats_, nullptr);

  const DBoW2::FeatureVector &feat_vec_kf = pKF->feat_vec_;

  int nmatches = 0;

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = HISTO_LENGTH / 360.0f;

  // We perform the matching over ORB that belong to the same vocabulary node
  DBoW2::FeatureVector::const_iterator KFit = feat_vec_kf.begin();
  DBoW2::FeatureVector::const_iterator Fit = F.feat_vec_.begin();
  DBoW2::FeatureVector::const_iterator KFend = feat_vec_kf.end();
  DBoW2::FeatureVector::const_iterator Fend = F.feat_vec_.end();

  while (KFit != KFend && Fit != Fend) {
    if (KFit->first == Fit->first) {
      const vector<unsigned int> vIndicesKF = KFit->second;
      const vector<unsigned int> vIndicesF = Fit->second;

      for (size_t iKF = 0; iKF < vIndicesKF.size(); iKF++) {
        const unsigned int realIdxKF = vIndicesKF[iKF];

        MapPoint *pMP = mappoints[realIdxKF];

        if (!pMP)
          continue;

        if (pMP->not_valid_)
          continue;

        const cv::Mat &dKF = pKF->features_[realIdxKF].desc;

        int bestDist1 = 256;
        int bestIdxF = -1;
        int bestDist2 = 256;

        for (size_t iF = 0; iF < vIndicesF.size(); iF++) {
          const unsigned int realIdxF = vIndicesF[iF];

          if (matches[realIdxF])
            continue;

          const cv::Mat &dF = F.features_[realIdxF].desc;

          const int dist = DescriptorDistance(dKF, dF);

          if (dist < bestDist1) {
            bestDist2 = bestDist1;
            bestDist1 = dist;
            bestIdxF = realIdxF;
          } else if (dist < bestDist2) {
            bestDist2 = dist;
          }
        }

        if (bestDist1 <= TH_LOW) {
          if (static_cast<float>(bestDist1) <
              nn_ratio_ * static_cast<float>(bestDist2)) {
            matches[bestIdxF] = pMP;

            const auto &kp = pKF->features_[realIdxKF];

            if (check_orientation_) {
              float rot = kp.angle - F.features_[bestIdxF].angle;
              if (rot < 0.0)
                rot += 360.0f;
              int bin = round(rot * factor);
              if (bin == HISTO_LENGTH)
                bin = 0;
              assert(bin >= 0 && bin < HISTO_LENGTH);
              rotHist[bin].push_back(bestIdxF);
            }
            nmatches++;
          }
        }
      }

      KFit++;
      Fit++;
    } else if (KFit->first < Fit->first) {
      KFit = feat_vec_kf.lower_bound(Fit->first);
    } else {
      // Fit = F.mFeatVec.lower_bound(KFit->first);
      Fit = F.feat_vec_.lower_bound(KFit->first);
    }
  }

  if (check_orientation_) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    computeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i == ind1 || i == ind2 || i == ind3)
        continue;

      for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
        matches[rotHist[i][j]] = nullptr;
        nmatches--;
      }
    }
  }

  return nmatches;
}

int ORBmatcher::searchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                                   const float th, const bool bMono) {
  int nmatches = 0;

  vector<int> rotHist[HISTO_LENGTH];
  for (int i = 0; i < HISTO_LENGTH; i++)
    rotHist[i].reserve(500);
  const float factor = HISTO_LENGTH / 360.0f;

  auto Tcw = CurrentFrame.getTcw();
  auto Twc = CurrentFrame.getTwc();
  auto Tlw = LastFrame.getTcw();

  const Vector3d tlc = Tlw.rotation() * Twc.translation() + Tlw.translation();

  const bool bForward = tlc.z() > CurrentFrame.mb && !bMono;
  const bool bBackward = -tlc.z() > CurrentFrame.mb && !bMono;

  for (int i = 0; i < LastFrame.num_feats_; i++) {
    MapPoint *mappt = LastFrame.mappoints_[i];

    if (mappt) {
      if (!LastFrame.is_outlier_[i]) {
        Vector3d pt = mappt->getPosition();
        Vector3d ptc = Tcw.map(pt);

        const float xc = ptc(0);
        const float yc = ptc(1);
        const float invzc = 1.0 / ptc(2);

        if (invzc < 0)
          continue;

        float u = CurrentFrame.camera_->fx() * xc * invzc +
                  CurrentFrame.camera_->cx();
        float v = CurrentFrame.camera_->fy() * yc * invzc +
                  CurrentFrame.camera_->cy();

        if (u < 0 || u > camera::width)
          continue;
        if (v < 0 || v > camera::height)
          continue;

        int nLastOctave = LastFrame.features_[i].octave;

        // Search in a window. Size depends on scale
        float radius = th * frame::scale_factors[nLastOctave];

        vector<size_t> vIndices2;

        if (bForward)
          vIndices2 = CurrentFrame.getFeaturesInArea(u, v, radius, nLastOctave);
        else if (bBackward)
          vIndices2 =
              CurrentFrame.getFeaturesInArea(u, v, radius, 0, nLastOctave);
        else
          vIndices2 = CurrentFrame.getFeaturesInArea(
              u, v, radius, nLastOctave - 1, nLastOctave + 1);

        if (vIndices2.empty())
          continue;

        const cv::Mat dMP = mappt->getDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (vector<size_t>::const_iterator vit = vIndices2.begin(),
                                            vend = vIndices2.end();
             vit != vend; vit++) {
          const size_t i2 = *vit;
          if (CurrentFrame.mappoints_[i2])
            if (CurrentFrame.mappoints_[i2]->countObservations() > 0)
              continue;

          if (CurrentFrame.features_[i2].u_right > 0) {
            const float ur = u - CurrentFrame.mbf * invzc;
            const float er = fabs(ur - CurrentFrame.features_[i2].u_right);
            if (er > radius)
              continue;
          }

          const cv::Mat &d = CurrentFrame.features_[i2].desc;

          const int dist = DescriptorDistance(dMP, d);

          if (dist < bestDist) {
            bestDist = dist;
            bestIdx2 = i2;
          }
        }

        if (bestDist <= TH_HIGH) {
          CurrentFrame.mappoints_[bestIdx2] = mappt;
          nmatches++;

          if (check_orientation_) {
            float rot = LastFrame.features_[i].angle -
                        CurrentFrame.features_[bestIdx2].angle;
            if (rot < 0.0)
              rot += 360.0f;
            int bin = round(rot * factor);
            if (bin == HISTO_LENGTH)
              bin = 0;
            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
          }
        }
      }
    }
  }

  // Apply rotation consistency
  if (check_orientation_) {
    int ind1 = -1;
    int ind2 = -1;
    int ind3 = -1;

    computeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

    for (int i = 0; i < HISTO_LENGTH; i++) {
      if (i != ind1 && i != ind2 && i != ind3) {
        for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++) {
          CurrentFrame.mappoints_[rotHist[i][j]] =
              static_cast<MapPoint *>(NULL);
          nmatches--;
        }
      }
    }
  }

  return nmatches;
}

void ORBmatcher::computeThreeMaxima(vector<int> *histo, const int L, int &ind1,
                                    int &ind2, int &ind3) {
  int max1 = 0;
  int max2 = 0;
  int max3 = 0;

  for (int i = 0; i < L; i++) {
    const int s = histo[i].size();
    if (s > max1) {
      max3 = max2;
      max2 = max1;
      max1 = s;
      ind3 = ind2;
      ind2 = ind1;
      ind1 = i;
    } else if (s > max2) {
      max3 = max2;
      max2 = s;
      ind3 = ind2;
      ind2 = i;
    } else if (s > max3) {
      max3 = s;
      ind3 = i;
    }
  }

  if (max2 < 0.1f * (float)max1) {
    ind2 = -1;
    ind3 = -1;
  } else if (max3 < 0.1f * (float)max1) {
    ind3 = -1;
  }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b) {
  const int *pa = a.ptr<int32_t>();
  const int *pb = b.ptr<int32_t>();

  int dist = 0;

  for (int i = 0; i < 8; i++, pa++, pb++) {
    unsigned int v = *pa ^ *pb;
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
  }

  return dist;
}

} // namespace gmmloc
