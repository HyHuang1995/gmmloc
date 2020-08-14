#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

#include "../types/frame.h"
#include "../types/keyframe.h"
#include "../types/mappoint.h"

namespace gmmloc {

class ORBmatcher {
public:
  ORBmatcher(float nnratio = 0.6, bool check_ori = true);

  // Computes the Hamming distance between two ORB descriptors
  static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

  // Search matches between Frame keypoints and projected MapPoints. Returns
  // number of matches Used to track the local map (Tracking)
  int searchByProjection(Frame &frame, const std::vector<MapPoint *> &mappts,
                         const std::vector<ProjStat> &stats,
                         const float th = 3);

  int searchByProjection(Frame &CurrentFrame, const Frame &LastFrame,
                         const float th, const bool bMono);

  int searchByBoW(KeyFrame *kf_ptr, Frame &F,
                  std::vector<MapPoint *> &vpMapPointMatches);

  int searchForTriangulation(
      KeyFrame *kf1_ptr, KeyFrame *kf2_ptr,
      std::vector<std::pair<size_t, size_t>> &vMatchedPairs,
      const bool bOnlyStereo);

public:
  static const int TH_LOW;
  static const int TH_HIGH;
  static const int HISTO_LENGTH;

protected:
  bool checkEpipolarDist(const Feature &kp1, const Feature &kp2,
                         const Matrix3d &fmat, const KeyFrame *kf_ptr);

  float computeRadiusByViewingCos(const float &viewCos);

  void computeThreeMaxima(std::vector<int> *histo, const int L, int &ind1,
                          int &ind2, int &ind3);

  float nn_ratio_;
  bool check_orientation_;
};

} // namespace gmmloc
