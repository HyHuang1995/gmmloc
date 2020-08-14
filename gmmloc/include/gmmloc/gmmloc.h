#pragma once

#include <string>
#include <thread>

#include <opencv2/core/core.hpp>

#include "modules/localization.h"
#include "modules/tracking.h"

#include "visualization/visualizer.h"

#include "gmm/gaussian_mixture.h"

#include "utils/dataloader.h"
#include "utils/cv_utils.h"

#include "cv/orb_extractor.h"

namespace gmmloc {

class GMMLoc {
public:
  explicit GMMLoc(ros::NodeHandle &nh);

  ~GMMLoc();

  void spin();

  Frame *processFrame(DataFrame::Ptr data);

  bool needNewKeyFrame(const TrackStat &stat);

  KeyFrame *processNewKeyFrame(Frame *frame);

  KeyFrame *processKeyFrame(Frame *frame, bool is_first = false);

  void createMapPointsFromStereo(Frame *frame, KeyFrame *kf_ptr,
                                 bool check_depth = true);

  void associateMapElements(KeyFrame *kf);

  static StrOptStat optimizePoint(const Eigen::Vector3d &pt3d,
                                  const Feature &kp, const SE3Quat &Tcw,
                                  GaussianComponent::ConstPtr comp,
                                  const double proj_z);

  GaussianComponent::ConstPtr checkMapAssociation(Vector3d &pt3d, KeyFrame *kf,
                                                  size_t idx);

  void initialize();

  void stop();

  ros::NodeHandle nh_;

  bool initialized_ = false;

  cv::Mat im_left_, im_right_;

  Frame *curr_frame_ = nullptr, *last_frame_ = nullptr;
  KeyFrame *curr_keyframe_;

  Map *map_ = nullptr;

  Rectify *recter_ = nullptr;

  // TOOD: smart pointer?
  ORBextractor *extractor_left_ = nullptr, *extractor_right_ = nullptr;
  Tracking *tracker_ = nullptr;

  Localization *localizer_ = nullptr;

  ViewerGMMLoc *viewer_ = nullptr;

  PinholeCamera *camera_;

  // model
  GMM::Ptr gmm_model_ = nullptr;

  // loader
  Dataloader *loader_;

  std::unique_ptr<std::thread> thread_loc_ = nullptr;
  std::unique_ptr<std::thread> thread_viewer_ = nullptr;

  atomic_bool pause_;

  eigen_aligned_std_vector<Quaterniond> rot_gt_;
  eigen_aligned_std_vector<Vector3d> trans_gt_;
};

} // namespace gmmloc
