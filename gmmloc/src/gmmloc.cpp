#include "gmmloc/gmmloc.h"

#include <chrono>
#include <iostream>

#include "glog/logging.h"

#include "gmmloc/config.h"

#include "gmmloc/cv/orb_vocabulary.h"

#include "gmmloc/global.h"

#include "gmmloc/init_config.hpp"

#include "gmmloc/utils/timing.h"

#include "gmmloc/gmm/gmm_utils.h"

namespace gmmloc {

using namespace std;

GMMLoc::GMMLoc(ros::NodeHandle &nh) : nh_(nh), pause_(true) {

  // init config
  initParameters(nh_);

  LOG(INFO) << "init parameters done.";

  // vocabulary
  bool voc_load_res = ORBVocabulary::initialize(common::voc_path);
  CHECK(voc_load_res) << "fail to load vocabulary";

  // map
  map_ = new Map();

  // camera function
  camera_ = new PinholeCamera(camera::fx, camera::fy, camera::cx, camera::cy,
                              camera::width, camera::height);

  timing::Timer timer_map("map");
  GMMUtility::loadGMMModel(common::gmm_path, gmm_model_);
  timer_map.Stop();

  gmm_model_->setCamera(camera_);

  map_->setGMMMap(gmm_model_);

  localizer_ = new Localization();
  localizer_->setMap(map_);
  localizer_->setModel(gmm_model_);

  tracker_ = new Tracking();

  if (common::online) {
    thread_loc_ =
        unique_ptr<thread>(new thread(&gmmloc::Localization::spin, localizer_));
  }

  {
    loader_ =
        new DataloaderEuRoC(common::data_path, common::gt_path,
                            DataType::GT | DataType::Mono | DataType::Depth);
  }

  loader_->getTrajectory(rot_gt_, trans_gt_);
  LOG(WARNING) << "#frames in sequence: " << loader_->getSize();

  map_->setTrajectory(rot_gt_, trans_gt_);

  if (common::viewer) {
    viewer_ = new ViewerGMMLoc(gmm_model_, nh_);
    // viewer_->setFrameDrawer(global::frame_drawer);
    viewer_->setMap(map_);
    viewer_->setTrajectory(rot_gt_, trans_gt_, "gt");

    thread_viewer_ =
        unique_ptr<thread>(new thread(&ViewerGMMLoc::spin, viewer_));
  }

  recter_ = new Rectify(common::rect_config);

  extractor_left_ = new ORBextractor(frame::num_features);
  extractor_right_ = new ORBextractor(frame::num_features);

  LOG(INFO) << "init done.";
}

GMMLoc::~GMMLoc() {
  delete camera_;

  if (viewer_) {
    delete viewer_;
  }

  delete loader_;

  delete localizer_;

  delete tracker_;

  delete recter_;

  delete extractor_left_;
  delete extractor_right_;

  auto kfs = map_->getAllKeyFrames();
  for (auto &kf : kfs) {
    if (kf)
      delete kf;
  }

  auto mps = map_->getAllMapPoints();
  for (auto &mp : mps) {
    if (mp)
      delete mp;
  }

  delete map_;
}

void GMMLoc::spin() {
  ros::Rate rate(20);

  // sizt_t ni = 0;

  while (ros::ok() && !global::stop) {

    if (!global::pause || global::step) {
      auto data_frame = loader_->getNextFrame();

      if (!data_frame)
        break;

      LOG(INFO) << "current frame idx: " << data_frame->idx
                << " stamp: " << (uint64_t)(data_frame->timestamp * 1e9);

      processFrame(data_frame);

      if (!initialized_) {
        initialize();

        tracker_->initialize(curr_frame_);

        if (common::viewer) {
          viewer_->setImage(data_frame->mono);
          viewer_->setTransform(curr_frame_->getTwc().rotation(),
                                curr_frame_->getTwc().translation(), "camera");
        }

        initialized_ = true;
      } else {
        auto track_stat = tracker_->track(curr_frame_);
        auto curr_pose = curr_frame_->getTwc();

        if (!track_stat.res) {
          break;
        }

        if (viewer_) {
          viewer_->setImage(data_frame->mono);
          viewer_->setTransform(curr_pose.rotation(), curr_pose.translation(),
                                "camera");
        }

        if (needNewKeyFrame(track_stat)) {
          curr_keyframe_ = processKeyFrame(curr_frame_);

          localizer_->insertKeyFrame(curr_keyframe_);

          if (!common::online) {
            localizer_->spinOnce();
          }
        }
      }

      // for update info
      if (!curr_frame_->ref_keyframe_) {
        curr_frame_->ref_keyframe_ = tracker_->ref_keyframe_;
      }
      map_->updateFrameInfo(curr_frame_);
    }
    if (global::step)
      global::step = false;

    // Wait to load the next frame
    ros::spinOnce();
    if (common::online) {
      rate.sleep();
    } else {
      this_thread::sleep_for(chrono::milliseconds(3));
    }
  }

  map_->summarize();
}

Frame *GMMLoc::processFrame(DataFrame::Ptr data) {
  // LOG(INFO) << "process frame";

  CHECK(!data->mono.empty());
  CHECK(!data->depth.empty());

  // image processing
  {
    // rectification
    if (camera::do_rectify) {
      CHECK_NOTNULL(recter_);

      // TODO: threading
      recter_->doRectifyL(data->mono, data->mono);
      recter_->doRectifyR(data->depth, data->depth);
    }

    // conversion
    if (data->mono.channels() == 3) {
      cv::cvtColor(data->mono, im_left_, CV_BGR2GRAY);
      cv::cvtColor(data->depth, im_right_, CV_BGR2GRAY);
    }

    // equalization
    if (camera::do_equalization) {
      cv::equalizeHist(im_left_, im_left_);
      cv::equalizeHist(im_right_, im_right_);
    }
  }

  Frame *frame = new Frame(1);
  // LOG(INFO) << "process frame: " << frame->idx_;
  frame->timestamp_ = data->timestamp;
  frame->camera_ = camera_;
  frame->mb = frame->mbf / camera_->fx();

  {
    CHECK_NOTNULL(extractor_left_);
    CHECK_NOTNULL(extractor_right_);

    vector<cv::KeyPoint> kps_left, kps_right;
    cv::Mat desc_left, desc_right;
    thread thread_left(&ORBextractor::detect, extractor_left_,
                       std::ref(im_left_), cv::Mat(), std::ref(kps_left),
                       std::ref(desc_left));
    thread thread_right(&ORBextractor::detect, extractor_right_,
                        std::ref(im_right_), cv::Mat(), std::ref(kps_right),
                        std::ref(desc_right));

    thread_left.join();
    thread_right.join();

    frame->num_feats_ = kps_left.size();

    for (size_t i = 0; i < frame->num_feats_; i++) {
      const auto &kp = kps_left[i];
      const auto &desc = desc_left.row(i);
      frame->features_.push_back(Feature(kp, desc));
    }

    frame->computeStereoMatches(kps_right, desc_right,
                                extractor_left_->mvImagePyramid,
                                extractor_right_->mvImagePyramid);

    frame->mappoints_ = vector<MapPoint *>(frame->num_feats_, nullptr);
    frame->is_outlier_ = vector<bool>(frame->num_feats_, false);

    frame->assignFeaturesToGrid();
  }

  // pose init guess
  SE3Quat init_pose;
  {
    if (curr_frame_) {
      KeyFrame *ref_kf = curr_frame_->ref_keyframe_;
      CHECK_NOTNULL(ref_kf);
      curr_frame_->setTcw((*curr_frame_->T_c_r_) * ref_kf->getTcw());
    }

    // initial pose
    if (data->idx == 0) {
      Eigen::Quaterniond rot_c_w = data->rot_w_c.conjugate();
      Eigen::Vector3d trans_c_w = -(rot_c_w * data->t_w_c);

      init_pose = SE3Quat(rot_c_w, trans_c_w);
    } else if (data->idx == 1) {
      init_pose = curr_frame_->getTcw();
    } else {

      SE3Quat delta = curr_frame_->getTcw() * last_frame_->getTwc();
      init_pose = delta * curr_frame_->getTcw();
    }
  }
  frame->setTcw(init_pose);

  if (last_frame_) {
    delete last_frame_;
    last_frame_ = nullptr;
  }
  last_frame_ = curr_frame_;
  curr_frame_ = frame;

  return frame;
}

void GMMLoc::initialize() {
  LOG(INFO) << "initialize";

  auto init_kf = processKeyFrame(curr_frame_, true);

  LOG(INFO) << "New map created with " << map_->countMapPoints() << " points"
            << endl;

  // checkMapAssociations(init_kf);

  localizer_->insertKeyFrame(init_kf);
  if (!common::online) {
    localizer_->spinOnce();
  }

  // TODO:
  curr_frame_->ref_keyframe_ = init_kf;
  curr_keyframe_ = init_kf;
}

bool GMMLoc::needNewKeyFrame(const TrackStat &stat) {
  const int num_kfs = map_->countKeyFrames();

  const float th_ref_ratio = num_kfs < 2 ? 0.4f : 0.75f;
  const float th_map_ratio = stat.num_match_inliers > 300 ? 0.2f : 0.35f;

  // Tracked MapPoints in the reference keyframe
  const int num_obs = num_kfs <= 2 ? 2 : 3;
  int num_ref_matches = tracker_->ref_keyframe_->countMapPoints(num_obs);

  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  const bool c1a = curr_frame_->idx_ >= curr_keyframe_->frame_idx_ + camera::fps;

  // Condition 1c: tracking is weak
  const bool c1b =
      stat.num_match_inliers < num_ref_matches * 0.25f || stat.ratio_map < 0.3f;

  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 = ((stat.num_match_inliers < num_ref_matches * th_ref_ratio ||
                    stat.ratio_map < th_map_ratio) &&
                   stat.num_match_inliers > 15);

  // auto bool2str = [](bool a) { return a ? "true" : "false"; };

  if ((c1a || c1b || localizer_->is_idle_) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (localizer_->is_idle_) {
      return true;
    } else {
      localizer_->interruptBA();

      if (localizer_->countKFsInQueue() < 3)
        return true;
      else
        return false;
    }
  } else
    return false;
}

void GMMLoc::stop() {

  LOG(WARNING) << "stop localizer... ";
  localizer_->stop();

  LOG(WARNING) << "stop viewer... ";
  if (viewer_) {
    viewer_->stop();
    while (!viewer_->isFinished()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    thread_viewer_->join();
  }

  LOG(WARNING) << "localizer finished... ";
  while (!localizer_->is_finished_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  LOG(WARNING) << "viewer finished... ";
  if (common::online) {
    thread_loc_->join();
  }
}

} // namespace gmmloc
