#pragma once

#include <mutex>

#include "../types/map.h"

#include "campose_visualizer.h"
#include "gmm_visualizer.h"

#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/core.hpp>

namespace gmmloc {

class TrajectoryViewer {
public:
  using Ptr = std::unique_ptr<TrajectoryViewer>;

  using ConstPtr = std::unique_ptr<const TrajectoryViewer>;

public:
  TrajectoryViewer() = delete;

  TrajectoryViewer(const std::string &name, ros::NodeHandle &nh);

  ~TrajectoryViewer() = default;

  void visualize(int start = 0, int end = -1);

  // static nav_msgs::
  static geometry_msgs::PoseStamped pose2msg(const SE3Quat &pose);

  std::string name_;
  ros::Publisher pub_;
  nav_msgs::Path::Ptr msg_ = nullptr;

  std::vector<SE3QuatConstPtr> traj_;
};

class ViewerGMMLoc {
protected:
  std::atomic_bool stop_, finished_;

public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
  ViewerGMMLoc(GMM::Ptr map, ros::NodeHandle &nh);

  ros::NodeHandle nh_;

  bool stop() {
    stop_ = true;
    return stop_;
  }

  bool isFinished() { return finished_; }

  void spin();

  void switchKey(char key);

  void drawKeyFrames();

  void drawMapPoints();

  void setMap(Map *pMap) { map_ = pMap; }

  void broadcastTF();

  void setTransform(const Quaterniond &rot, const Vector3d &trans,
                    const std::string &name = "camera");

  void setTrajectory(const eigen_aligned_std_vector<Quaterniond> &rot,
                     const eigen_aligned_std_vector<Vector3d> &trans,
                     const std::string &name = "camera");

  void publishTrajectories();

  void setImage(const cv::Mat &img);

private:
  CameraPoseVisualizer::Ptr pose_viz_ = nullptr;
  Map *map_ = nullptr;

  ros::Publisher kf_pub_;
  ros::Publisher mp_pub_;

  sensor_msgs::PointCloud2::Ptr mappoints_ = nullptr;

  std::mutex mutex_pose_, mutex_gt_;

  std::unordered_map<std::string, TrajectoryViewer::Ptr> traj_records_;

  std::unordered_map<std::string, SE3QuatConstPtr> tf_records_;

  GMMVisualizer::Ptr gmm_visualizer_ = nullptr;

  std::mutex mutex_img_;
  cv::Mat image_;
};

} // namespace gmmloc
