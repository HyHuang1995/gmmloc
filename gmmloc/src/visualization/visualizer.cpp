#include "gmmloc/visualization/visualizer.h"

#include <thread>

#include "gmmloc/common/common.h"

#include "gmmloc/global.h"

#include <sensor_msgs/point_cloud2_iterator.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

namespace gmmloc {

using namespace std;

TrajectoryViewer::TrajectoryViewer(const std::string &name,
                                   ros::NodeHandle &nh) {
  name_ = name;
  pub_ = nh.advertise<nav_msgs::Path>(name, 1);

  msg_ = nav_msgs::Path::Ptr(new nav_msgs::Path);

  // TODO: set frame id
  msg_->header.frame_id = "map";
  msg_->header.seq = 0;
}

geometry_msgs::PoseStamped TrajectoryViewer::pose2msg(const SE3Quat &pose) {

  geometry_msgs::PoseStamped msg;
  const auto &q = pose.rotation();
  const auto &v = pose.translation();
  msg.pose.orientation.w = q.w();
  msg.pose.orientation.x = q.x();
  msg.pose.orientation.y = q.y();
  msg.pose.orientation.z = q.z();

  msg.pose.position.x = v.x();
  msg.pose.position.y = v.y();
  msg.pose.position.z = v.z();

  return msg;
}

void TrajectoryViewer::visualize(int start, int end) {
  msg_->header.seq++;
  msg_->header.stamp = ros::Time::now();

  msg_->poses.clear();

  int start_ = start;
  int end_ = end < 0 ? traj_.size() : end;

  for (int i = start_; i < end_; i++) {
    auto msg_pose = pose2msg(*traj_[i]);
    msg_pose.header = msg_->header;
    msg_->poses.push_back(msg_pose);
  }

  pub_.publish(msg_);
}

ViewerGMMLoc::ViewerGMMLoc(GMM::Ptr map, ros::NodeHandle &nh)
    : stop_(false), finished_(true), nh_(nh) {

  gmm_visualizer_ = GMMVisualizer::Ptr(new GMMVisualizer(map));

  // setup pose visualizer
  pose_viz_ = CameraPoseVisualizer::Ptr(new CameraPoseVisualizer(
      38.0f / 255.0f, 188.0 / 255.0f, 213.0f / 255.0f, 1.0f));
  pose_viz_->setScale(0.1);
  pose_viz_->setLineWidth(0.01);

  kf_pub_ = nh.advertise<visualization_msgs::MarkerArray>("keyframes", 1);
  mp_pub_ = nh.advertise<sensor_msgs::PointCloud2>("mappoints", 1);

  mappoints_ = sensor_msgs::PointCloud2::Ptr(new sensor_msgs::PointCloud2);
  auto pcm = sensor_msgs::PointCloud2Modifier(*mappoints_);
  pcm.setPointCloud2FieldsByString(2, "xyz", "rgb");

  image_ = cv::Mat(camera::height, camera::width, CV_8UC1, cv::Scalar::all(0));
}

void ViewerGMMLoc::publishTrajectories() {
  const int pub_length = 200;

  int start = 0, end = 0;

  std::unique_lock<mutex> lock(mutex_pose_);
  if (traj_records_.count("camera")) {
    end = traj_records_["camera"]->traj_.size();
    start = (pub_length >= end) ? 0 : end - pub_length;
  }

  for (const auto &record : traj_records_) {

    record.second->visualize(start, end);
  }
}

void ViewerGMMLoc::setImage(const cv::Mat &img) {
  unique_lock<mutex> lock(mutex_img_);
  img.copyTo(image_);
}

void ViewerGMMLoc::setTransform(const Quaterniond &rot, const Vector3d &trans,
                                const std::string &name) {

  std::unique_lock<mutex> lock(mutex_pose_);
  auto pose = SE3QuatConstPtr(new SE3Quat(rot, trans));
  // pose
  tf_records_[name] = pose;

  // cout << "what" << endl;

  if (name == "camera") {
    if (!traj_records_.count("camera")) {
      traj_records_["camera"] =
          TrajectoryViewer::Ptr(new TrajectoryViewer("camera", nh_));
    }

    auto &ptr = traj_records_["camera"];

    ptr->traj_.push_back(pose);
  }
}

void ViewerGMMLoc::setTrajectory(
    const eigen_aligned_std_vector<Quaterniond> &rot,
    const eigen_aligned_std_vector<Vector3d> &trans, const std::string &name) {
  std::unique_lock<mutex> lock(mutex_pose_);
  CHECK_EQ(rot.size(), trans.size());

  // if (! traj_records_.)
  {
    if (!traj_records_.count(name)) {
      traj_records_[name] =
          TrajectoryViewer::Ptr(new TrajectoryViewer(name, nh_));
    }

    auto &ptr = traj_records_[name];
    for (size_t i = 0; i < rot.size(); i++) {
      ptr->traj_.push_back(SE3QuatConstPtr(new SE3Quat(rot[i], trans[i])));
    }
  }
}

void ViewerGMMLoc::spin() {
  finished_ = false;

  cv::namedWindow("image");

  bool bFollow = true;
  bool bLocalizationMode = false;

  ros::Rate rate(30);

  CHECK_NOTNULL(map_);
  CHECK_NOTNULL(pose_viz_);

  std_msgs::Header kf_header;
  kf_header.frame_id = "map";

  while (!stop_) {

    kf_header.seq++;
    kf_header.stamp = ros::Time::now();

    mappoints_->header = kf_header;

    pose_viz_->reset();

    drawKeyFrames();

    drawMapPoints();

    mp_pub_.publish(mappoints_);

    publishTrajectories();

    broadcastTF();

    pose_viz_->publish(kf_pub_, kf_header);

    cv::Mat img;
    {
      unique_lock<mutex> lock(mutex_img_);
      if (image_.channels() == 1) {
        cv::cvtColor(image_, img, CV_GRAY2BGR);
      } else {
        image_.copyTo(img);
      }
    }
    cv::imshow("image", img);
    switchKey(cv::waitKey(1));

    rate.sleep();
  }

  finished_ = true;
}

void ViewerGMMLoc::switchKey(char key) {
  switch (key) {
  case ' ':
    global::pause = !global::pause;
    break;
  case 's':
    global::pause = true;
    global::step = true;
    break;
  case 'q':
    global::stop = true;
    break;

  default:
    break;
  }
}

void ViewerGMMLoc::drawKeyFrames() {
  const vector<KeyFrame *> vpKFs = map_->getAllKeyFrames();

  for (size_t i = 0; i < vpKFs.size(); i++) {
    KeyFrame *kf_ptr = vpKFs[i];
    auto Twc = kf_ptr->getTwc();

    pose_viz_->addKFPose(Twc.translation(), Twc.rotation());
  }

  for (size_t i = 0; i < vpKFs.size(); i++) {
    const vector<KeyFrame *> vCovKFs = vpKFs[i]->getCovisiblesByWeight(100);

    const auto &Twc = vpKFs[i]->getTwc();
    if (!vCovKFs.empty()) {
      for (vector<KeyFrame *>::const_iterator vit = vCovKFs.begin(),
                                              vend = vCovKFs.end();
           vit != vend; vit++) {
        if ((*vit)->idx_ < vpKFs[i]->idx_)
          continue;
        const auto &Twcj = vpKFs[i]->getTwc();
        pose_viz_->addEdge(Twc.translation(), Twcj.translation());
      }
    }
  }
}

void ViewerGMMLoc::drawMapPoints() {
  const vector<MapPoint *> &vpMPs = map_->getAllMapPoints();

  size_t map_size = 0;
  for (auto &&mp : vpMPs) {
    if (mp->not_valid_)
      continue;

    map_size++;
  }

  auto pcm = sensor_msgs::PointCloud2Modifier(*mappoints_);
  pcm.resize(map_size);

  // iterators
  sensor_msgs::PointCloud2Iterator<float> out_x(*mappoints_, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(*mappoints_, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(*mappoints_, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_r(*mappoints_, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_g(*mappoints_, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> out_b(*mappoints_, "b");

  for (auto &&mp : vpMPs) {
    if (mp->not_valid_)
      continue;
    Vector3d pos = mp->getPosition();

    *out_x = pos(0);
    *out_y = pos(1);
    *out_z = pos(2);

    if (mp->type_ == MapPoint::FromDepthGMM) {
      *out_r = 198, *out_g = 39, *out_b = 40;
    } else if (mp->type_ == MapPoint::FromDepth) {
      *out_r = 31, *out_g = 119, *out_b = 180;
    } else if (mp->type_ == MapPoint::FromTriMonoGMM) {
      *out_r = 198, *out_g = 39, *out_b = 40;
    } else if (mp->type_ == MapPoint::FromTriMono) {
      *out_r = 31, *out_g = 119, *out_b = 180;
    } else if (mp->type_ == MapPoint::FromTriStereoGMM) {
      *out_r = 198, *out_g = 39, *out_b = 40;
    } else if (mp->type_ == MapPoint::FromTriStereo) {
      *out_r = 31, *out_g = 119, *out_b = 180;
    }

    ++out_x, ++out_y, ++out_z;
    ++out_r, ++out_g, ++out_b;
  }
}

void ViewerGMMLoc::broadcastTF() {
  // cout
  static tf2_ros::TransformBroadcaster br;

  unique_lock<std::mutex> lock(mutex_pose_);
  for (auto &&info : tf_records_) {
    geometry_msgs::TransformStamped transformStamped;

    transformStamped.header.stamp = ros::Time::now();
    transformStamped.header.frame_id = "map";
    transformStamped.child_frame_id = info.first;

    Vector3d t_w_c = info.second->translation();
    Quaterniond rot_w_c = info.second->rotation();
    transformStamped.transform.translation.x = t_w_c.x();
    transformStamped.transform.translation.y = t_w_c.y();
    transformStamped.transform.translation.z = t_w_c.z();

    transformStamped.transform.rotation.x = rot_w_c.x();
    transformStamped.transform.rotation.y = rot_w_c.y();
    transformStamped.transform.rotation.z = rot_w_c.z();
    transformStamped.transform.rotation.w = rot_w_c.w();
    br.sendTransform(transformStamped);
  }
}

} // namespace gmmloc
