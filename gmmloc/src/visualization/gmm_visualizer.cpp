
#include "gmmloc/visualization/gmm_visualizer.h"

#include <unordered_set>

using namespace std;

namespace gmmloc {

GMMVisualizer::GMMVisualizer(GMM::ConstPtr model, const string topic_name,
                             const string frame_id, const double cov_factor)
    : frame_id_(frame_id), msg_(nullptr), cov_factor_(cov_factor) {
  ros::NodeHandle nh("~");

  pub_ = nh.advertise<visualization_msgs::MarkerArray>(topic_name, 1, true);

  republish(model);
}

void GMMVisualizer::republish(GMM::ConstPtr gmm) {

  const auto &model = gmm->getComponents();

  if (msg_ == nullptr) {
    msg_ =
        visualization_msgs::MarkerArrayPtr(new visualization_msgs::MarkerArray);
  }

  //   visualization_msgs::MarkerArray a;
  msg_->markers.clear();
  auto stamp = ros::Time::now();

  int count = 0;
  auto marker = visualization_msgs::Marker();
  marker.header.frame_id = frame_id_;
  marker.header.seq = 0;
  marker.header.stamp = stamp;
  for (auto &g : model) {

    marker.id = count++;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = sqrt(g->scale_.x()) * cov_factor_;
    marker.scale.y = sqrt(g->scale_.y()) * cov_factor_;
    marker.scale.z = sqrt(g->scale_.z()) * cov_factor_;
    marker.color.a = 0.5;
    marker.color.r = 0.4;
    marker.color.g = 0.4;
    marker.color.b = 0.4;

    marker.pose.orientation.x = g->rot_.x();
    marker.pose.orientation.y = g->rot_.y();
    marker.pose.orientation.z = g->rot_.z();
    marker.pose.orientation.w = g->rot_.w();
    marker.pose.position.x = g->mean_.x();
    marker.pose.position.y = g->mean_.y();
    marker.pose.position.z = g->mean_.z();

    msg_->markers.push_back(marker);
  }

  pub_.publish(msg_);
}

void GMMVisualizer::republish(GMM::ConstPtr gmm,
                              const std::vector<int> &indices) {
  std::unordered_set<int> set;
  std::copy(indices.begin(), indices.end(), std::inserter(set, set.end()));

  const auto &model = gmm->getComponents();

  if (msg_ == nullptr) {
    msg_ =
        visualization_msgs::MarkerArrayPtr(new visualization_msgs::MarkerArray);
  }

  //   visualization_msgs::MarkerArray a;
  msg_->markers.clear();
  auto stamp = ros::Time::now();

  int count = 0;
  auto marker = visualization_msgs::Marker();
  marker.header.frame_id = frame_id_;
  marker.header.seq = 0;
  marker.header.stamp = stamp;
  for (int i = 0; i < model.size(); i++) {
    auto &&g = model[i];

    marker.id = count++;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = sqrt(g->scale_.x()) * cov_factor_;
    marker.scale.y = sqrt(g->scale_.y()) * cov_factor_;
    marker.scale.z = sqrt(g->scale_.z()) * cov_factor_;

    if (set.count(i)) {
      marker.color.r = 0.8;
      marker.color.g = 0.0;
      marker.color.b = 0.1;
      // marker.color.r = 0.7764705882352941;
      // marker.color.g = 0.15294117647058825;
      // marker.color.b = 0.1568627450980392;
    } else {
      marker.color.r = 0.4;
      marker.color.g = 0.4;
      marker.color.b = 0.4;
    }

    marker.color.a = 1.0;
    marker.pose.orientation.x = g->rot_.x();
    marker.pose.orientation.y = g->rot_.y();
    marker.pose.orientation.z = g->rot_.z();
    marker.pose.orientation.w = g->rot_.w();
    marker.pose.position.x = g->mean_.x();
    marker.pose.position.y = g->mean_.y();
    marker.pose.position.z = g->mean_.z();

    msg_->markers.push_back(marker);
  }

  pub_.publish(msg_);
}

} // namespace gmmloc