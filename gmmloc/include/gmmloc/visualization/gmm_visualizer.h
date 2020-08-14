#pragma once

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>

#include "../gmm/gaussian_mixture.h"

namespace gmmloc {

class GMMVisualizer {

public:
  using Ptr = std::unique_ptr<GMMVisualizer>;

  using ConstPtr = std::unique_ptr<const GMMVisualizer>;

  //   GMMVisualizer(const GaussianMixture& model, ros::NodeHandle& nh);
  explicit GMMVisualizer(GMM::ConstPtr model,
                         const std::string topic_name = "gmm_model",
                         const std::string frame_id = "map",
                         const double cov_factor = 3);

  ~GMMVisualizer() = default;

  void republish(GMM::ConstPtr model);

  void republish(GMM::ConstPtr model, const std::vector<int>& indices);

public:
  std::string frame_id_;
  ros::Publisher pub_;

  visualization_msgs::MarkerArrayPtr msg_;

  double cov_factor_;
};

} // namespace gmmloc
