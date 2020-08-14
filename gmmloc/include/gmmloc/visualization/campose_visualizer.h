// adapted from vins-mono

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace gmmloc {
class CameraPoseVisualizer {
public:
  using Ptr = std::shared_ptr<CameraPoseVisualizer>;

  using ConstPtr = std::shared_ptr<const CameraPoseVisualizer>;

public:
  std::string m_marker_ns;

  CameraPoseVisualizer(float r, float g, float b, float a);

  CameraPoseVisualizer(const CameraPoseVisualizer &cam) = delete;

  void setImageBoundaryColor(float r, float g, float b, float a = 1.0);
  void setOpticalCenterConnectorColor(float r, float g, float b, float a = 1.0);
  void setScale(double s);
  void setLineWidth(double width);

  void addKFPose(const Eigen::Vector3d &p, const Eigen::Quaterniond &q);
  void addPose(const Eigen::Vector3d &p, const Eigen::Quaterniond &q);
  void reset();

  void publish(ros::Publisher &pub, const std_msgs::Header &header);
  void addEdge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1);
  void addLoopEdge(const Eigen::Vector3d &p0, const Eigen::Vector3d &p1);

private:
  std::vector<visualization_msgs::Marker> m_markers;
  std_msgs::ColorRGBA m_image_boundary_color;
  std_msgs::ColorRGBA m_optical_center_connector_color;
  double m_scale;
  double m_line_width;

  static const Eigen::Vector3d imlt;
  static const Eigen::Vector3d imlb;
  static const Eigen::Vector3d imrt;
  static const Eigen::Vector3d imrb;
  static const Eigen::Vector3d oc;
  static const Eigen::Vector3d lt0;
  static const Eigen::Vector3d lt1;
  static const Eigen::Vector3d lt2;
};

} // namespace gmmloc
