// adapted from vins-mono
#include "gmmloc/visualization/campose_visualizer.h"

namespace gmmloc {
const Eigen::Vector3d CameraPoseVisualizer::imlt =
    Eigen::Vector3d(-1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::imrt =
    Eigen::Vector3d(1.0, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::imlb =
    Eigen::Vector3d(-1.0, 0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::imrb =
    Eigen::Vector3d(1.0, 0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::lt0 =
    Eigen::Vector3d(-0.7, -0.5, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::lt1 =
    Eigen::Vector3d(-0.7, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::lt2 =
    Eigen::Vector3d(-1.0, -0.2, 1.0);
const Eigen::Vector3d CameraPoseVisualizer::oc = Eigen::Vector3d(0.0, 0.0, 0.0);

void Eigen2Point(const Eigen::Vector3d &v, geometry_msgs::Point &p) {
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
}

CameraPoseVisualizer::CameraPoseVisualizer(float r, float g, float b, float a)
    : m_marker_ns("cam_pose"), m_scale(0.2), m_line_width(0.01) {
  m_image_boundary_color.r = r;
  m_image_boundary_color.g = g;
  m_image_boundary_color.b = b;
  m_image_boundary_color.a = a;
  m_optical_center_connector_color.r = r;
  m_optical_center_connector_color.g = g;
  m_optical_center_connector_color.b = b;
  m_optical_center_connector_color.a = a;
}

void CameraPoseVisualizer::setImageBoundaryColor(float r, float g, float b,
                                                 float a) {
  m_image_boundary_color.r = r;
  m_image_boundary_color.g = g;
  m_image_boundary_color.b = b;
  m_image_boundary_color.a = a;
}

void CameraPoseVisualizer::setOpticalCenterConnectorColor(float r, float g,
                                                          float b, float a) {
  m_optical_center_connector_color.r = r;
  m_optical_center_connector_color.g = g;
  m_optical_center_connector_color.b = b;
  m_optical_center_connector_color.a = a;
}

void CameraPoseVisualizer::setScale(double s) { m_scale = s; }
void CameraPoseVisualizer::setLineWidth(double width) { m_line_width = width; }
void CameraPoseVisualizer::addEdge(const Eigen::Vector3d &p0,
                                   const Eigen::Vector3d &p1) {
  visualization_msgs::Marker marker;

  marker.ns = "edge";
  marker.id = m_markers.size() + 1;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.005;
  marker.pose.orientation.w = 1.0f;
  marker.pose.orientation.x = 0.0f;
  marker.pose.orientation.y = 0.0f;
  marker.pose.orientation.z = 0.0f;

  marker.color.g = 1.0f;
  marker.color.a = 1.0;

  geometry_msgs::Point point0, point1;

  Eigen2Point(p0, point0);
  Eigen2Point(p1, point1);

  marker.points.push_back(point0);
  marker.points.push_back(point1);

  m_markers.push_back(marker);
}

void CameraPoseVisualizer::addLoopEdge(const Eigen::Vector3d &p0,
                                       const Eigen::Vector3d &p1) {
  visualization_msgs::Marker marker;

  marker.ns = m_marker_ns;
  marker.id = m_markers.size() + 1;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = 0.04;
  // marker.scale.x = 0.3;

  marker.color.r = 1.0f;
  marker.color.b = 1.0f;
  marker.color.a = 1.0;

  geometry_msgs::Point point0, point1;

  Eigen2Point(p0, point0);
  Eigen2Point(p1, point1);

  marker.points.push_back(point0);
  marker.points.push_back(point1);

  m_markers.push_back(marker);
}

void CameraPoseVisualizer::addPose(const Eigen::Vector3d &p,
                                   const Eigen::Quaterniond &q) {
  visualization_msgs::Marker marker;

  marker.ns = "camera";
  marker.id = m_markers.size() + 1;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = m_line_width;

  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  geometry_msgs::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1,
      pt_lt2;

  Eigen2Point(q * (m_scale * imlt) + p, pt_lt);
  Eigen2Point(q * (m_scale * imlb) + p, pt_lb);
  Eigen2Point(q * (m_scale * imrt) + p, pt_rt);
  Eigen2Point(q * (m_scale * imrb) + p, pt_rb);
  Eigen2Point(q * (m_scale * lt0) + p, pt_lt0);
  Eigen2Point(q * (m_scale * lt1) + p, pt_lt1);
  Eigen2Point(q * (m_scale * lt2) + p, pt_lt2);
  Eigen2Point(q * (m_scale * oc) + p, pt_oc);

  // image boundaries
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_lb);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_rb);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_rt);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_lt);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  // top-left indicator
  marker.points.push_back(pt_lt0);
  marker.points.push_back(pt_lt1);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_lt1);
  marker.points.push_back(pt_lt2);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  // optical center connector
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  m_markers.push_back(marker);
}

void CameraPoseVisualizer::addKFPose(const Eigen::Vector3d &p,
                                     const Eigen::Quaterniond &q) {
  visualization_msgs::Marker marker;

  marker.ns = "keyframes";
  marker.id = m_markers.size() + 1;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = m_line_width;

  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  geometry_msgs::Point pt_lt, pt_lb, pt_rt, pt_rb, pt_oc, pt_lt0, pt_lt1,
      pt_lt2;

  Eigen2Point(q * (m_scale * imlt) + p, pt_lt);
  Eigen2Point(q * (m_scale * imlb) + p, pt_lb);
  Eigen2Point(q * (m_scale * imrt) + p, pt_rt);
  Eigen2Point(q * (m_scale * imrb) + p, pt_rb);
  Eigen2Point(q * (m_scale * lt0) + p, pt_lt0);
  Eigen2Point(q * (m_scale * lt1) + p, pt_lt1);
  Eigen2Point(q * (m_scale * lt2) + p, pt_lt2);
  Eigen2Point(q * (m_scale * oc) + p, pt_oc);

  // image boundaries
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_lb);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_rb);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_rt);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_lt);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  // top-left indicator
  marker.points.push_back(pt_lt0);
  marker.points.push_back(pt_lt1);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  marker.points.push_back(pt_lt1);
  marker.points.push_back(pt_lt2);
  marker.colors.push_back(m_image_boundary_color);
  marker.colors.push_back(m_image_boundary_color);

  // optical center connector
  marker.points.push_back(pt_lt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  marker.points.push_back(pt_lb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  marker.points.push_back(pt_rt);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  marker.points.push_back(pt_rb);
  marker.points.push_back(pt_oc);
  marker.colors.push_back(m_optical_center_connector_color);
  marker.colors.push_back(m_optical_center_connector_color);

  m_markers.push_back(marker);
}

void CameraPoseVisualizer::reset() { m_markers.clear(); }

void CameraPoseVisualizer::publish(ros::Publisher &pub,
                                   const std_msgs::Header &header) {
  visualization_msgs::MarkerArray markerArray_msg;

  for (auto &marker : m_markers) {
    marker.header = header;
    markerArray_msg.markers.push_back(marker);
  }

  pub.publish(markerArray_msg);
}

} // namespace gmmloc
