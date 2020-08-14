#include "gmmloc/modules/tracking.h"

#include <mutex>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <Eigen/StdVector>

#include "gmmloc/config.h"

namespace gmmloc {

using namespace std;

int Tracking::optimizeCurrentPose() {

  g2o::SparseOptimizer optimizer;

  g2o::OptimizationAlgorithmLevenberg *solver;

  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  int num_init_correspondences = 0;

  // Set Frame vertex
  g2o::VertexSE3Expmap *vertex_se3 = new g2o::VertexSE3Expmap();
  vertex_se3->setEstimate(curr_frame_->getTcw());
  vertex_se3->setId(0);
  vertex_se3->setFixed(false);
  optimizer.addVertex(vertex_se3);

  // Set MapPoint vertices
  const int N = curr_frame_->num_feats_;

  // for Monocular
  vector<g2o::EdgeSE3ProjectXYZOnlyPose *> edges_mono;
  vector<size_t> indices_edge_mono;
  edges_mono.reserve(N);
  indices_edge_mono.reserve(N);

  // for Stereo
  vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> edges_stereo;
  vector<size_t> indices_edge_stereo;
  edges_stereo.reserve(N);
  indices_edge_stereo.reserve(N);

  const float delta_mono = sqrt(5.991);
  const float delta_stereo = sqrt(7.815);

  {
    unique_lock<mutex> lock(MapPoint::global_mutex_);

    for (int i = 0; i < N; i++) {
      MapPoint *mappt = curr_frame_->mappoints_[i];
      if (mappt) {
        // Monocular observation
        if (curr_frame_->features_[i].u_right < 0) {
          num_init_correspondences++;
          curr_frame_->is_outlier_[i] = false;

          const auto &kp = curr_frame_->features_[i];
          Vector2d obs = kp.uv;
          // obs << kp.uv;

          g2o::EdgeSE3ProjectXYZOnlyPose *e =
              new g2o::EdgeSE3ProjectXYZOnlyPose();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertex(0)));
          e->setMeasurement(obs);
          const float inv_sigma2 = frame::sigma2_inv[kp.octave];
          e->setInformation(Eigen::Matrix2d::Identity() * inv_sigma2);

          g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(delta_mono);

          e->fx = curr_frame_->camera_->fx();
          e->fy = curr_frame_->camera_->fy();
          e->cx = curr_frame_->camera_->cx();
          e->cy = curr_frame_->camera_->cy();

          e->Xw = mappt->getPosition();

          optimizer.addEdge(e);

          edges_mono.push_back(e);
          indices_edge_mono.push_back(i);
        } else {
          num_init_correspondences++;
          curr_frame_->is_outlier_[i] = false;

          // SET EDGE
          Eigen::Matrix<double, 3, 1> obs;
          const auto &kp = curr_frame_->features_[i];
          const float &kp_ur = curr_frame_->features_[i].u_right;
          obs << kp.uv, kp_ur;

          g2o::EdgeStereoSE3ProjectXYZOnlyPose *e =
              new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertex(0)));
          e->setMeasurement(obs);
          const float inv_sigma2 = frame::sigma2_inv[kp.octave];
          Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * inv_sigma2;
          e->setInformation(Info);

          g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(delta_stereo);

          e->fx = curr_frame_->camera_->fx();
          e->fy = curr_frame_->camera_->fy();
          e->cx = curr_frame_->camera_->cx();
          e->cy = curr_frame_->camera_->cy();
          e->bf = curr_frame_->mbf;
          e->Xw = mappt->getPosition();

          optimizer.addEdge(e);

          edges_stereo.push_back(e);
          indices_edge_stereo.push_back(i);
        }
      }
    }
  }

  if (num_init_correspondences < 3)
    return 0;

  // We perform 4 optimizations, after each optimization we classify observation
  // as inlier/outlier At the next optimization, outliers are not included, but
  // at the end they can be classified as inliers again.
  const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
  const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
  const int its[4] = {10, 10, 10, 10};

  int num_bad = 0;
  for (size_t it = 0; it < 4; it++) {

    vertex_se3->setEstimate(curr_frame_->getTcw());
    optimizer.initializeOptimization(0);
    optimizer.optimize(its[it]);

    num_bad = 0;
    for (size_t i = 0, iend = edges_mono.size(); i < iend; i++) {
      g2o::EdgeSE3ProjectXYZOnlyPose *e = edges_mono[i];

      const size_t idx = indices_edge_mono[i];

      if (curr_frame_->is_outlier_[idx]) {
        e->computeError();
      }

      const float chi2 = e->chi2();

      if (chi2 > chi2Mono[it]) {
        curr_frame_->is_outlier_[idx] = true;
        e->setLevel(1); // outlier
        num_bad++;
      } else {
        curr_frame_->is_outlier_[idx] = false;
        e->setLevel(0); // inlier
      }

      if (it == 2)
        e->setRobustKernel(0);
    }

    for (size_t i = 0, iend = edges_stereo.size(); i < iend; i++) {
      g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = edges_stereo[i];

      const size_t idx = indices_edge_stereo[i];

      if (curr_frame_->is_outlier_[idx]) {
        e->computeError();
      }

      const float chi2 = e->chi2();

      if (chi2 > chi2Stereo[it]) {
        curr_frame_->is_outlier_[idx] = true;
        e->setLevel(1);
        num_bad++;
      } else {
        e->setLevel(0);
        curr_frame_->is_outlier_[idx] = false;
      }

      if (it == 2)
        e->setRobustKernel(0);
    }

    if (optimizer.edges().size() < 10)
      break;
  }

  // Recover optimized pose and return number of inliers
  g2o::VertexSE3Expmap *vertex_se3_recov =
      static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
  g2o::SE3Quat pose = vertex_se3_recov->estimate();
  // cv::Mat pose = Converter::toCvMat(SE3quat_recov);
  curr_frame_->setTcw(pose);

  return num_init_correspondences - num_bad;
}

} // namespace gmmloc