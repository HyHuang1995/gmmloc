#include "gmmloc/gmmloc.h"

#include "gmmloc/cv/orb_vocabulary.h"

#include "gmmloc/utils/timing.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "gmmloc/gmm/factors.h"

namespace gmmloc {

KeyFrame *GMMLoc::processKeyFrame(Frame *frame, bool is_first) {
  LOG(INFO) << "process new keyframe";

  frame->is_keyframe_ = true;
  KeyFrame *kf_ptr = new KeyFrame(*frame);

  ORBVocabulary::transform(kf_ptr);

  frame->ref_keyframe_ = kf_ptr;

  associateMapElements(kf_ptr);

  createMapPointsFromStereo(frame, kf_ptr, !is_first);

  return kf_ptr;
}

void GMMLoc::createMapPointsFromStereo(Frame *frame, KeyFrame *kf_ptr,
                                          bool check_depth) {
  std::vector<pair<float, size_t>> depth_indices;
  depth_indices.reserve(frame->num_feats_);
  for (size_t i = 0; i < frame->num_feats_; i++) {
    float z = frame->features_[i].depth;
    if (z > 0) {
      depth_indices.push_back(make_pair(z, i));
    }
  }

  // SE3Quat Tcw = frame->getTcw();
  if (!depth_indices.empty()) {
    std::sort(depth_indices.begin(), depth_indices.end());

    int num_points = 0;
    for (size_t j = 0; j < depth_indices.size(); j++) {
      size_t i = depth_indices[j].second;

      bool create_new = false;

      MapPoint *mappt = frame->mappoints_[i];
      if (!mappt)
        create_new = true;
      else if (mappt->countObservations() < 1) {
        create_new = true;
        frame->mappoints_[i] = nullptr;
      }

      if (create_new) {
        const auto &feat = frame->features_[i];
        const auto &comps = kf_ptr->comps_[i];

        if (feat.depth > 0.0f) {
          Vector3d pt3d;
          // frame->camera_->unproject3(feat.uv, feat.depth, &ptc);
          // Vector3d pt3d = frame->getTwc().map(ptc);
          frame->unproject3(i, &pt3d);

          GaussianComponent::ConstPtr str_ptr = nullptr;
          if (!comps.empty()) {
            str_ptr = checkMapAssociation(pt3d, kf_ptr, i);

            if (!str_ptr)
              continue;
          }

          MapPoint *mappt = new MapPoint(pt3d, kf_ptr);

          if (str_ptr) {
            mappt->asscociations_.push_back(str_ptr);
            mappt->type_ = MapPoint::FromDepthGMM;
          } else {
            mappt->type_ = MapPoint::FromDepth;
          }

          mappt->addObservation(kf_ptr, i);
          mappt->computeDistinctiveDescriptors();
          mappt->updateNormalAndDepth();

          kf_ptr->addObservation(mappt, i);

          map_->addObservation(mappt);

          frame->mappoints_[i] = mappt;
        }
        num_points++;
      } else {
        num_points++;
      }

      LOG(INFO) << "k";
      if (check_depth && depth_indices[j].first > frame::th_depth &&
          num_points > 100)
        break;
    }
  }
}

void GMMLoc::associateMapElements(KeyFrame *kf) {
  CHECK_NOTNULL(gmm_model_);

  auto Tcw = kf->getTcw();
  Quaterniond rot_c_w = Tcw.rotation();
  Vector3d t_c_w = Tcw.translation();

  {
    timing::Timer t("loc/render_view");
    gmm_model_->renderView(rot_c_w, t_c_w);
    t.Stop();
  }

  LOG(INFO) << "searching correspondences";
  {
    timing::Timer t("map/search_corr");
    vector<GaussianComponents2d> vcomps;
    gmm_model_->searchCorrespondence(kf->features_, vcomps);
    kf->comps_ = vcomps;
    t.Stop();
  }

  // cv::Mat viz_img(480, 752, CV_8UC3, cv::Scalar(255, 255, 255));
  // gmm_model_->visualize2d(viz_img);

  // viz_img.copyTo(viz_img_);

  // for (size_t i = 0; i < comps.size(); i++) {
  //   if (!comps[i])
  //     continue;
  //   const auto &kp = curr_kf_->mvKeysUn[i].pt;
  //   const auto &mean = comps[i]->mean();
  //   cv::line(viz_img, kp, cv::Point(mean.x(), mean.y()), cv::Scalar(0, 255,
  //   0));
  // }
  // cv::imshow("win", viz_img);
  // cv::waitKey(-1);

  // LOG(INFO) << "done.";
}

GaussianComponent::ConstPtr
GMMLoc::checkMapAssociation(Vector3d &pt3d, KeyFrame *kf, size_t idx) {
  const auto &comps = kf->comps_[idx];
  const auto &kp = curr_frame_->features_[idx];
  const SE3Quat Tcw = curr_frame_->getTcw();

  if (comps.empty()) { // no associations
    return nullptr;
  }

  Vector3d uvr(kp.uv.x(), kp.uv.y(), kp.u_right);
  const Eigen::Vector3d pt_init = pt3d;

  const Eigen::Vector3d ptc = Tcw.map(pt3d);
  double proj_z = ptc.z();
  proj_z = proj_z > 1.0 ? 1.0 : proj_z;
  const double proj_z2 = proj_z * proj_z;
  // const double proj_z2_inv = 1.0 / proj_z2;

  int min_idx = -1;
  double min_value = numeric_limits<double>::max();
  Eigen::Vector3d min_res;

  for (size_t i = 0; i < comps.size(); i++) {
    auto &&gc2d = comps[i];
    if (!gc2d)
      continue;
    // STEP. check degeneration
    // if (loc::tri_check_deg) {
    //   // if (!curr_kf_->comp_[idx]->parent_->is_degenerated) {
    //   // continue;
    //   // }
    // }

    auto opt_res = optimizePoint(pt_init, kp, Tcw, gc2d->parent_, proj_z2);

    if (opt_res.res && opt_res.chi2_proj < min_value) {
      min_idx = i;
      min_value = opt_res.chi2_proj;
      min_res = opt_res.pt_est;
    }
  }

  bool use_nn3d = true;
  if (min_idx != -1) {

    auto g3d_ptr = comps[min_idx]->parent_;
    double ll = g3d_ptr->chi2(min_res);

    // bool local_extrema = true;
    auto str_ptr = g3d_ptr;

    CHECK_NOTNULL(g3d_ptr);
    for (auto &&np : g3d_ptr->nbs_) {
      CHECK_NOTNULL(np.ptr);
      double ln = np.ptr->chi2(min_res);
      if (ln < ll) {
        // local_extrema = false;
        ll = ln;
        str_ptr = np.ptr;
      }
    }
    // StrOptStat opt_res;
    if (str_ptr != g3d_ptr) {
      auto opt_res = optimizePoint(pt_init, kp, Tcw, str_ptr, proj_z2);

      if (opt_res.res) {
        min_res = opt_res.pt_est;
      } else {
        str_ptr = g3d_ptr;
        ll = g3d_ptr->chi2(min_res);
      }
    }

    if (ll > 9.0) {
      return nullptr;
    }

    pt3d = min_res;
    return str_ptr;

  } else if (use_nn3d) {

    vector<int> res;
    gmm_model_->queryPoint(pt_init, res);
    auto gc = gmm_model_->getComponent3d(res[0]); // TODO: better strategy
    if (!gc->is_degenerated)
      return nullptr;

    auto opt_res = optimizePoint(pt_init, kp, Tcw, gc, proj_z2);

    if (opt_res.res) {
      pt3d = opt_res.pt_est;

    } else {

      return nullptr;
    }
  }
  // else {
  return nullptr;
  // }
}

StrOptStat GMMLoc::optimizePoint(const Vector3d &pt3d, const Feature &kp,
                                    const SE3Quat &Tcw,
                                    GaussianComponent::ConstPtr comp,
                                    const double proj_z2) {

  const float &kp_ur = kp.u_right;
  const float &sigma2_inv = frame::sigma2_inv[kp.octave];

  g2o::SparseOptimizer optimizer;

  g2o::OptimizationAlgorithmGaussNewton *solver;
  solver = new g2o::OptimizationAlgorithmGaussNewton(
      g2o::make_unique<g2o::BlockSolverX>(
          g2o::make_unique<g2o::LinearSolverEigen<
              g2o::BlockSolverX::LandmarkMatrixType>>()));
  optimizer.setAlgorithm(solver);

  g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
  v->setEstimate(pt3d);
  v->setId(0);
  optimizer.addVertex(v);

  g2o::EdgeProjectXYZOnlyStereo *factor_proj;
  g2o::EdgePt2GaussianDeg *factor_str;

  // Stereo observation
  {
    g2o::EdgeProjectXYZOnlyStereo *e = new g2o::EdgeProjectXYZOnlyStereo();

    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

    e->setMeasurement(Vector3d(kp.uv.x(), kp.uv.y(), kp_ur));
    e->setInformation(Eigen::Matrix3d::Identity() * sigma2_inv);

    e->rot_c_w_ = Tcw.rotation();
    e->t_c_w_ = Tcw.translation();
    e->fx = camera::fx;
    e->fy = camera::fy;
    e->cx = camera::cx;
    e->cy = camera::cy;
    e->bf = camera::bf;
    optimizer.addEdge(e);
    factor_proj = e;
  }

  // structure observation
  {
    g2o::EdgePt2GaussianDeg *e = new g2o::EdgePt2GaussianDeg();

    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(v));
    e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() *
                      loc::tri_lambda2 * proj_z2);

    e->normal_ = comp->axis_.col(0);
    e->mean_ = comp->mean();
    optimizer.addEdge(e);

    factor_str = e;
  }

  // Optimize
  StrOptStat opt_res;
  {
    int num_iter = 5;
    optimizer.initializeOptimization();
    optimizer.optimize(num_iter);
    opt_res.res = true;
    opt_res.chi2_proj = factor_proj->chi2();
    opt_res.chi2_str = factor_str->chi2();
    opt_res.pt_est = v->estimate();

    const double thresh = 7.815;
    if (opt_res.chi2_proj > thresh) {
      opt_res.res = false;
    }
    if (loc::tri_check_str_chi2 &&
        opt_res.chi2_str > loc::tri_str_thresh * loc::tri_lambda2) {
      opt_res.res = false;
    }
  }
  return opt_res;
}

} // namespace gmmloc
