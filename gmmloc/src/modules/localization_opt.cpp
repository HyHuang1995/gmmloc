#include "gmmloc/modules/localization.h"

#include <chrono>
#include <mutex>

#include "gmmloc/config.h"
#include "gmmloc/global.h"

#include "gmmloc/cv/orb_matcher.h"

#include "gmmloc/utils/math_utils.h"
#include "gmmloc/utils/timing.h"

#include "gmmloc/gmm/factors.h"
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

namespace gmmloc {

using namespace std;

GaussianComponent *Localization::optimizeTriangulationVec(Vector3d &x3d,
                                                          KeyFrame *kf_ptr,
                                                          size_t idx1,
                                                          size_t idx2) {
  // if (loc::tri_check_deg) {
  //   if (!curr_kf_->comp_[idx1]->parent_->is_degenerated) {
  //     return false;
  //   }
  // }

  const double &fx1 = curr_kf_->camera_->fx();
  const double &fy1 = curr_kf_->camera_->fy();
  const double &cx1 = curr_kf_->camera_->cx();
  const double &cy1 = curr_kf_->camera_->cy();

  const auto &comps1 = curr_kf_->comps_[idx1];
  const auto &comps2 = kf_ptr->comps_[idx2];

  g2o::SparseOptimizer optimizer;
  g2o::VertexSBAPointXYZ *vertex_pt = new g2o::VertexSBAPointXYZ();
  const Vector3d pt_init = x3d;
  {
    // setup solver
    g2o::OptimizationAlgorithmGaussNewton *solver;
    solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<g2o::BlockSolverX>(
            g2o::make_unique<g2o::LinearSolverEigen<
                g2o::BlockSolverX::LandmarkMatrixType>>()));
    optimizer.setAlgorithm(solver);

    // add vertex
    vertex_pt->setEstimate(pt_init);
    vertex_pt->setId(0);
    optimizer.addVertex(vertex_pt);
  }

  auto addEdgeStereo = [&](const SE3Quat &Tcw, const Feature &kp,
                           const double sigm2_inv) {
    g2o::EdgeProjectXYZOnlyStereo *e = new g2o::EdgeProjectXYZOnlyStereo();
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

    e->setMeasurement(Vector3d(kp.uv.x(), kp.uv.y(), kp.u_right));
    e->setInformation(Eigen::Matrix3d::Identity() * sigm2_inv);

    e->rot_c_w_ = Tcw.rotation();
    e->t_c_w_ = Tcw.translation();
    e->fx = fx1;
    e->fy = fy1;
    e->cx = cx1;
    e->cy = cy1;
    e->bf = curr_kf_->mbf;
    optimizer.addEdge(e);

    return e;
  };
  auto addEdgeMono = [&](const SE3Quat &Tcw, const Feature &kp,
                         const double sigm2_inv) {
    g2o::EdgeProjectXYZOnly *e = new g2o::EdgeProjectXYZOnly();
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));

    e->setMeasurement(kp.uv);
    e->setInformation(Eigen::Matrix2d::Identity() * sigm2_inv);

    e->rot_c_w_ = Tcw.rotation();
    e->t_c_w_ = Tcw.translation();
    e->fx = fx1;
    e->fy = fy1;
    e->cx = cx1;
    e->cy = cy1;
    optimizer.addEdge(e);

    return e;
  };
  auto addEdgeGaussianDeg = [&](GaussianComponent::Ptr comp) {
    g2o::EdgePt2GaussianDeg *e = new g2o::EdgePt2GaussianDeg();
    e->setVertex(
        0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
    e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() *
                      loc::tri_lambda2);

    e->normal_ = comp->axis_.col(0);
    e->mean_ = comp->mean();
    // optimizer.addEdge(e);

    return e;
  };

  // STEP. 2 generate visual factor
  g2o::OptimizableGraph::Edge *edge_kf1, *edge_kf2;
  double th_kf1 = 5.991, th_kf2 = 5.991;
  {
    const auto &kp1 = curr_kf_->features_[idx1];
    const auto &kp2 = kf_ptr->features_[idx2];
    const float &sigma2_inv1 = frame::sigma2_inv[kp1.octave];
    const float &sigma2_inv2 = frame::sigma2_inv[kp2.octave];

    if (kp1.depth > 0.0f) {
      edge_kf1 = addEdgeStereo(curr_kf_->getTcw(), kp1, sigma2_inv1);
      th_kf1 = 7.8;
    } else {
      edge_kf1 = addEdgeMono(curr_kf_->getTcw(), kp1, sigma2_inv1);
    }
    if (kp2.depth > 0.0f) {
      edge_kf2 = addEdgeStereo(kf_ptr->getTcw(), kp2, sigma2_inv1);
      th_kf2 = 7.8;
    } else {
      edge_kf2 = addEdgeMono(kf_ptr->getTcw(), kp2, sigma2_inv1);
    }
  }

  g2o::OptimizableGraph::Edge *edge_str = nullptr;
  GaussianComponent *min_comp = nullptr;
  double min_value = numeric_limits<double>::max();
  Vector3d min_res;

  unordered_set<GaussianComponent *> comps;
  for (auto &&comp : comps1) {
    if (!comps.count(comp->parent_))
      comps.emplace(comp->parent_);
  }
  for (auto &&comp : comps2) {
    if (!comps.count(comp->parent_))
      comps.emplace(comp->parent_);
  }

  for (auto &&g3d : comps) {
    // auto &&g3d = comp->parent_;
    if (!g3d->is_degenerated) {
      continue;
    }

    vertex_pt->setEstimate(pt_init);
    if (edge_str) {
      if (!optimizer.removeEdge(edge_str))
        throw std::runtime_error("remove edge failed");
    }
    edge_str = addEdgeGaussianDeg(g3d);

    optimizer.addEdge(edge_str);

    const int num_iter = 20;
    optimizer.initializeOptimization(0);
    optimizer.optimize(num_iter);

    bool opt_res = true;
    double err_sum;
    {
      if (loc::tri_check_str_chi2 &&
          edge_str->chi2() > loc::tri_str_thresh * loc::tri_lambda2) {
        opt_res = false;
      }

      const double err1 = edge_kf1->chi2();
      const double err2 = edge_kf2->chi2();
      err_sum = err1 + err2;
      if (err1 > th_kf1 || err2 > th_kf2) {
        opt_res = false;
      }
    }

    if (opt_res) {
      if (err_sum < min_value) {
        min_res = vertex_pt->estimate();
        min_comp = g3d;
        min_value = err_sum;
      }
    }
  }

  // TODO: verify likelihood
  if (min_comp) {
    x3d = min_res;
  }

  return min_comp;
}

void Localization::createMapPoints() {
  LOG(INFO) << "create new mappoints";

  // Retrieve neighbor keyframes in covisibility graph
  const int nn = 10;

  const vector<KeyFrame *> neigh_kfs =
      curr_kf_->getBestCovisibilityKeyFrames(nn);

  ORBmatcher matcher(0.6, false);

  const auto Tcw1 = curr_kf_->getTcw();
  const auto Twc1 = curr_kf_->getTwc();
  const Quaterniond &rot_w_c1 = Twc1.rotation();
  Matrix4d Tcw1_mat = Tcw1.to_homogeneous_matrix();

  // rcw_kf0_ = Tcw1.rotation();
  // tcw_kf0_ = Tcw1.translation();

  Vector3d t_w_c1 = Twc1.translation();
  Matrix3d K1 = curr_kf_->camera_->getCameraMatrix();

  const float fx1 = curr_kf_->camera_->fx();
  const float fy1 = curr_kf_->camera_->fy();
  const float cx1 = curr_kf_->camera_->cx();
  const float cy1 = curr_kf_->camera_->cy();
  const float invfx1 = 1.0f / fx1;
  const float invfy1 = 1.0f / fy1;

  const float ratio_factor = 1.5f * frame::scale_factor;

  int nnew = 0;

  LOG(INFO) << "kf neighborhood size: " << neigh_kfs.size();

  // Search matches with epipolar restriction and triangulate
  for (size_t i = 0; i < neigh_kfs.size(); i++) {
    if (i > 0 && checkNewKeyFrames())
      return;

    KeyFrame *kf2_ptr = neigh_kfs[i];

    // Check first that baseline is not too short
    Matrix3d K2 = kf2_ptr->camera_->getCameraMatrix();
    auto Tcw2 = kf2_ptr->getTcw();
    auto Twc2 = kf2_ptr->getTwc();
    Vector3d t_w_c2 = Twc2.translation();
    const Quaterniond &rot_w_c2 = Twc2.rotation();
    Vector3d vec_baseline = t_w_c2 - t_w_c1;

    Matrix4d Tcw2_mat = Tcw2.to_homogeneous_matrix();

    const float baseline = vec_baseline.norm();

    {
      if (baseline < kf2_ptr->mb)
        continue;
    }

    // Search matches that fullfil epipolar constraint
    vector<pair<size_t, size_t>> vMatchedIndices;
    matcher.searchForTriangulation(curr_kf_, kf2_ptr, vMatchedIndices, false);

    Quaterniond rot_c2_w = Tcw2.rotation();
    Vector3d t_c2_w = Tcw2.translation();
    // rcw_kf1_ = rot_c2_w;
    // tcw_kf1_ = t_c2_w;

    const float fx2 = kf2_ptr->camera_->fx();
    const float fy2 = kf2_ptr->camera_->fy();
    const float cx2 = kf2_ptr->camera_->cx();
    const float cy2 = kf2_ptr->camera_->cy();
    const float invfx2 = 1.0f / fx2;
    const float invfy2 = 1.0f / fy2;

    // Triangulate each match
    const int nmatches = vMatchedIndices.size();
    int nnew_frame = 0;
    int nsvd = 0, nproj = 0, ndist = 0, nscale = 0;
    for (int ikp = 0; ikp < nmatches; ikp++) {
      const int &idx1 = vMatchedIndices[ikp].first;
      const int &idx2 = vMatchedIndices[ikp].second;

      const auto &kp1 = curr_kf_->features_[idx1];
      const float kp1_ur = curr_kf_->features_[idx1].u_right;
      bool bStereo1 = kp1_ur >= 0;

      const auto &kp2 = kf2_ptr->features_[idx2];
      const float kp2_ur = kf2_ptr->features_[idx2].u_right;
      bool bStereo2 = kp2_ur >= 0;

      Vector3d xn1((kp1.uv.x() - cx1) * invfx1, (kp1.uv.y() - cy1) * invfy1,
                   1.0);
      Vector3d xn2((kp2.uv.x() - cx2) * invfx2, (kp2.uv.y() - cy2) * invfy2,
                   1.0);
      Vector3d ray1 = rot_w_c1 * xn1;
      Vector3d ray2 = rot_w_c2 * xn2;
      const float cosParallaxRays =
          ray1.dot(ray2) / (ray1.norm() * ray2.norm());

      float cosParallaxStereo = cosParallaxRays + 1;
      float cosParallaxStereo1 = cosParallaxStereo;
      float cosParallaxStereo2 = cosParallaxStereo;

      if (bStereo1)
        cosParallaxStereo1 =
            cos(2 * atan2(curr_kf_->mb / 2, curr_kf_->features_[idx1].depth));
      else if (bStereo2)
        cosParallaxStereo2 =
            cos(2 * atan2(kf2_ptr->mb / 2, kf2_ptr->features_[idx2].depth));

      cosParallaxStereo = min(cosParallaxStereo1, cosParallaxStereo2);

      Vector3d pt3d, ptc;
      bool from_mono = false;
      if (cosParallaxRays < cosParallaxStereo && cosParallaxRays > 0 &&
          (bStereo1 || bStereo2 || cosParallaxRays < 0.9998)) {
        Matrix4d A;
        A.row(0) = xn1(0) * Tcw1_mat.row(2) - Tcw1_mat.row(0);
        A.row(1) = xn1(1) * Tcw1_mat.row(2) - Tcw1_mat.row(1);
        A.row(2) = xn2(0) * Tcw2_mat.row(2) - Tcw2_mat.row(0);
        A.row(3) = xn2(1) * Tcw2_mat.row(2) - Tcw2_mat.row(1);

        Eigen::JacobiSVD<Matrix4d> svd(A, Eigen::ComputeFullU |
                                              Eigen::ComputeFullV);
        Vector4d vt = svd.matrixV().col(3);

        if (vt(4) <= numeric_limits<double>::epsilon()) {
          nsvd++;
          continue;
        }

        // Euclidean coordinates
        pt3d = vt.topRows(3) / vt(4);
        from_mono = true;
      } else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2) {
        curr_kf_->unproject3(idx1, &pt3d);
      } else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1) {
        kf2_ptr->unproject3(idx2, &pt3d);
      } else
        continue; // No stereo and very low parallax

      bool opt_res = false;
      timing::Timer timer_opt_tri("loc/opt_mono");

      GaussianComponent *str_ptr = nullptr;
      str_ptr = optimizeTriangulationVec(pt3d, kf2_ptr, idx1, idx2);
      if (str_ptr) {
        opt_res = true;
      }
      timer_opt_tri.Stop();

      Vector3d uvr1, uvr2;
      if (!curr_kf_->project3(pt3d, &uvr1) || !kf2_ptr->project3(pt3d, &uvr2)) {
        nproj++;
        continue;
      }

      {
        double proj_err = kp1.error(uvr1);
        const float sigma2 = frame::sigma2[kp1.octave];
        const double thresh = bStereo1 ? 7.8 : 5.991;

        if (proj_err > thresh * sigma2) {
          nproj++;
          continue;
        }
      }
      {
        double proj_err = kp2.error(uvr2);
        const float sigma2 = frame::sigma2[kp1.octave];
        const double thresh = bStereo2 ? 7.8 : 5.991;

        if (proj_err > thresh * sigma2) {
          nproj++;
          continue;
        }
      }

      // Check scale consistency
      Vector3d normal1 = pt3d - t_w_c1;
      float dist1 = normal1.norm();
      Vector3d normal2 = pt3d - t_w_c2;
      float dist2 = normal2.norm();

      if (dist1 <= numeric_limits<float>::epsilon() ||
          dist2 <= numeric_limits<float>::epsilon()) {
        ndist++;
        continue;
      }

      const float ratio_dist = dist2 / dist1;
      const float ratio_octave =
          frame::scale_factors[kp1.octave] / frame::scale_factors[kp2.octave];

      if (ratio_dist * ratio_factor < ratio_octave ||
          ratio_dist > ratio_octave * ratio_factor) {
        nscale++;
        continue;
      }

      // Triangulation is succesfull
      MapPoint *mappt = new MapPoint(pt3d, curr_kf_);
      if (from_mono) {
        if (opt_res) {
          mappt->type_ = MapPoint::FromTriMonoGMM;
        } else {
          mappt->type_ = MapPoint::FromTriMono;
        }
      } else {
        if (opt_res) {
          mappt->type_ = MapPoint::FromTriStereoGMM;
        } else {
          mappt->type_ = MapPoint::FromTriStereo;
        }
      }

      if (str_ptr) {
        mappt->asscociations_.push_back(str_ptr);

        if (mappt->asscociations_.size() > 1)
          throw std::runtime_error("mismatch size");
      }

      mappt->addObservation(curr_kf_, idx1);
      mappt->addObservation(kf2_ptr, idx2);

      curr_kf_->addObservation(mappt, idx1);
      kf2_ptr->addObservation(mappt, idx2);

      mappt->computeDistinctiveDescriptors();

      mappt->updateNormalAndDepth();

      map_->addObservation(mappt);

      candidate_mappts_.push_back(mappt);

      nnew++;
      nnew_frame++;
    }

    LOG(INFO) << "#match: " << nmatches << " #svd: " << nsvd
              << " #proj: " << nproj << " #dist: " << ndist
              << " #scale: " << nscale;
  }

  LOG(INFO) << "create new: " << nnew;
}

void Localization::jointOptimization(KeyFrame *kf_ptr, bool *pbStopFlag,
                                     Map *pMap) {
  LOG(INFO) << "enter local bundle adjustment.";

  list<KeyFrame *> local_kfs;

  local_kfs.push_back(kf_ptr);
  kf_ptr->ba_local_kf_ = kf_ptr->idx_;

  const vector<KeyFrame *> neigh_kfs = kf_ptr->getVectorCovisibleKeyFrames();
  for (int i = 0, iend = neigh_kfs.size(); i < iend; i++) {
    KeyFrame *kfi_ptr = neigh_kfs[i];
    kfi_ptr->ba_local_kf_ = kf_ptr->idx_;
    if (!kfi_ptr->not_valid_)
      local_kfs.push_back(kfi_ptr);
  }

  list<MapPoint *> local_mappts;
  for (list<KeyFrame *>::iterator lit = local_kfs.begin(),
                                  lend = local_kfs.end();
       lit != lend; lit++) {
    vector<MapPoint *> vpMPs = (*lit)->getMapPoints();
    for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end();
         vit != vend; vit++) {
      MapPoint *mappt = *vit;
      if (mappt) {
        if (!mappt->not_valid_)
          if (mappt->ba_local_kf_ != kf_ptr->idx_) {
            local_mappts.push_back(mappt);
            mappt->ba_local_kf_ = kf_ptr->idx_;
          }
      }
    }
  }

  list<KeyFrame *> fixed_kfs;
  unordered_map<KeyFrame *, int> fixcam_obs;
  for (list<MapPoint *>::iterator lit = local_mappts.begin(),
                                  lend = local_mappts.end();
       lit != lend; lit++) {
    std::unordered_map<KeyFrame *, size_t> observations =
        (*lit)->getObservations();
    for (std::unordered_map<KeyFrame *, size_t>::iterator
             mit = observations.begin(),
             mend = observations.end();
         mit != mend; mit++) {
      KeyFrame *kfi_ptr = mit->first;

      if (kfi_ptr->ba_local_kf_ != kf_ptr->idx_ &&
          kfi_ptr->fixed_kf_idx_ba != kf_ptr->idx_) {
        kfi_ptr->fixed_kf_idx_ba = kf_ptr->idx_;
        if (!kfi_ptr->not_valid_) {
          fixed_kfs.push_back(kfi_ptr);
          fixcam_obs[kfi_ptr] = 0;
        }
      } else if (kfi_ptr->fixed_kf_idx_ba == kf_ptr->idx_ &&
                 !kfi_ptr->not_valid_) {
        fixcam_obs[kfi_ptr]++;
      }
    }
  }
  KeyFrame *best_obs = nullptr;
  if (!fixcam_obs.empty()) {
    std::vector<std::pair<KeyFrame *, int>> v(fixcam_obs.begin(),
                                              fixcam_obs.end());
    std::sort(v.begin(), v.end(),
              [](const std::pair<KeyFrame *, int> &left,
                 const std::pair<KeyFrame *, int> &right) {
                return left.second > right.second;
              });
    best_obs = v.front().first;
  }

  LOG(INFO) << "TO opt Frame size: " << local_kfs.size() << endl;
  LOG(INFO) << "Fixed Frame size: " << fixed_kfs.size() << endl;

  // Setup optimizer
  g2o::SparseOptimizer optimizer;
  g2o::OptimizationAlgorithmLevenberg *solver;
  solver = new g2o::OptimizationAlgorithmLevenberg(
      g2o::make_unique<g2o::BlockSolver_6_3>(
          g2o::make_unique<
              g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>()));
  optimizer.setAlgorithm(solver);

  if (pbStopFlag)
    optimizer.setForceStopFlag(pbStopFlag);

  unsigned long maxKFid = 0;

  // Set Local KeyFrame vertices
  for (list<KeyFrame *>::iterator lit = local_kfs.begin(),
                                  lend = local_kfs.end();
       lit != lend; lit++) {
    KeyFrame *kfi_ptr = *lit;
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(kfi_ptr->getTcw());
    vSE3->setId(kfi_ptr->idx_);
    optimizer.addVertex(vSE3);
    if (kfi_ptr->idx_ > maxKFid)
      maxKFid = kfi_ptr->idx_;

    if (kfi_ptr->idx_ == 0) {

      if (loc::ba_first_as_prior) {
        g2o::EdgeSE3QuatPrior *e = new g2o::EdgeSE3QuatPrior();

        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                            optimizer.vertex(0)));

        e->setMeasurement(kfi_ptr->getTcw()); // TODO: prior pose

        double sigma_rot = 2.0 * M_PI / 180.0;
        double sigma_rot2_inv = 1.0 / (sigma_rot * sigma_rot);
        double sigma_trans2_inv = 1.0 / (0.01 * 0.01);
        // double sigma_trans2_inv = 1.0;
        Eigen::Matrix<double, 6, 6> info = Eigen::Matrix<double, 6, 6>::Zero();
        info.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * sigma_rot2_inv;
        info.block(3, 3, 3, 3) =
            Eigen::Matrix3d::Identity() * sigma_trans2_inv; // trans
        e->setInformation(info);
        optimizer.addEdge(e);

      } else {
        vSE3->setFixed(kfi_ptr->idx_ == 0);
      }
    }
  }

  bool flag_fixsingle = false;
  if (flag_fixsingle && best_obs) {
    g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(best_obs->getTcw());
    vSE3->setId(best_obs->idx_);
    vSE3->setFixed(true);
    optimizer.addVertex(vSE3);
    if (best_obs->idx_ > maxKFid)
      maxKFid = best_obs->idx_;

  } else {
    for (list<KeyFrame *>::iterator lit = fixed_kfs.begin(),
                                    lend = fixed_kfs.end();
         lit != lend; lit++) {
      KeyFrame *kfi_ptr = *lit;
      g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
      vSE3->setEstimate(kfi_ptr->getTcw());
      vSE3->setId(kfi_ptr->idx_);
      vSE3->setFixed(true);
      optimizer.addVertex(vSE3);
      if (kfi_ptr->idx_ > maxKFid)
        maxKFid = kfi_ptr->idx_;
    }
  }

  // Set MapPoint vertices
  const int nExpectedSize =
      (local_kfs.size() + fixed_kfs.size()) * local_mappts.size();

  vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
  vpEdgesMono.reserve(nExpectedSize);

  vector<KeyFrame *> vpEdgeKFMono;
  vpEdgeKFMono.reserve(nExpectedSize);

  vector<MapPoint *> vpMapPointEdgeMono;
  vpMapPointEdgeMono.reserve(nExpectedSize);

  vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
  vpEdgesStereo.reserve(nExpectedSize);

  vector<KeyFrame *> vpEdgeKFStereo;
  vpEdgeKFStereo.reserve(nExpectedSize);

  vector<MapPoint *> vpMapPointEdgeStereo;
  vpMapPointEdgeStereo.reserve(nExpectedSize);

  vector<g2o::EdgePt2GaussianDeg *> edges_gmm_deg;
  vector<g2o::EdgePt2Gaussian *> vpEdgePt2Gaussian;
  vector<MapPoint *> vpMapPointEdgeGMMDeg, vpMapPointEdgeGMMNondeg;

  const float thHuberMono = sqrt(5.991);
  const float thHuberStereo = sqrt(7.815);

  for (list<MapPoint *>::iterator lit = local_mappts.begin(),
                                  lend = local_mappts.end();
       lit != lend; lit++) {
    MapPoint *mappt = *lit;
    g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
    vPoint->setEstimate(mappt->getPosition());
    int id = mappt->idx_ + maxKFid + 1;
    vPoint->setId(id);
    vPoint->setMarginalized(true);
    optimizer.addVertex(vPoint);

    if (!mappt->asscociations_.empty()) {
      if (mappt->asscociations_.size() > 1) {
        throw std::runtime_error("mismatch size");
      }

      for (size_t i = 0; i < mappt->asscociations_.size(); i++) {
        auto &&comp3d = mappt->asscociations_[i];
        if (comp3d->is_degenerated) {
          g2o::EdgePt2GaussianDeg *e = new g2o::EdgePt2GaussianDeg();

          e->setVertex(0,
                       dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPoint));
          e->setInformation(Eigen::Matrix<double, 1, 1>::Identity() *
                            loc::ba_lambda2);
          e->normal_ = comp3d->axis_.col(0);
          e->mean_ = comp3d->mean();

          optimizer.addEdge(e);
          edges_gmm_deg.push_back(e);
          vpMapPointEdgeGMMDeg.push_back(mappt);
        } else {
          g2o::EdgePt2Gaussian *e = new g2o::EdgePt2Gaussian();

          e->setVertex(0,
                       dynamic_cast<g2o::OptimizableGraph::Vertex *>(vPoint));
          e->setInformation(Eigen::Matrix<double, 3, 3>::Identity());
          e->comp_ = comp3d;

          optimizer.addEdge(e);
          vpEdgePt2Gaussian.push_back(e);
          vpMapPointEdgeGMMNondeg.push_back(mappt);
        }
      }
    }

    const unordered_map<KeyFrame *, size_t> observations =
        mappt->getObservations();

    for (unordered_map<KeyFrame *, size_t>::const_iterator
             mit = observations.begin(),
             mend = observations.end();
         mit != mend; mit++) {
      KeyFrame *kfi_ptr = mit->first;
      if (flag_fixsingle && kfi_ptr->fixed_kf_idx_ba == kf_ptr->idx_ &&
          kfi_ptr != best_obs) {
        continue;
      }

      if (!kfi_ptr->not_valid_) {
        const auto &kpUn = kfi_ptr->features_[mit->second];

        // Monocular observation
        if (kfi_ptr->features_[mit->second].u_right < 0) {
          Eigen::Matrix<double, 2, 1> obs = kpUn.uv.cast<double>();
          // obs << kpUn.uv.cast;

          g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertex(id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertex(kfi_ptr->idx_)));
          e->setMeasurement(obs);
          const float &invSigma2 = frame::sigma2_inv[kpUn.octave];
          e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

          g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuberMono);

          e->fx = kfi_ptr->camera_->fx();
          e->fy = kfi_ptr->camera_->fy();
          e->cx = kfi_ptr->camera_->cx();
          e->cy = kfi_ptr->camera_->cy();

          optimizer.addEdge(e);
          vpEdgesMono.push_back(e);
          vpEdgeKFMono.push_back(kfi_ptr);
          vpMapPointEdgeMono.push_back(mappt);
        } else // Stereo observation
        {
          Eigen::Matrix<double, 3, 1> obs;
          const float kp_ur = kfi_ptr->features_[mit->second].u_right;
          obs << kpUn.uv.cast<double>(), kp_ur;

          g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertex(id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(
                              optimizer.vertex(kfi_ptr->idx_)));
          e->setMeasurement(obs);
          const float &invSigma2 = frame::sigma2_inv[kpUn.octave];
          Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
          e->setInformation(Info);

          g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
          e->setRobustKernel(rk);
          rk->setDelta(thHuberStereo);

          e->fx = kfi_ptr->camera_->fx();
          e->fy = kfi_ptr->camera_->fy();
          e->cx = kfi_ptr->camera_->cx();
          e->cy = kfi_ptr->camera_->cy();
          e->bf = kfi_ptr->mbf;

          optimizer.addEdge(e);
          vpEdgesStereo.push_back(e);
          vpEdgeKFStereo.push_back(kfi_ptr);
          vpMapPointEdgeStereo.push_back(mappt);
        }
      }
    }
  }

  if (pbStopFlag)
    if (*pbStopFlag)
      return;

  optimizer.setVerbose(loc::ba_verbose);
  optimizer.initializeOptimization();
  optimizer.optimize(5);

  for (size_t i = 0, iend = vpMapPointEdgeGMMDeg.size(); i < iend; i++) {
    g2o::EdgePt2GaussianDeg *e = edges_gmm_deg[i];
    MapPoint *mappt = vpMapPointEdgeGMMDeg[i];

    if (mappt->not_valid_)
      continue;

    e->computeError();
    if (e->chi2() > loc::tri_str_thresh * loc::ba_lambda2) {
      e->setLevel(1);
    }

    e->setRobustKernel(0);
  }

  optimizer.initializeOptimization(0);
  optimizer.optimize(5);

  bool bDoMore = true;

  if (pbStopFlag)
    if (*pbStopFlag)
      bDoMore = false;

  if (bDoMore) {

    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
      g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
      MapPoint *mappt = vpMapPointEdgeMono[i];

      if (mappt->not_valid_)
        continue;

      if (e->chi2() > 5.991 || !e->isDepthPositive()) {
        e->setLevel(1);
      }

      e->setRobustKernel(0);
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
      g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
      MapPoint *mappt = vpMapPointEdgeStereo[i];

      if (mappt->not_valid_)
        continue;

      if (e->chi2() > 7.815 || !e->isDepthPositive()) {
        e->setLevel(1);
      }

      e->setRobustKernel(0);
    }

    optimizer.initializeOptimization(0);
    int actual_iter = optimizer.optimize(40);
    LOG(INFO) << "#ba_iter: " << actual_iter;
  }

  vector<pair<KeyFrame *, MapPoint *>> vToErase;
  vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

  // Check inlier observations
  // STEP. check pt2plane
  for (size_t i = 0, iend = vpMapPointEdgeGMMDeg.size(); i < iend; i++) {
    g2o::EdgePt2GaussianDeg *e = edges_gmm_deg[i];
    MapPoint *mappt = vpMapPointEdgeGMMDeg[i];

    if (mappt->not_valid_)
      continue;

    e->computeError();
    if (e->chi2() > loc::tri_str_thresh * loc::ba_lambda2) {
      mappt->asscociations_.clear();
      if (mappt->type_ == MapPoint::FromDepthGMM) {
        mappt->type_ = MapPoint::FromDepth;
      } else if (mappt->type_ == MapPoint::FromTriMonoGMM) {
        mappt->type_ = MapPoint::FromTriMono;
      } else if (mappt->type_ == MapPoint::FromTriStereoGMM) {
        mappt->type_ = MapPoint::FromTriStereo;
      }
    }
  }

  for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++) {
    g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
    MapPoint *mappt = vpMapPointEdgeMono[i];

    if (mappt->not_valid_)
      continue;

    if (e->chi2() > 5.991 || !e->isDepthPositive()) {
      KeyFrame *kfi_ptr = vpEdgeKFMono[i];
      vToErase.push_back(make_pair(kfi_ptr, mappt));
    }
  }

  for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++) {
    g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
    MapPoint *mappt = vpMapPointEdgeStereo[i];

    if (mappt->not_valid_)
      continue;

    if (e->chi2() > 7.815 || !e->isDepthPositive()) {
      KeyFrame *kfi_ptr = vpEdgeKFStereo[i];
      vToErase.push_back(make_pair(kfi_ptr, mappt));
    }
  }

  // Get Map Mutex
  if (!vToErase.empty()) {
    for (size_t i = 0; i < vToErase.size(); i++) {
      KeyFrame *kfi_ptr = vToErase[i].first;
      MapPoint *pMPi = vToErase[i].second;
      kfi_ptr->removeObservation(pMPi);

      if (pMPi->removeObservation(kfi_ptr)) {
        map_->removeMapPoint(pMPi);
      }
    }
  }

  // Recover optimized data
  // Keyframes
  for (list<KeyFrame *>::iterator lit = local_kfs.begin(),
                                  lend = local_kfs.end();
       lit != lend; lit++) {
    KeyFrame *kf_ptr = *lit;
    g2o::VertexSE3Expmap *vSE3 =
        static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(kf_ptr->idx_));
    g2o::SE3Quat pose = vSE3->estimate();
    // if (kf_ptr->idx_ == 0) {
    //   auto Twc = pose.inverse();
    //   cout << "first frame t: " << Twc.translation().transpose()
    //        << " q: " << Twc.rotation().coeffs().transpose() << endl;
    // }
    kf_ptr->setTcw(pose);
  }

  // Points
  for (list<MapPoint *>::iterator lit = local_mappts.begin(),
                                  lend = local_mappts.end();
       lit != lend; lit++) {
    MapPoint *mappt = *lit;
    g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(
        optimizer.vertex(mappt->idx_ + maxKFid + 1));
    mappt->setPosition(vPoint->estimate());
    mappt->updateNormalAndDepth();
  }

  LOG(INFO) << "local BA done.";
}
} // namespace gmmloc