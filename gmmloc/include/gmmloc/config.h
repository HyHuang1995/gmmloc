#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace gmmloc {

namespace common {

extern std::string voc_path;

extern std::string data_path;

extern std::string output_path;

extern std::string gt_path;

extern std::string rect_config;

extern std::string gmm_path;

extern bool online;

extern bool verbose;

extern bool viewer;

} // namespace common

namespace gmmmap {

extern double neighbor_dist_thresh;

} // namespace gmmmap

namespace camera {
extern float fx, fy, cx, cy;

extern float fx_inv, fy_inv;

extern float k1, k2, p1, p2, k3;

extern int width, height;

extern float fps;

extern float bf;

extern bool do_rectify;

extern bool do_equalization;
} // namespace camera

namespace frame {

const int grid_cols = 64, grid_rows = 48;

extern float num_grid_col_inv, num_grid_row_inv;

extern const int num_levels;
extern const float scale_factor, scale_factor_log;

extern std::vector<float> scale_factors;
extern std::vector<float> scale_factors_inv;
extern std::vector<float> sigma2;
extern std::vector<float> sigma2_inv;

extern int num_features;

extern float th_depth;

} // namespace frame

namespace loc {

extern bool tri_check_deg;

extern bool tri_use_stereo;

extern float tri_lambda2;

extern bool tri_check_str_chi2;

extern float tri_str_thresh;

extern bool ba_verbose;

extern float ba_lambda2;

extern bool ba_first_as_prior;

extern double ba_prior_sigma_trans;

extern double ba_prior_sigma_rot;

} // namespace loc

namespace viewer {

extern int traj_length;

} // namespace viewer

} // namespace gmmloc
