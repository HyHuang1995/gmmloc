#include "gmmloc/config.h"

#include <limits>

namespace gmmloc {

namespace common {

std::string voc_path;

std::string data_path;

std::string gt_path;

std::string output_path;

std::string rect_config;

std::string gmm_path;

bool online = false;

bool verbose = false;

bool viewer = false;

} // namespace common

namespace camera {

float fx, fy, cx, cy;

float fx_inv, fy_inv;

float k1 = 0.0f, k2 = 0.0f, p1 = 0.0f, p2 = 0.0f, k3 = 0.0f;

int width = 0, height = 0;

float fps;

float bf;

bool do_rectify = false;

bool do_equalization = false;

} // namespace camera

namespace frame {

float num_grid_col_inv, num_grid_row_inv;

const int num_levels = 8;

const float scale_factor = 1.2;

const float scale_factor_log = log(scale_factor);

std::vector<float> scale_factors;
std::vector<float> scale_factors_inv;
std::vector<float> sigma2;
std::vector<float> sigma2_inv;

float th_depth = 35.0f;

int num_features = 1000;

} // namespace frame

namespace gmmmap {

double neighbor_dist_thresh = 0.5;

} // namespace gmmmap

namespace loc {

bool tri_use_stereo = false;

bool tri_check_deg = false;

float tri_lambda2 = 100.0f;

bool tri_check_str_chi2 = false;

float tri_str_thresh = 0.1;

bool ba_verbose = false;

float ba_lambda2 = 100.0f;

bool ba_first_as_prior = false;

} // namespace loc

namespace viewer {

int traj_length = 200;

} // namespace viewer

} // namespace gmmloc
