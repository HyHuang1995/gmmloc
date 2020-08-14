#pragma once

#include <glog/logging.h>

#include <ros/ros.h>

#include "config.h"

namespace gmmloc {

void initParameters(ros::NodeHandle &nh) {

#define GPARAM(x, y)                                                           \
  do {                                                                         \
    if (!nh.getParam(x, y)) {                                                  \
      LOG(WARNING) << "retrive pararm " #x " error!";                          \
    }                                                                          \
  } while (0)

  GPARAM("gt_path", common::gt_path);
  GPARAM("vocabulary_path", common::voc_path);
  GPARAM("data_path", common::data_path);
  GPARAM("output_path", common::output_path);

  GPARAM("rect_config", common::rect_config);

  GPARAM("gmm_path", common::gmm_path);
  GPARAM("online", common::online);
  GPARAM("verbose", common::verbose);
  GPARAM("viewer", common::viewer);

  GPARAM("map/neighbor_dist_thresh", gmmmap::neighbor_dist_thresh);

  GPARAM("camera/bf", camera::bf);
  GPARAM("camera/fx", camera::fx);
  GPARAM("camera/fy", camera::fy);
  GPARAM("camera/cx", camera::cx);
  GPARAM("camera/cy", camera::cy);
  GPARAM("camera/fps", camera::fps);
  GPARAM("camera/width", camera::width);
  GPARAM("camera/height", camera::height);
  GPARAM("camera/do_rectify", camera::do_rectify);
  GPARAM("camera/do_equalization", camera::do_equalization);

  // processing
  {
    camera::fx_inv = 1.0f / camera::fx;
    camera::fy_inv = 1.0f / camera::fy;

    frame::num_grid_col_inv =
        static_cast<float>(frame::grid_cols) / camera::width;

    frame::num_grid_row_inv =
        static_cast<float>(frame::grid_rows) / camera::height;
  }

  // frame
  GPARAM("frame/th_depth", frame::th_depth);
  GPARAM("frame/num_features", frame::num_features);
  {
    frame::th_depth = camera::bf * frame::th_depth / camera::fx;

    frame::scale_factors.resize(frame::num_levels);
    frame::scale_factors_inv.resize(frame::num_levels);
    frame::sigma2.resize(frame::num_levels);
    frame::sigma2_inv.resize(frame::num_levels);
    frame::scale_factors[0] = 1.0f;
    frame::sigma2[0] = 1.0f;
    frame::scale_factors_inv[0] = 1.0f;
    frame::sigma2_inv[0] = 1.0f;
    for (int i = 1; i < frame::num_levels; i++) {
      frame::scale_factors[i] =
          frame::scale_factors[i - 1] * frame::scale_factor;
      frame::sigma2[i] = frame::scale_factors[i] * frame::scale_factors[i];

      frame::scale_factors_inv[i] = 1.0f / frame::scale_factors[i];
      frame::sigma2_inv[i] = 1.0f / frame::sigma2[i];
    }
  }

  GPARAM("loc/tri_check_deg", loc::tri_check_deg);
  GPARAM("loc/tri_use_stereo", loc::tri_use_stereo);
  GPARAM("loc/tri_lambda2", loc::tri_lambda2);
  GPARAM("loc/tri_check_str_chi2", loc::tri_check_str_chi2);
  GPARAM("loc/tri_str_thresh", loc::tri_str_thresh);

  GPARAM("loc/ba_lambda2", loc::ba_lambda2);
  GPARAM("loc/ba_verbose", loc::ba_verbose);

  GPARAM("loc/ba_first_as_prior", loc::ba_first_as_prior);

#undef GPARAM
}

void printParameters() {}

} // namespace gmmloc
