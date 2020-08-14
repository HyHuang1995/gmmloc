#include "gmmloc/utils/cv_utils.h"

#include <opencv2/opencv.hpp>

namespace gmmloc {

using namespace std;

Rectify::Rectify(const string &config) {
  cv::FileStorage fsSettings(config, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    cerr << "ERROR: Wrong path to settings" << endl;
    throw;
  }

  cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
  fsSettings["LEFT.K"] >> K_l;
  fsSettings["RIGHT.K"] >> K_r;

  fsSettings["LEFT.P"] >> P_l;
  fsSettings["RIGHT.P"] >> P_r;

  fsSettings["LEFT.R"] >> R_l;
  fsSettings["RIGHT.R"] >> R_r;

  fsSettings["LEFT.D"] >> D_l;
  fsSettings["RIGHT.D"] >> D_r;

  int rows_l = fsSettings["LEFT.height"];
  int cols_l = fsSettings["LEFT.width"];
  int rows_r = fsSettings["RIGHT.height"];
  int cols_r = fsSettings["RIGHT.width"];

  if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() ||
      R_r.empty() || D_l.empty() || D_r.empty() || rows_l == 0 || rows_r == 0 ||
      cols_l == 0 || cols_r == 0) {
    cerr << "ERROR: Calibration parameters to rectify stereo are missing!"
         << endl;
    throw;
  }

  cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3),
                              cv::Size(cols_l, rows_l), CV_32F, M1l, M2l);
  cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3),
                              cv::Size(cols_r, rows_r), CV_32F, M1r, M2r);
}

void Rectify::doRectifyL(const cv::Mat &img_src, cv::Mat &img_rec) {
  cv::remap(img_src, img_rec, M1l, M2l, cv::INTER_LINEAR);
}

void Rectify::doRectifyR(const cv::Mat &img_src, cv::Mat &img_rec) {
  cv::remap(img_src, img_rec, M1r, M2r, cv::INTER_LINEAR);
}

} // namespace gmmloc
