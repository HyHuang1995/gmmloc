#pragma once

#include <opencv2/core.hpp>

namespace gmmloc {

class Rectify {
public:
  Rectify(const std::string &config);

  void doRectifyL(const cv::Mat &img_src, cv::Mat &img_rec);

  void doRectifyR(const cv::Mat &img_src, cv::Mat &img_rec);

  cv::Mat M1l, M2l, M1r, M2r;
};

} // namespace gmmloc
