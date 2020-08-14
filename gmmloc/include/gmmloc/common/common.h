#pragma once

#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <memory>
#include <thread>
#include <mutex>
#include <shared_mutex>

#include <chrono>
#include <algorithm>
#include <numeric>

#include <cstdint>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "eigen_stl_types.h"
#include "eigen_types.h"

#include <g2o/types/slam3d/se3quat.h>

namespace gmmloc {

using g2o::SE3Quat;

using SE3QuatPtr = std::shared_ptr<SE3Quat>;

using SE3QuatConstPtr = std::shared_ptr<const SE3Quat>;
}