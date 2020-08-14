#pragma once

#include <cstdint>
#include <memory>

#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include "../common/eigen_stl_types.h"
#include "../common/eigen_types.h"

namespace gmmloc {

enum class DataType : std::uint8_t {
  Mono = 1,
  Stereo = 1 << 1,
  Depth = 1 << 2,
  IMU = 1 << 3,
  Lidar = 1 << 4,
  Odom = 1 << 5,

  GT = 1 << 7

};

inline DataType operator|(DataType a, DataType b) {
  return static_cast<DataType>(static_cast<uint8_t>(a) |
                               static_cast<uint8_t>(b));
}

inline uint8_t operator&(DataType a, DataType b) {
  return static_cast<uint8_t>(a) & static_cast<uint8_t>(b);
}

struct DataFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  uint32_t idx = 0;

  using Ptr = std::shared_ptr<DataFrame>;

  using ConstPtr = std::shared_ptr<const DataFrame>;

  cv::Mat mono, depth;
  double timestamp;

  Vector3d t_w_c;
  Quaterniond rot_w_c;
};

class Dataloader {
public:
  using Ptr = std::unique_ptr<Dataloader>;

  using ConstPtr = std::unique_ptr<const Dataloader>;

  Dataloader() = default;

  Dataloader(const std::string &str, DataType cfg);

  virtual ~Dataloader() = default;

  virtual DataFrame::Ptr getNextFrame() = 0;

  virtual DataFrame::Ptr getFrameByIndex(size_t idx) = 0;

  virtual bool getTrajectory(eigen_aligned_std_vector<Quaterniond> &rot,
                             eigen_aligned_std_vector<Vector3d> &trans);

  size_t getSize() const { return num_; }

protected:
  size_t num_ = 0, idx_ = 0;
  DataType cfg_;

  std::string base_path_;

  std::vector<std::string> mono_file_, depth_file_;
  std::vector<double> time_stamp_;

  eigen_aligned_std_vector<Quaterniond> rot_;
  eigen_aligned_std_vector<Vector3d> trans_;
};

class DataloaderEuRoC : public Dataloader {
public:
  using Ptr = std::shared_ptr<DataloaderEuRoC>;

  using ConstPtr = std::shared_ptr<const DataloaderEuRoC>;

  DataloaderEuRoC(const std::string &str, const std::string &gt_path,
                  DataType cfg);

  ~DataloaderEuRoC() = default;

  DataFrame::Ptr getNextFrame() override;

  DataFrame::Ptr getFrameByIndex(size_t idx) override;

private:
  void loadTrajectory(const std::string &traj_file);

  void loadImages(const std::string &ass_file);
};

} // namespace gmmloc