#include "gmmloc/utils/dataloader.h"

#include <fstream>
#include <iostream>

#include <glog/logging.h>

#include <sys/stat.h>

namespace gmmloc {

using namespace std;

bool isPathExist(const std::string &s) {
  struct stat buffer;
  return (stat(s.c_str(), &buffer) == 0);
}

Dataloader::Dataloader(const std::string &str, DataType cfg)
    : cfg_(cfg), base_path_(str) {}

bool Dataloader::getTrajectory(eigen_aligned_std_vector<Quaterniond> &rot,
                               eigen_aligned_std_vector<Vector3d> &trans) {
  rot = rot_;
  trans = trans_;

  return true;
}

DataloaderEuRoC::DataloaderEuRoC(const std::string &str,
                                 const std::string &traj_file, DataType cfg)
    : Dataloader(str, cfg) {

  LOG(WARNING) << "loading data from: " << base_path_;
  if (!isPathExist(base_path_))
    throw std::runtime_error("base path not exists: " + base_path_);

  if (cfg_ & DataType::GT) {
    if (!isPathExist(traj_file))
      throw std::runtime_error("traj_file not exists: " + traj_file);

    loadTrajectory(traj_file);
    LOG(WARNING) << "load trajectory: " << traj_file;
    num_ = rot_.size();
  }

  if (cfg_ & DataType::Mono) {
    loadImages(base_path_);
    num_ = mono_file_.size();
  }
}

DataFrame::Ptr DataloaderEuRoC::getNextFrame() {
  if (idx_ >= num_)
    return nullptr;

  auto ptr = getFrameByIndex(idx_);

  idx_++;
  return ptr;
}

DataFrame::Ptr DataloaderEuRoC::getFrameByIndex(size_t idx) {
  if (idx >= num_)
    return nullptr;

  auto ptr = DataFrame::Ptr(new DataFrame());

  if (cfg_ & DataType::GT) {
    ptr->rot_w_c = rot_[idx];
    ptr->t_w_c = trans_[idx];
  }

  if (cfg_ & DataType::Mono) {
    if (isPathExist(mono_file_[idx])) {
      ptr->mono = cv::imread(mono_file_[idx]);
    } else {
      ptr->mono = cv::Mat();
    }

    ptr->timestamp = time_stamp_[idx];
  }
  if (cfg_ & DataType::Depth) {
    if (isPathExist(depth_file_[idx])) {
      ptr->depth = cv::imread(depth_file_[idx]);
    } else {
      ptr->depth = cv::Mat();
    }
  }

  ptr->idx = idx;
  return ptr;
}

void DataloaderEuRoC::loadImages(const string &base_path) {
  ifstream fTimes;
  string strPathTimeFile = base_path + "/cam0/data.csv";
  string strPrefixLeft = base_path + "/cam0/data/";
  string strPrefixRight = base_path + "/cam1/data/";

  fTimes.open(strPathTimeFile.c_str());
  string s;
  getline(fTimes, s);
  while (!fTimes.eof()) {
    string s;
    getline(fTimes, s);
    if (!s.empty()) {
      int index = s.find_first_of(",");
      string t = s.substr(0, index);

      time_stamp_.push_back(stod(t) / 10.0e8);
      mono_file_.push_back(strPrefixLeft + t + ".png");
      depth_file_.push_back(strPrefixRight + t + ".png");
    }
  }
}

void DataloaderEuRoC::loadTrajectory(const std::string &traj_file) {
  ifstream fs_mean(traj_file.c_str());
  string str_line;
  while (getline(fs_mean, str_line) && !fs_mean.eof()) {
    stringstream ss(str_line);
    double time, x, y, z, qx, qy, qz, qw;
    ss >> time;
    ss >> x;
    ss >> y;
    ss >> z;
    ss >> qx;
    ss >> qy;
    ss >> qz;
    ss >> qw;

    Eigen::Vector3d trans_tmp(x, y, z);
    Eigen::Quaterniond q_tmp(qw, qx, qy, qz);
    trans_.push_back(trans_tmp);
    rot_.push_back(q_tmp);
  }
}

} // namespace gmmloc
