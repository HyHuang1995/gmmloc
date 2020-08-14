#include "gmmloc/gmm/gmm_utils.h"

#include "gmmloc/utils/protobuf_utils.h"

#include "GMM.pb.h"

namespace gmmloc {

bool GMMUtility::loadGMMModel(const std::string &file_path, GMM::Ptr &model) {
  //   CHECK_NOTNULL(model);

  GaussianComponents comps;

  CHECK(!file_path.empty());
  std::fstream infile;
  infile.open(file_path, std::fstream::in);
  //   std::ios_base::openmode file_flags = std::fstream::out |
  //   std::fstream::binary;
  if (!infile.is_open()) {
    LOG(ERROR) << "Could not open protobuf file to load layer: " << file_path;
    return false;
  }

  uint32_t tmp_byte_offset = 0;

  uint32_t num_comps;
  if (!pb::readProtoMsgCountFromStream(&infile, &num_comps, &tmp_byte_offset)) {
    LOG(ERROR) << "failed read number of messages.";
    return false;
  }

  if (num_comps == 0u) {
    LOG(WARNING) << "protobuf file empty!";
    return false;
  }

  for (uint32_t i = 0; i < num_comps; i++) {

    // FIXME: eof not working?
    // if (infile.eof()) {
    //   LOG(ERROR) << "incomplete map file..";
    // }

    ComponentProto comp_proto;
    if (!pb::readProtoMsgFromStream(&infile, &comp_proto, &tmp_byte_offset)) {
      LOG(ERROR) << "failed to read component message.";
      return false;
    }

    CHECK_EQ(comp_proto.mean_size(), 3);
    CHECK_EQ(comp_proto.covariance_size(), 9);
    Vector3d mean;
    Matrix3d cov;
    mean << comp_proto.mean(0), comp_proto.mean(1), comp_proto.mean(2);
    cov << comp_proto.covariance(0), comp_proto.covariance(1),
        comp_proto.covariance(2), comp_proto.covariance(3),
        comp_proto.covariance(4), comp_proto.covariance(5),
        comp_proto.covariance(6), comp_proto.covariance(7),
        comp_proto.covariance(8);

    comps.push_back(new GaussianComponent(mean, cov));
  }

  model = GMM::Ptr(new GMM(comps));

  return true;
}

bool GMMUtility::saveGMMModel(const std::string &file_path, GMM::Ptr &model,
                              bool clear_file) {
  CHECK_NOTNULL(model);

  GMMProto proto_gmm;
  CHECK(!file_path.empty());
  std::fstream outfile;
  std::ios_base::openmode file_flags = std::fstream::out | std::fstream::binary;

  outfile.open(file_path, file_flags);
  if (!outfile.is_open()) {
    LOG(ERROR) << "fail to save model, path: " << file_path;
    return false;
  }

  size_t num_to_write = model->countComponents();
  const auto &comps = model->getComponents();

  const uint32_t num_messages = num_to_write;
  if (!pb::writeProtoMsgCountToStream(num_messages, &outfile)) {
    LOG(ERROR) << "fail to write message number to file.";
    outfile.close();
    return false;
  }

  for (auto &comp : comps) {
    ComponentProto proto_comp;

    proto_comp.set_is_degenerated(comp->is_degenerated);

    proto_comp.set_is_salient(comp->is_salient);

    const Vector3d &mean = comp->mean();
    const Matrix3d &cov = comp->cov();
    for (size_t i = 0; i < 3; i++) {
      proto_comp.add_mean(mean(i));
    }
    for (size_t i = 0; i < 9; i++) {
      proto_comp.add_covariance(cov(i));
    }

    if (!pb::writeProtoMsgToStream(proto_comp, &outfile)) {
      LOG(ERROR) << "failed to write component message.";
      outfile.close();
      return false;
    }
  }

  outfile.close();
  return true;
}

GaussianComponent2d::Ptr
GMMUtility::projectGaussian(const GaussianComponent &g, PinholeCamera::Ptr cam,
                            const Eigen::Quaterniond &rot_c_w,
                            const Eigen::Vector3d &t_c_w) {
  auto &&cov3d = g.cov();
  auto &&mu3d = g.mean();

  Eigen::Vector3d mu3d_cam = rot_c_w * mu3d + t_c_w;

  Eigen::Vector2d mu2d;
  Eigen::Matrix<double, 2, 3> jacob_proj;
  auto res_proj = cam->project3(mu3d_cam, &mu2d, &jacob_proj);

  if (res_proj.getDetailedStatus() !=
      ProjectionResult::Status::KEYPOINT_VISIBLE) {
    // cout << "not visible" << endl;
    return nullptr;
  }

  auto rot = rot_c_w.toRotationMatrix();

  auto cov2d =
      jacob_proj * rot * cov3d * rot.transpose() * jacob_proj.transpose();

  return GaussianComponent2d::Ptr(new GaussianComponent2d(mu2d, cov2d));
}

} // namespace gmmloc
