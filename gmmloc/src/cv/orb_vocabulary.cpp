
#include "gmmloc/cv/orb_vocabulary.h"

#include "gmmloc/common/common.h"

namespace gmmloc {

using namespace std;

ORBVocabulary::Ptr ORBVocabulary::voc_ = nullptr;

bool ORBVocabulary::initialize(const string &file) {
  if (!voc_) {
    voc_ = unique_ptr<voc_type_t>(new voc_type_t);
  }

  return voc_->loadFromBinaryFile(file);
}

void ORBVocabulary::transform(Frame *frame) {
  CHECK_NOTNULL(voc_);

  vector<cv::Mat> vec_desc;
  for (size_t i = 0; i < frame->num_feats_; i++) {
    vec_desc.push_back(frame->features_[i].desc);
  }

  voc_->transform(vec_desc, frame->bow_vec_, frame->feat_vec_, 4);
}

void ORBVocabulary::transform(KeyFrame *frame) {
  CHECK_NOTNULL(voc_);

  vector<cv::Mat> vec_desc;
  for (size_t i = 0; i < frame->num_feats_; i++) {
    vec_desc.push_back(frame->features_[i].desc);
  }

  voc_->transform(vec_desc, frame->bow_vec_, frame->feat_vec_, 4);
}

} // namespace gmmloc