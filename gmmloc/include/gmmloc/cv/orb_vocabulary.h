#pragma once

#include <memory>

#include "orb_dbow2/dbow2/FORB.h"
#include "orb_dbow2/dbow2/TemplatedVocabulary.h"

#include "../types/frame.h"
#include "../types/keyframe.h"

namespace gmmloc {

class ORBVocabulary {
private:
  using voc_type_t =
      DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;

  static std::unique_ptr<voc_type_t> voc_;

public:
  using Ptr = std::unique_ptr<voc_type_t>;

  ORBVocabulary() = delete;

  ORBVocabulary(const ORBVocabulary &voc) = delete;

  static bool initialize(const std::string &voc_file);

  static void transform(Frame *frame);

  static void transform(KeyFrame *frame);
};

} // namespace gmmloc
