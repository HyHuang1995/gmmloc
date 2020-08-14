#pragma once

#include "../common/common.h"

namespace gmmloc {

class MathUtils {
public:
  static Matrix3d skew(const Vector3d &vec);

  static Matrix3d computeEssentialMatrix(const SE3Quat &Tcw1,
                                         const SE3Quat &Tcw2

  );

  static Matrix3d computeFundamentalMatrix(const SE3Quat &Tcw1,
                                           const Matrix3d &K1,
                                           const SE3Quat &Tcw2,
                                           const Matrix3d &K2

  );
};

} // namespace gmmloc
