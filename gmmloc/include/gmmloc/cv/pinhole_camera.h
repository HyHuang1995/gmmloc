// adapted from aslam_cv2

#pragma once

#include <iostream>
#include <memory>

#include <Eigen/Dense>

namespace gmmloc {

/// \struct ProjectionResult
/// \brief This struct is returned by the camera projection methods and holds
/// the result state
///        of the projection operation.
struct ProjectionResult {
  /// Possible projection state.
  enum class Status {
    /// Keypoint is visible and projection was successful.
    KEYPOINT_VISIBLE,
    /// Keypoint is NOT visible but projection was successful.
    KEYPOINT_OUTSIDE_IMAGE_BOX,
    /// The projected point lies behind the camera plane.
    POINT_BEHIND_CAMERA,
    /// The projection was unsuccessful.
    PROJECTION_INVALID,
    /// Default value after construction.
    UNINITIALIZED
  };

  // Make the enum values accessible from the outside without the additional
  // indirection.
  static Status KEYPOINT_VISIBLE;
  static Status KEYPOINT_OUTSIDE_IMAGE_BOX;
  static Status POINT_BEHIND_CAMERA;
  static Status PROJECTION_INVALID;
  static Status UNINITIALIZED;

  constexpr ProjectionResult() : status_(Status::UNINITIALIZED){};
  constexpr ProjectionResult(Status status) : status_(status){};

  /// \brief ProjectionResult can be typecasted to bool and is true if the
  /// projected keypoint
  ///        is visible. Simplifies the check for a successful projection.
  ///        Example usage:
  /// @code
  ///          aslam::ProjectionResult ret =
  ///          camera_->project3(Eigen::Vector3d(0, 0, -10), &keypoint); if(ret)
  ///          std::cout << "Projection was successful!\n";
  /// @endcode
  explicit operator bool() const { return isKeypointVisible(); };

  /// \brief Compare objects.
  bool operator==(const ProjectionResult &other) const {
    return status_ == other.status_;
  };

  /// \brief Compare projection status.
  bool operator==(const ProjectionResult::Status &other) const {
    return status_ == other;
  };

  /// \brief Convenience function to print the state using streams.
  friend std::ostream &operator<<(std::ostream &out,
                                  const ProjectionResult &state);

  /// \brief Check whether the projection was successful and the point is
  /// visible in the image.
  bool isKeypointVisible() const {
    return (status_ == Status::KEYPOINT_VISIBLE);
  };

  /// \brief Returns the exact state of the projection operation.
  ///        Example usage:
  /// @code
  ///          aslam::ProjectionResult ret =
  ///          camera_->project3(Eigen::Vector3d(0, 0, -1), &keypoint);
  ///          if(ret.getDetailedStatus() ==
  ///          aslam::ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX)
  ///            std::cout << "Point behind camera! Lets do something...\n";
  /// @endcode
  Status getDetailedStatus() const { return status_; };

private:
  /// Stores the projection state.
  Status status_;
};

/// \class PinholeCamera
/// \brief An implementation of the pinhole camera model with (optional)
/// distortion.
///
/// The usual model of a pinhole camera follows these steps:
///    - Transformation: Transform the point into a coordinate frame associated
///    with the camera
///    - Normalization:  Project the point onto the normalized image plane:
///    \f$\mathbf y := \left[ x/z,y/z\right] \f$
///    - Distortion:     apply a nonlinear transformation to \f$y\f$ to account
///    for radial and tangential distortion of the lens
///    - Projection:     Project the point into the image using a standard \f$3
///    \time 3\f$ projection matrix
///
///  Intrinsic parameters ordering: fu, fv, cu, cv
///  Reference: http://en.wikipedia.org/wiki/Pinhole_camera_model
class PinholeCamera {

  enum { kNumOfParams = 4 };

public:
  // using Ptr = std::shared_ptr<PinholeCamera>;

  // using ConstPtr = std::shared_ptr<const PinholeCamera>;

  using Ptr = PinholeCamera *;

  using ConstPtr = const PinholeCamera *;

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum Parameters { kFu = 0, kFv = 1, kCu = 2, kCv = 3 };

private:
  Eigen::Vector4d intrinsics_;
  uint32_t width_, height_;

  // TODO(slynen) Enable commented out PropertyTree support
  // PinholeCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

public:
  /// \brief Empty constructor for serialization interface.
  PinholeCamera() = default;

  /// Copy constructor for clone operation.
  PinholeCamera(const PinholeCamera &other) = default;

  void operator=(const PinholeCamera &) = delete;

public:
  PinholeCamera(const Eigen::Vector4d &intrinsics, uint32_t image_width,
                uint32_t image_height);

  PinholeCamera(double focallength_cols, double focallength_rows,
                double imagecenter_cols, double imagecenter_rows,
                uint32_t image_width, uint32_t image_height);

  virtual ~PinholeCamera() = default;

  /// \brief Convenience function to print the state using streams.
  //   friend std::ostream &operator<<(std::ostream &out,
  //                                   const PinholeCamera &camera);

  //   virtual bool backProject3(const Eigen::Ref<const Eigen::Vector2d>
  //   &keypoint,
  //                             Eigen::Vector3d *out_point_3d) const;

  bool isKeypointVisible(const Eigen::Vector2d &keypoint) const;

  inline const ProjectionResult
  evaluateProjectionResult(const Eigen::Vector2d &keypoint,
                           const Eigen::Vector3d &point_3d) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Functional methods to project and back-project points
  /// @{

  // Get the overloaded non-virtual project3Functional(..) from base into
  // scope.
  //   using Camera::project3Functional;
  const ProjectionResult project3(const Eigen::Vector3d &point_3d,
                                  Eigen::Vector2d *out_keypoint) const;

  void unproject3(const Eigen::Vector2d &uv, const double& z,
                  Eigen::Vector3d *pt3d) const;

  void unproject3(const Eigen::Quaterniond &rot_w_c,
                  const Eigen::Vector3d &t_w_c, const Eigen::Vector2d &uv,
                  double z, Eigen::Vector3d *pt3d) const;

  const ProjectionResult project3(
      const Eigen::Vector3d &point_3d, Eigen::Vector2d *out_keypoint,
      Eigen::Matrix<double, 2, 3> *out_jacobian_point,
      Eigen::Matrix<double, 2, 4> *out_jacobian_intrinsics = nullptr) const;

  /// \brief Get a set of border rays
  //   TODO:(hhy)
  void getBorderRays(Eigen::MatrixXd &rays) const;

  /// @}

public:
  //////////////////////////////////////////////////////////////
  /// \name Methods to access intrinsics.
  /// @{

  /// \brief Returns the camera matrix for the pinhole projection.
  inline Eigen::Matrix3d getCameraMatrix() const {
    Eigen::Matrix3d K;
    K << fx(), 0.0, cx(), 0.0, fy(), cy(), 0.0, 0.0, 1.0;
    return K;
  }

  /// \brief The horizontal focal length in pixels.
  double fx() const { return intrinsics_[Parameters::kFu]; };
  /// \brief The vertical focal length in pixels.
  double fy() const { return intrinsics_[Parameters::kFv]; };
  /// \brief The horizontal image center in pixels.
  double cx() const { return intrinsics_[Parameters::kCu]; };
  /// \brief The vertical image center in pixels.
  double cy() const { return intrinsics_[Parameters::kCv]; };

  /// Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  //   virtual void printParameters(std::ostream &out,
  //                                const std::string &text) const;

private:
  /// \brief Minimal depth for a valid projection.
  static const double kMinimumDepth;

  //   bool isValidImpl() const override;
  //   void setRandomImpl() override;
  //   bool isEqualImpl(const Sensor &other, const bool verbose) const
  //   override;
}; // namespace gmmloc

} // namespace gmmloc
