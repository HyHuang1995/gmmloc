#pragma once

#include <Eigen/Dense>

namespace gmmloc {
using scalar_t = double;

/// The empty column vector (zero rows, one column), templated on scalar type.
template <typename Scalar> using Vector0 = Eigen::Matrix<Scalar, 0, 1>;

/// A column vector of size 1 (that is, a scalar), templated on scalar type.
template <typename Scalar> using Vector1 = Eigen::Matrix<Scalar, 1, 1>;
using Vector1d = Eigen::Matrix<double, 1, 1>;
using Vector1f = Eigen::Matrix<float, 1, 1>;
using Vector1i = Eigen::Matrix<int, 1, 1>;
using Vector1s = Eigen::Matrix<scalar_t, 1, 1>;

/// A column vector of size 2, templated on scalar type.
template <typename Scalar> using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
// using Vector2d = Eigen::Vector<double, >
using Eigen::Vector2d;
using Eigen::Vector2f;
using Eigen::Vector2i;
using Vector2s = Eigen::Matrix<scalar_t, 2, 1>;

/// A column vector of size 3, templated on scalar type.
template <typename Scalar> using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
using Eigen::Vector3d;
using Eigen::Vector3f;
using Eigen::Vector3i;
using Vector3s = Eigen::Matrix<scalar_t, 3, 1>;

/// A column vector of size 4, templated on scalar type.
template <typename Scalar> using Vector4 = Eigen::Matrix<Scalar, 4, 1>;
using Eigen::Vector4d;
using Eigen::Vector4f;
using Eigen::Vector4i;
using Vector4s = Eigen::Matrix<scalar_t, 4, 1>;

/// A column vector of size 6.
template <typename Scalar> using Vector5 = Eigen::Matrix<Scalar, 5, 1>;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector5f = Eigen::Matrix<float, 5, 1>;
using Vector5i = Eigen::Matrix<int, 5, 1>;
using Vector5s = Eigen::Matrix<scalar_t, 5, 1>;

/// A column vector of size 6.
template <typename Scalar> using Vector6 = Eigen::Matrix<Scalar, 6, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector6i = Eigen::Matrix<int, 6, 1>;
using Vector6s = Eigen::Matrix<scalar_t, 6, 1>;

/// A column vector templated on the number of rows.
template <typename Scalar, int Rows>
using Vector = Eigen::Matrix<Scalar, Rows, 1>;

/// A column vector of any size, templated on scalar type.
template <typename Scalar>
using VectorX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/// A vector of dynamic size templated on scalar type, up to a maximum of 6
/// elements.
template <typename Scalar>
using VectorUpTo6 = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, 0, 6, 1>;

/// A row vector of size 2, templated on scalar type.
template <typename Scalar> using RowVector2 = Eigen::Matrix<Scalar, 1, 2>;
using Eigen::RowVector2d;
using Eigen::RowVector2f;
using Eigen::RowVector2i;
using RowVector2s = Eigen::Matrix<scalar_t, 1, 2>;

/// A row vector of size 3, templated on scalar type.
template <typename Scalar> using RowVector3 = Eigen::Matrix<Scalar, 1, 3>;
using Eigen::RowVector3d;
using Eigen::RowVector3f;
using Eigen::RowVector3i;
using RowVector3s = Eigen::Matrix<scalar_t, 1, 3>;

/// A row vector of size 4, templated on scalar type.
template <typename Scalar> using RowVector4 = Eigen::Matrix<Scalar, 1, 4>;
using Eigen::RowVector4d;
using Eigen::RowVector4f;
using Eigen::RowVector4i;
using RowVector4s = Eigen::Matrix<scalar_t, 1, 4>;

/// A row vector of size 6.
template <typename Scalar> using RowVector6 = Eigen::Matrix<Scalar, 1, 6>;
using RowVector6d = Eigen::Matrix<double, 1, 6>;
using RowVector6f = Eigen::Matrix<float, 1, 6>;
using RowVector6i = Eigen::Matrix<int, 1, 6>;
using RowVector6s = Eigen::Matrix<scalar_t, 1, 6>;

/// A row vector templated on the number of columns.
template <typename Scalar, int Cols>
using RowVector = Eigen::Matrix<Scalar, 1, Cols>;

/// A row vector of any size, templated on scalar type.
template <typename Scalar>
using RowVectorX = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

/// A matrix of 2 rows and 2 columns, templated on scalar type.
template <typename Scalar> using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
using Eigen::Matrix2d;
using Eigen::Matrix2f;
using Eigen::Matrix2i;
using Matrix2s = Eigen::Matrix<scalar_t, 2, 2>;

/// A matrix of 3 rows and 3 columns, templated on scalar type.
template <typename Scalar> using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
using Eigen::Matrix3d;
using Eigen::Matrix3f;
using Eigen::Matrix3i;
using Matrix3s = Eigen::Matrix<scalar_t, 3, 3>;

/// A matrix of 4 rows and 4 columns, templated on scalar type.
template <typename Scalar> using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
using Eigen::Matrix4d;
using Eigen::Matrix4f;
using Eigen::Matrix4i;
using Matrix4s = Eigen::Matrix<scalar_t, 4, 4>;

/// A matrix of 6 rows and 6 columns, templated on scalar type.
template <typename Scalar> using Matrix6 = Eigen::Matrix<Scalar, 6, 6>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix6f = Eigen::Matrix<float, 6, 6>;
using Matrix6i = Eigen::Matrix<int, 6, 6>;
using Matrix6s = Eigen::Matrix<scalar_t, 6, 6>;

/// A matrix of 2 rows, dynamic columns, templated on scalar type.
template <typename Scalar>
using Matrix2X = Eigen::Matrix<Scalar, 2, Eigen::Dynamic>;

/// A matrix of 3 rows, dynamic columns, templated on scalar type.
template <typename Scalar>
using Matrix3X = Eigen::Matrix<Scalar, 3, Eigen::Dynamic>;

/// A matrix of 4 rows, dynamic columns, templated on scalar type.
template <typename Scalar>
using Matrix4X = Eigen::Matrix<Scalar, 4, Eigen::Dynamic>;

/// A matrix of 6 rows, dynamic columns, templated on scalar type.
template <typename Scalar>
using Matrix6X = Eigen::Matrix<Scalar, 6, Eigen::Dynamic>;

/// A matrix of dynamic size, templated on scalar type.
template <typename Scalar>
using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

/// A matrix of dynamic size templated on scalar type, up to a maximum of 6 rows
/// and 6 columns. Rectangular matrices, with different number of rows and
/// columns, are allowed.
template <typename Scalar>
using MatrixUpTo6 =
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6>;

/// A quaternion templated on scalar type.
template <typename Scalar> using Quaternion = Eigen::Quaternion<Scalar>;
using Eigen::Quaterniond;
using Eigen::Quaternionf;
using Quaternions = Eigen::Quaternion<scalar_t>;

/// An AngleAxis templated on scalar type.
template <typename Scalar> using AngleAxis = Eigen::AngleAxis<Scalar>;
using Eigen::AngleAxisd;
using Eigen::AngleAxisf;
using AngleAxiss = Eigen::AngleAxis<scalar_t>;

/// An Isometry templated on scalar type.
template <typename Scalar>
using Isometry3 = Eigen::Transform<Scalar, 3, Eigen::Isometry>;
using Eigen::Isometry3d;
using Eigen::Isometry3f;
using Isometry3s = Eigen::Transform<scalar_t, 3, Eigen::Isometry>;

/// A translation in 3D templated on scalar type.
template <typename Scalar> using Translation3 = Eigen::Translation<Scalar, 3>;

} // namespace gmmloc
