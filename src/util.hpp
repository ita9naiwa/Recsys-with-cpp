#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>

using rowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using colVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;