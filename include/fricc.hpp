#ifndef FRICC_HPP
#define FRICC_HPP

// General imports
#include <iostream>
#include <chrono>

// Eigen Imports
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowTensor2d = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using RowTensor4d = Eigen::Tensor<double, 4, Eigen::RowMajor>;
using TMap2d = Eigen::TensorMap<RowTensor2d>;
using TMap4d = Eigen::TensorMap<RowTensor4d>;

#endif