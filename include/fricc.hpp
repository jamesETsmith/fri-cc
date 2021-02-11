#ifndef FRICC_HPP
#define FRICC_HPP

// General imports
#include <iostream>

// Eigen Imports
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowTensor2Xd = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using RowTensor4Xd = Eigen::Tensor<double, 4, Eigen::RowMajor>;

#endif