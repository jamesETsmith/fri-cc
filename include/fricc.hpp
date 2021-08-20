#ifndef FRICC_HPP
#define FRICC_HPP

// General imports
#include <math.h>
#include <stdlib.h>
// #include <omp.h>
// #include <stdio.h>

#include <chrono>
#include <iostream>
#include <string>

// Eigen Imports
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

// Type short-hands
using VecXST = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
using RowMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowTensor2d = Eigen::Tensor<double, 2, Eigen::RowMajor>;
using RowTensor4d = Eigen::Tensor<double, 4, Eigen::RowMajor>;
using TMap2d = Eigen::TensorMap<RowTensor2d>;
using TMap4d = Eigen::TensorMap<RowTensor4d>;

// Timing utilities
/**
 * @brief Return the elapsed time given a starging time.
 *
 * @tparam Clock
 * @param start
 * @return double
 */
template <typename Clock>
double get_timing(std::chrono::time_point<Clock> start) {
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;  // in seconds
  return elapsed.count();
}

/**
 * @brief Given a message and a start time, print the message and the time
 * elapsed since the start time.
 *
 * @tparam Clock
 * @param msg Message to print to stdout.
 * @param start Starting time.
 */
template <typename Clock>
double log_timing(std::string msg, std::chrono::time_point<Clock> start) {
  double time_elapsed = get_timing(start);
  printf("%-24s %6.4f (s)\n", msg.c_str(), time_elapsed);
  return time_elapsed;
}

#endif