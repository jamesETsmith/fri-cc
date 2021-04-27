#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <math.h>

#include "doctest.h"

//

#include <fri_utils.hpp>
#include <sparse_tensor.hpp>

//
// Helpers
//

void check_errors(RowTensor4d& result, RowTensor4d& my_result) {
  //
  // Error Checks
  //

  // Check tensor dimensions
  for (int i = 0; i < 4; i++) {
    CHECK(result.dimension(i) == my_result.dimension(i));
  }

  // Check norm of difference between Eigen and our results
  RowTensor4d error = result - my_result;
  auto one_norm = p_norm<1>(error);
  auto two_norm = p_norm<2>(error);

  std::cout << "L-1 Error " << one_norm << std::endl;
  std::cout << "L-2 Error " << two_norm << std::endl;
  std::cout << "L-1 Error per element " << one_norm / result.size()
            << std::endl;
  std::cout << "L-2 Error per element " << two_norm / result.size()
            << std::endl;

  CHECK(one_norm / result.size() < 1e-10);
  CHECK(two_norm / result.size() < 1e-12);
}

//
// Tests
//

TEST_CASE("Sparse Tensor (4,4,10,10)") {
  const std::array<size_t, 4> dims{4, 4, 10, 10};
  // Set up random tensor
  Eigen::Tensor<double, 4, Eigen::RowMajor> tensor(dims[0], dims[1], dims[2],
                                                   dims[3]);
  tensor.setRandom();

  // Create sparse tensor with all the same values
  SparseTensor4d sparse_tensor(dims, tensor.size());
  size_t idx = 0;
  for (size_t i = 0; i < tensor.dimension(0); i++) {
    for (size_t j = 0; j < tensor.dimension(1); j++) {
      for (size_t a = 0; a < tensor.dimension(2); a++) {
        for (size_t b = 0; b < tensor.dimension(3); b++) {
          sparse_tensor.set_element(idx, idx, tensor(i, j, a, b));
          idx += 1;
        }
      }
    }
  }

  for (size_t mi = 0; mi < tensor.size(); mi++) {
    std::array<size_t, 4> idx;
    double value;
    sparse_tensor.get_element(mi, idx, value);

    int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
    CHECK(tensor(i, j, a, b) == value);
  }
}

TEST_CASE("Sparse Tensor (12, 2, 1, 10)") {
  const std::array<size_t, 4> dims{12, 2, 1, 10};

  // Set up random tensor
  RowTensor4d tensor(dims[0], dims[1], dims[2], dims[3]);
  tensor.setRandom();

  // Create sparse tensor with all the same values
  const size_t nnz = tensor.size();
  SparseTensor4d sparse_tensor(dims, nnz);
  size_t idx = 0;
  for (size_t i = 0; i < tensor.dimension(0); i++) {
    for (size_t j = 0; j < tensor.dimension(1); j++) {
      for (size_t a = 0; a < tensor.dimension(2); a++) {
        for (size_t b = 0; b < tensor.dimension(3); b++) {
          sparse_tensor.set_element(idx, idx, tensor(i, j, a, b));
          idx += 1;
        }
      }
    }
  }

  for (size_t mi = 0; mi < tensor.size(); mi++) {
    std::array<size_t, 4> idx;
    double value;
    sparse_tensor.get_element(mi, idx, value);

    int i = idx[0], j = idx[1], a = idx[2], b = idx[3];
    CHECK(tensor(i, j, a, b) == value);
  }
}

TEST_CASE("Sparse Tensor Contraction: 0101") {
  const std::array<size_t, 4> dims{10, 14, 8, 12};

  // Set up random tensor
  RowTensor4d W(dims[0], dims[1], dims[2], dims[3]);
  W.setRandom();

  RowTensor4d T(dims[0], dims[1], dims[2], dims[3]);
  T.setRandom();

  // Zero out half the values of the tensor and put those non-zero values in the
  // sparse tensor
  const size_t nnz = T.size() * 0.5;
  SparseTensor4d T_sparse(dims, nnz);
  size_t idx = 0;
  for (size_t i = 0; i < T.dimension(0); i++) {
    for (size_t j = 0; j < T.dimension(1); j++) {
      for (size_t a = 0; a < T.dimension(2); a++) {
        for (size_t b = 0; b < T.dimension(3); b++) {
          if (idx < nnz) {
            T_sparse.set_element(idx, idx, T(i, j, a, b));

          } else {
            T(i, j, a, b) = 0.0;
          }
          idx += 1;
        }
      }
    }
  }

  // Contraction Helpers
  const Eigen::array<Eigen::IndexPair<int>, 2> contraction_dims_2d{
      Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 1)};
  const Eigen::array<int, 4> shuffle_idx_4d{0, 1, 2, 3};

  // Eigen Contraction
  auto start = std::chrono::steady_clock::now();
  RowTensor4d result = W.contract(T, contraction_dims_2d);
  log_timing("Eigen Contraction Timing", start);

  // Sparse Tensor
  start = std::chrono::steady_clock::now();
  RowTensor4d my_result = contract_SparseTensor4d(
      W, T_sparse, contraction_dims_2d, shuffle_idx_4d, "0101");
  log_timing("Sparse Tensor Timing", start);

  // ERROR CHECKING
  check_errors(result, my_result);
}

TEST_CASE("Sparse Tensor Contraction: 2323") {
  const size_t no = 10;
  const size_t nv = 100;
  const std::array<size_t, 4> dims{no, no, nv, nv};

  // Set up random tensor
  RowTensor4d W(nv, nv, nv, nv);
  W.setRandom();

  RowTensor4d T(dims[0], dims[1], dims[2], dims[3]);
  T.setRandom();

  // Zero out half the values of the tensor and put those non-zero values in the
  // sparse tensor
  const size_t nnz = T.size() * 0.1;
  SparseTensor4d T_sparse(dims, nnz);
  size_t idx = 0;
  for (size_t i = 0; i < T.dimension(0); i++) {
    for (size_t j = 0; j < T.dimension(1); j++) {
      for (size_t a = 0; a < T.dimension(2); a++) {
        for (size_t b = 0; b < T.dimension(3); b++) {
          if (idx < nnz) {
            T_sparse.set_element(idx, idx, T(i, j, a, b));
          } else {
            T(i, j, a, b) = 0.0;
          }
          idx += 1;
        }
      }
    }
  }

  // Contraction Helpers
  const Eigen::array<Eigen::IndexPair<int>, 2> contraction_dims_2d{
      Eigen::IndexPair<int>(2, 2), Eigen::IndexPair<int>(3, 3)};
  const Eigen::array<int, 4> shuffle_idx_4d{2, 3, 0, 1};

  // Eigen Contraction
  auto start = std::chrono::steady_clock::now();
  RowTensor4d result =
      W.contract(T, contraction_dims_2d).shuffle(shuffle_idx_4d);
  log_timing("Eigen Contraction Timing", start);

  // Sparse Tensor
  start = std::chrono::steady_clock::now();
  RowTensor4d my_result = contract_SparseTensor4d(
      W, T_sparse, contraction_dims_2d, shuffle_idx_4d, "2323");
  log_timing("Sparse Tensor Timing", start);

  // ERROR CHECKING
  check_errors(result, my_result);
}