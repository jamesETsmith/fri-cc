#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <fri_utils.hpp>
#include <fricc.hpp>
#include <span>

namespace py = pybind11;

// Tensor utilities
// template <int p>
// double p_norm(RowTensor4d& error) {
//   double norm;
// #pragma omp parallel for collapse(4) reduction(+ : norm)
//   for (size_t i = 0; i < error.dimension(0); i++) {
//     for (size_t j = 0; j < error.dimension(1); j++) {
//       for (size_t a = 0; a < error.dimension(2); a++) {
//         for (size_t b = 0; b < error.dimension(3); b++) {
//           norm += pow(abs(error(i, j, a, b)), p);
//         }
//       }
//     }
//   }
//   return pow(norm, 1. / p);
// }

class SparseTensor4d {
  //
  const size_t nnz;
  const std::array<size_t, 4> dims;
  std::vector<std::array<size_t, 4>> indices;
  std::vector<double> data;
  // std::shared_ptr<SparseTensor4d> this_shared_ptr;

  //
  size_t flat_idx(const int i, const int j, const int a, const int b);
  std::array<size_t, 4> unpack_idx(const size_t idx);

 public:
  // Constructors
  SparseTensor4d(std::array<size_t, 4> dims, const size_t nnz)
      : dims(dims), nnz(nnz) {
    indices.resize(nnz);
    data.resize(nnz);
  }

  SparseTensor4d(py::array_t<double> tensor_arr, std::array<size_t, 4> dims,
                 const size_t m, const std::string compression = "fri",
                 const std::string sampling_method = "pivotal",
                 const bool verbose = false)
      : dims(dims), nnz(m) {
    //
    auto _total = std::chrono::steady_clock::now();
    indices.resize(m);
    data.resize(m);

    std::span tensor_flat(tensor_arr.data(),
                          static_cast<size_t>(tensor_arr.size()));

    // Input Checking
    if (!compression.compare("largest")) {
      // Sort dense tensor
      auto t_largest_idx = argsort(tensor_flat, m);

// Put them into sparse tensor (in parallel)
#pragma omp parallel for schedule(static, 16)
      for (size_t i = 0; i < m; i++) {
        const size_t idx = t_largest_idx[i];
        this->set_element(i, idx, tensor_flat[idx]);
      }

      // Fast randomized iteration (FRI) compression
    } else if (!compression.compare("fri")) {
      std::vector<size_t> compressed_idx;
      std::vector<double> compressed_vals;
      auto _t_compress = std::chrono::steady_clock::now();
      std::tie(compressed_idx, compressed_vals) =
          fri_compression(tensor_flat, m, sampling_method, verbose);
      auto t_compress = get_timing(_t_compress);
      std::cout << "t compress " << t_compress << std::endl;

      auto _t_set = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static, 16)
      for (size_t i = 0; i < m; i++) {
        this->set_element(i, compressed_idx[i], compressed_vals[i]);
      }
      auto t_set = get_timing(_t_set);
      std::cout << t_set << std::endl;

      // Bad compression method
    } else {
      std::cerr << "ERROR";
      std::cerr << "\tThe compression method you chose (" << compression
                << ") isn't supported ";
      std::cerr << "compression must be 'largest' or 'fri'" << std::endl;
      exit(EXIT_FAILURE);
    }

    auto t_total = get_timing(_total);
    std::cout << "Total time for init of SparseTensor4d " << t_total
              << std::endl;
  }

  // Getters/Setters
  void set_element(const size_t mi, const size_t idx, const double value);
  void get_element(const size_t mi, std::array<size_t, 4>& idx, double& value);
  const size_t dimension(const int i) { return dims[i]; }
  std::array<size_t, 4> dimensions() { return dims; }
  const size_t size() { return nnz; }

  // Contraction

  // IO Helpers
  friend std::ostream& operator<<(std::ostream& os,
                                  const SparseTensor4d& tensor) {
    os << "i   j   k   l        val" << std::endl;
    for (int i = 0; i < tensor.data.size(); i++) {
      os << tensor.indices[i][0] << "   " << tensor.indices[i][1] << "   ";
      os << tensor.indices[i][2] << "   " << tensor.indices[i][3] << "        ";
      os << tensor.data[i] << std::endl;
    }
    return os;
  }
  void print() { std::cout << *this << std::endl; }
};

void contract_SparseTensor4d_wrapper(py::array_t<double> W, SparseTensor4d& T,
                                     py::array_t<double> output,
                                     const std::string term);
#endif