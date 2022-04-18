#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <array>
#include <fri_utils.hpp>
#include <fricc.hpp>
#include <span>

namespace py = pybind11;

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
      std::tie(compressed_idx, compressed_vals) =
          fri_compression(tensor_flat, m, sampling_method, verbose);

#pragma omp parallel for schedule(static, 16)
      for (size_t i = 0; i < m; i++) {
        this->set_element(i, compressed_idx[i], compressed_vals[i]);
      }

      // Bad compression method
    } else {
      std::cerr << "ERROR";
      std::cerr << "\tThe compression method you chose (" << compression
                << ") isn't supported ";
      std::cerr << "compression must be 'largest' or 'fri'" << std::endl;
      exit(EXIT_FAILURE);
    }
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

inline size_t SparseTensor4d::flat_idx(const int i, const int j, const int a,
                                       const int b) {
  return i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + a * dims[3] +
         b;
}

inline std::array<size_t, 4> SparseTensor4d::unpack_idx(const size_t idx) {
  size_t i0 = idx / (dims[1] * dims[2] * dims[3]);
  size_t i1 = idx % (dims[1] * dims[2] * dims[3]) / (dims[2] * dims[3]);
  size_t i2 = idx % (dims[2] * dims[3]) / dims[3];
  size_t i3 = idx % dims[3];
  std::array<size_t, 4> idx_arr = {i0, i1, i2, i3};

  // TODO take out for performance
  for (int i = 0; i < 4; i++) {
    if (idx_arr[i] >= dims[i]) {
      std::cerr << "ERROR: " << idx_arr[i] << " >= " << dims[i]
                << " for dimension " << i << " !!!" << std::endl;
      std::cerr << "Indices are out of range for given tensor" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return idx_arr;
}

inline void SparseTensor4d::set_element(const size_t mi, const size_t idx,
                                        const double value) {
  if (mi > nnz) {
    std::cerr << "Index you requested is larger than the specified size"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  indices[mi] = unpack_idx(idx);
  data[mi] = value;
}

inline void SparseTensor4d::get_element(const size_t mi,
                                        std::array<size_t, 4>& idx_arr,
                                        double& value) {
  idx_arr = indices[mi];
  value = data[mi];
}

void contract_SparseTensor4d_wrapper(py::array_t<double> W, SparseTensor4d& T,
                                     py::array_t<double> output,
                                     const std::string term);

void contract_SparseTensor4d_wrapper_experimental(py::array_t<double> W,
                                                  SparseTensor4d& T,
                                                  py::array_t<double> output,
                                                  const std::string term);
#endif