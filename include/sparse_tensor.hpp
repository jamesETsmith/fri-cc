#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP
#include <array>
#include <fri_utils.hpp>
#include <fricc.hpp>

using VecXST = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

// Tensor utilities
template <int p>
double p_norm(RowTensor4d& error) {
  double norm;
#pragma omp parallel reduction(+ : norm)
  for (size_t i = 0; i < error.dimension(0); i++) {
    for (size_t j = 0; j < error.dimension(1); j++) {
      for (size_t a = 0; a < error.dimension(2); a++) {
        for (size_t b = 0; b < error.dimension(3); b++) {
          norm += pow(abs(error(i, j, a, b)), p);
        }
      }
    }
  }
  return pow(norm, 1. / p);
}

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
  SparseTensor4d(const Eigen::Ref<Eigen::VectorXd>& tensor_flat,
                 std::array<size_t, 4> dims, const size_t m)
      : dims(dims), nnz(m) {
    indices.resize(m);
    data.resize(m);

    // Sort dense tensor
    auto t_largest_idx = partial_argsort_paired(tensor_flat, m);

// Put them into sparse tensor (in parallel)
#pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < m; i++) {
      const size_t idx = t_largest_idx[i];
      this->set_element(i, idx, tensor_flat(idx));
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

void contract_SparseTensor4d(RowTensor4d& W, SparseTensor4d& T,
                             RowTensor4d& output, const std::string term);

void contract_SparseTensor4d_2323(RowTensor4d& W, SparseTensor4d& T,
                                  RowTensor4d& output);
void contract_SparseTensor4d_2323_wrapper(TMap4d& W, SparseTensor4d& T,
                                          TMap4d& output);

void contract_SparseTensor4d_wrapper(Eigen::Ref<Eigen::VectorXd>& W_vec,
                                     SparseTensor4d& T,
                                     Eigen::Ref<Eigen::VectorXd>& output_vec,
                                     const std::string term);
#endif