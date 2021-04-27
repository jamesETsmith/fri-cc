#include <array>
#include <fricc.hpp>
#include <string>

class SparseTensor4d {
  //
  size_t nnz;
  const std::array<size_t, 4> dims;
  std::vector<std::array<size_t, 4>> indices;
  std::vector<double> data;

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

  // Getters/Setters
  void set_element(const size_t mi, const size_t idx, const double value);
  void get_element(const size_t mi, std::array<size_t, 4>& idx, double& value);
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
};

RowTensor4d contract_SparseTensor4d(
    RowTensor4d& W, SparseTensor4d& T,
    const Eigen::array<Eigen::IndexPair<int>, 2>& contraction_dims_2d,
    const Eigen::array<int, 4>& shuffle_idx_4d, const std::string term);