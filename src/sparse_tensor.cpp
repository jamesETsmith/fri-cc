#include <sparse_tensor.hpp>

inline size_t SparseTensor4d::flat_idx(const int i, const int j, const int a,
                                       const int b) {
  return i * dims[1] * dims[2] * dims[3] + j * dims[2] * dims[3] + a * dims[3] +
         b;
}

std::array<size_t, 4> SparseTensor4d::unpack_idx(const size_t idx) {
  size_t i0 = idx / (dims[1] * dims[2] * dims[3]);
  size_t i1 = idx % (dims[1] * dims[2] * dims[3]) / (dims[2] * dims[3]);
  size_t i2 = idx % (dims[2] * dims[3]) / dims[3];
  size_t i3 = idx % dims[3];
  std::array<size_t, 4> idx_arr = {i0, i1, i2, i3};

  // TODO take out for performance
  for (int i = 0; i < 4; i++) {
    if (idx_arr[i] >= dims[i]) {
      std::cout << "ERROR: " << idx_arr[i] << " >= " << dims[i]
                << " for dimension " << i << " !!!" << std::endl;
      throw "Indices are out of range for given tensor";
    }
  }
  return idx_arr;
}

void SparseTensor4d::set_element(const size_t mi, const size_t idx,
                                 const double value) {
  indices[mi] = unpack_idx(idx);
  data[mi] = value;
}

void SparseTensor4d::get_element(const size_t mi,
                                 std::array<size_t, 4>& idx_arr,
                                 double& value) {
  idx_arr = indices[mi];
  value = data[mi];
}

//
// Contraction Helpers
//

RowTensor4d contract_SparseTensor4d_0101(RowTensor4d& W, SparseTensor4d& T) {
  size_t di = W.dimensions()[2], dj = W.dimensions()[3], da = T.dimensions()[2],
         db = T.dimensions()[3];
  RowTensor4d output(di, dj, da, db);
  output.setZero();
  const size_t sp_size = T.size();

#pragma omp parallel for
  for (size_t i = 0; i < di; i++) {
    for (size_t j = 0; j < dj; j++) {
      for (size_t a = 0; a < da; a++) {
        for (size_t b = 0; b < db; b++) {
          // Loop over sparse indices
          for (size_t s = 0; s < sp_size; s++) {
            std::array<size_t, 4> idx;
            double value;
            T.get_element(s, idx, value);
            if (idx[2] == a && idx[3] == b) {
              size_t k = idx[0], l = idx[1];
              output(i, j, a, b) += W(k, l, i, j) * value;
            }
          }
        }
      }
    }
  }

  return output;
}

// t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
RowTensor4d contract_SparseTensor4d_2323(RowTensor4d& W, SparseTensor4d& T) {
  size_t di = T.dimensions()[0], dj = T.dimensions()[1], da = W.dimensions()[0],
         db = W.dimensions()[1];
  RowTensor4d output(di, dj, da, db);
  output.setZero();
  const size_t sp_size = T.size();

#pragma omp parallel for
  for (size_t i = 0; i < di; i++) {
    for (size_t j = 0; j < dj; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[0] == i && idx[1] == j) {
          size_t c = idx[2], d = idx[3];
          for (size_t a = 0; a < da; a++) {
            for (size_t b = 0; b < db; b++) {
              output(i, j, a, b) += W(a, b, c, d) * value;
            }
          }
        }
      }
    }
  }

  return output;
}

RowTensor4d contract_SparseTensor4d(
    RowTensor4d& W, SparseTensor4d& T,
    const Eigen::array<Eigen::IndexPair<int>, 2>& contraction_dims_2d,
    const Eigen::array<int, 4>& shuffle_idx_4d, const std::string term) {
  if (term == "0101") {
    return contract_SparseTensor4d_0101(W, T).shuffle(shuffle_idx_4d);
  } else if (term == "2323") {
    return contract_SparseTensor4d_2323(W, T);  //.shuffle(shuffle_idx_4d);
  } else {
    throw "CASE NOT IMPLEMENTED";
  }
}
