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
      std::cerr << "ERROR: " << idx_arr[i] << " >= " << dims[i]
                << " for dimension " << i << " !!!" << std::endl;
      std::cerr << "Indices are out of range for given tensor" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return idx_arr;
}

void SparseTensor4d::set_element(const size_t mi, const size_t idx,
                                 const double value) {
  if (mi > nnz) {
    std::cerr << "Index you requested is larger than the specified size"
              << std::endl;
    exit(EXIT_FAILURE);
  }
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

// void contract_SparseTensor4d_0101(RowTensor4d& W, SparseTensor4d& T,
//                                   RowTensor4d& output) {
//   size_t di = W.dimensions()[2], dj = W.dimensions()[3], da =
//   T.dimensions()[2],
//          db = T.dimensions()[3];

//   const size_t sp_size = T.size();

// #pragma omp parallel for schedule(dynamic)
//   for (size_t i = 0; i < di; i++) {
//     for (size_t j = 0; j < dj; j++) {
//       for (size_t a = 0; a < da; a++) {
//         for (size_t b = 0; b < db; b++) {
//           // Loop over sparse indices
//           for (size_t s = 0; s < sp_size; s++) {
//             std::array<size_t, 4> idx;
//             double value;
//             T.get_element(s, idx, value);
//             if (idx[2] == a && idx[3] == b) {
//               size_t k = idx[0], l = idx[1];
//               output(i, j, a, b) += W(k, l, i, j) * value;
//             }
//           }
//         }
//       }
//     }
//   }
// }

void contract_SparseTensor4d_0101_wrapper(
    Eigen::Ref<Eigen::VectorXd>& W_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d W(W_vec.data(), no, no, no, no);
  TMap4d output(output_vec.data(), no, no, nv, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < no; i++) {
    for (size_t j = 0; j < no; j++) {
      for (size_t a = 0; a < nv; a++) {
        for (size_t b = 0; b < nv; b++) {
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
}

// t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
// void contract_SparseTensor4d_2323(RowTensor4d& W, SparseTensor4d& T,
//                                   RowTensor4d& output) {
//   size_t di = T.dimensions()[0], dj = T.dimensions()[1], da =
//   W.dimensions()[0],
//          db = W.dimensions()[1];
//   const size_t sp_size = T.size();

// #pragma omp parallel for schedule(dynamic)
//   for (size_t i = 0; i < di; i++) {
//     for (size_t j = 0; j < dj; j++) {
//       // Loop over sparse indices
//       for (size_t s = 0; s < sp_size; s++) {
//         std::array<size_t, 4> idx;
//         double value;
//         T.get_element(s, idx, value);
//         if (idx[0] == i && idx[1] == j) {
//           size_t c = idx[2], d = idx[3];
//           for (size_t a = 0; a < da; a++) {
//             // #pragma unroll
//             for (size_t b = 0; b < db; b++) {
//               output(i, j, a, b) += W(a, b, c, d) * value;
//             }
//           }
//         }
//       }
//     }
//   }
// }

// t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)
void contract_SparseTensor4d_2323_wrapper(
    Eigen::Ref<Eigen::VectorXd>& W_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d W(W_vec.data(), nv, nv, nv, nv);
  TMap4d output(output_vec.data(), no, no, nv, nv);

  size_t di = T.dimensions()[0], dj = T.dimensions()[1], da = W.dimensions()[0],
         db = W.dimensions()[1];
  const size_t sp_size = T.size();

  std::array<size_t, 4> idx;
  double value;
#pragma omp parallel for schedule(dynamic) collapse(2) private(idx, value)
  for (size_t i = 0; i < di; i++) {
    for (size_t j = 0; j < dj; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
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
}

// 1302 O^3V^3 `tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)`
void contract_SparseTensor4d_1302_wrapper(
    Eigen::Ref<Eigen::VectorXd>& W_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d W(W_vec.data(), nv, no, no, nv);
  TMap4d output(output_vec.data(), no, no, nv, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < nv; b++) {
    for (size_t j = 0; j < no; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[3] == b && idx[1] == j) {
          size_t k = idx[0], c = idx[2];
          for (size_t i = 0; i < no; i++) {
            for (size_t a = 0; a < nv; a++) {
              // #pragma unroll
              output(i, j, a, b) += 2 * W(a, k, i, c) * value;
            }
          }
        }
      }
    }
  }
}

//  O^3V^3 `tmp -=   lib.einsum('akci,kjcb->ijab', Wvovo, t2)`
void contract_SparseTensor4d_1202_wrapper(
    Eigen::Ref<Eigen::VectorXd>& W_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d W(W_vec.data(), nv, no, nv, no);
  TMap4d output(output_vec.data(), no, no, nv, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < nv; b++) {
    for (size_t j = 0; j < no; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[3] == b && idx[1] == j) {
          size_t k = idx[0], c = idx[2];
          for (size_t i = 0; i < no; i++) {
            for (size_t a = 0; a < nv; a++) {
              // #pragma unroll
              output(i, j, a, b) -= W(a, k, c, i) * value;
            }
          }
        }
      }
    }
  }
}

// 1303 O^3V^3 `tmp = lib.einsum('akic,kjbc->ijab', Wvoov, t2)`
void contract_SparseTensor4d_1303_wrapper(
    Eigen::Ref<Eigen::VectorXd>& W_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d W(W_vec.data(), nv, no, no, nv);
  TMap4d output(output_vec.data(), no, no, nv, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t b = 0; b < nv; b++) {
    for (size_t j = 0; j < no; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[2] == b && idx[1] == j) {
          size_t k = idx[0], c = idx[3];
          for (size_t i = 0; i < no; i++) {
            for (size_t a = 0; a < nv; a++) {
              // #pragma unroll
              output(i, j, a, b) += W(a, k, i, c) * value;
            }
          }
        }
      }
    }
  }
}

// 1203 O^3V^3 `tmp = lib.einsum('bkci,kjac->ijab', Wvovo, t2)`
void contract_SparseTensor4d_1203_wrapper(
    Eigen::Ref<Eigen::VectorXd>& W_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d W(W_vec.data(), nv, no, nv, no);
  TMap4d output(output_vec.data(), no, no, nv, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t a = 0; a < nv; a++) {
    for (size_t j = 0; j < no; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[2] == a && idx[1] == j) {
          size_t k = idx[0], c = idx[3];
          for (size_t i = 0; i < no; i++) {
            for (size_t b = 0; b < nv; b++) {
              // #pragma unroll
              output(i, j, a, b) += W(b, k, c, i) * value;
            }
          }
        }
      }
    }
  }
}

// `Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)`
void contract_SparseTensor4d_1323_wrapper(
    Eigen::Ref<Eigen::VectorXd>& ovov_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d ovov(ovov_vec.data(), no, nv, no, nv);
  TMap4d output(output_vec.data(), no, no, no, no);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < no; i++) {
    for (size_t j = 0; j < no; j++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[0] == i && idx[1] == j) {
          size_t c = idx[2], d = idx[3];
          for (size_t k = 0; k < no; k++) {
            for (size_t l = 0; l < no; l++) {
              // #pragma unroll
              output(k, l, i, j) += ovov(k, c, l, d) * value;
            }
          }
        }
      }
    }
  }
}

// `Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)`
void contract_SparseTensor4d_0112_wrapper(
    Eigen::Ref<Eigen::VectorXd>& ovov_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d ovov(ovov_vec.data(), no, nv, no, nv);
  TMap4d output(output_vec.data(), nv, no, no, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t a = 0; a < nv; a++) {
    for (size_t i = 0; i < no; i++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[3] == a && idx[0] == i) {
          size_t l = idx[1], d = idx[2];
          for (size_t k = 0; k < no; k++) {
            for (size_t c = 0; c < nv; c++) {
              // #pragma unroll
              output(a, k, i, c) -= 0.5 * ovov(l, d, k, c) * value;
            }
          }
        }
      }
    }
  }
}

// `Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)`
void contract_SparseTensor4d_0313_wrapper(
    Eigen::Ref<Eigen::VectorXd>& ovov_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d ovov(ovov_vec.data(), no, nv, no, nv);
  TMap4d output(output_vec.data(), nv, no, no, nv);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t a = 0; a < nv; a++) {
    for (size_t i = 0; i < no; i++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[2] == a && idx[0] == i) {
          size_t l = idx[1], d = idx[3];
          for (size_t k = 0; k < no; k++) {
            for (size_t c = 0; c < nv; c++) {
              // #pragma unroll
              output(a, k, i, c) -= 0.5 * ovov(l, c, k, d) * value;
            }
          }
        }
      }
    }
  }
}

// `Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)`
void contract_SparseTensor4d_0312_wrapper(
    Eigen::Ref<Eigen::VectorXd>& ovov_vec, SparseTensor4d& T,
    Eigen::Ref<Eigen::VectorXd>& output_vec) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  TMap4d ovov(ovov_vec.data(), no, nv, no, nv);
  TMap4d output(output_vec.data(), nv, no, nv, no);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t a = 0; a < nv; a++) {
    for (size_t i = 0; i < no; i++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[3] == a && idx[0] == i) {
          size_t l = idx[1], d = idx[2];
          for (size_t k = 0; k < no; k++) {
            for (size_t c = 0; c < nv; c++) {
              // #pragma unroll
              output(a, k, c, i) -= 0.5 * ovov(l, c, k, d) * value;
            }
          }
        }
      }
    }
  }
}

// void contract_SparseTensor4d(RowTensor4d& W, SparseTensor4d& T,
//                              RowTensor4d& output, const std::string term) {
//   if (term == "0101") {
//     contract_SparseTensor4d_0101(W, T, output);
//   } else if (term == "2323") {
//     return contract_SparseTensor4d_2323(W, T, output);
//   } else {
//     throw "CASE NOT IMPLEMENTED";
//   }
// }

void contract_SparseTensor4d_wrapper(Eigen::Ref<Eigen::VectorXd>& W_vec,
                                     SparseTensor4d& T,
                                     Eigen::Ref<Eigen::VectorXd>& output_vec,
                                     const std::string term) {
  if (term == "0101") {
    // t2new += lib.einsum('klij,klab->ijab', Woooo, tau)
    contract_SparseTensor4d_0101_wrapper(W_vec, T, output_vec);
  } else if (term == "2323") {
    contract_SparseTensor4d_2323_wrapper(W_vec, T, output_vec);
  } else if (term == "1302") {
    contract_SparseTensor4d_1302_wrapper(W_vec, T, output_vec);
  } else if (term == "1202") {
    contract_SparseTensor4d_1202_wrapper(W_vec, T, output_vec);
  } else if (term == "1303") {
    contract_SparseTensor4d_1303_wrapper(W_vec, T, output_vec);
  } else if (term == "1203") {
    contract_SparseTensor4d_1203_wrapper(W_vec, T, output_vec);
  } else if (term == "1323") {
    contract_SparseTensor4d_1323_wrapper(W_vec, T, output_vec);
  } else if (term == "0112") {
    contract_SparseTensor4d_0112_wrapper(W_vec, T, output_vec);
  } else if (term == "0313") {
    contract_SparseTensor4d_0313_wrapper(W_vec, T, output_vec);
  } else if (term == "0312") {
    contract_SparseTensor4d_0312_wrapper(W_vec, T, output_vec);
  } else {
    std::cerr < "CASE NOT IMPLEMENTED" << std::endl;
    exit(EXIT_FAILURE);
  }
}