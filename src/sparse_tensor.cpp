#include <omp.h>

#include <sparse_tensor.hpp>

using pytensor_4d = pybind11::detail::unchecked_mutable_reference<double, 4>;

//
// Contraction Helpers
//

// 0101 O^4V^2
void contract_SparseTensor4d_0101_wrapper(pytensor_4d W, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
  for (size_t a = 0; a < nv; a++) {
    for (size_t b = 0; b < nv; b++) {
      // Loop over sparse indices
      for (size_t s = 0; s < sp_size; s++) {
        std::array<size_t, 4> idx;
        double value;
        T.get_element(s, idx, value);
        if (idx[2] == a && idx[3] == b) {
          size_t k = idx[0], l = idx[1];
          for (size_t i = 0; i < no; i++) {
            for (size_t j = 0; j < no; j++) {
              output(i, j, a, b) += W(k, l, i, j) * value;
            }
          }
        }
      }
    }
  }
}

// 2323 O^2V^4 `t2new += lib.einsum('abcd,ijcd->ijab', Wvvvv, tau)`
void contract_SparseTensor4d_2323_wrapper(pytensor_4d W, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

  // std::cout << "Pointer in C++ " << output_vec.data() << std::endl;

  size_t di = T.dimensions()[0], dj = T.dimensions()[1], da = W.shape(0),
         db = W.shape(1);

  // Debug
  // double one_norm = 0.0;
  // for (size_t i = 0; i < no * no * nv * nv; i++) {
  //   one_norm += abs(output_vec[i]);
  // }
  // std::cout << "1-NORM INSIDE DTSpT b4 contraction " << one_norm <<
  // std::endl;
  //

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

  // DEBUG ONLY
  // double one_norm = 0.0;
  // for (size_t i = 0; i < no * no * nv * nv; i++) {
  //   one_norm += abs(output_vec[i]);
  // }
  // std::cout << "1-NORM INSIDE DTSpT " << one_norm << std::endl;
}

// 1302 O^3V^3 `tmp  = 2*lib.einsum('akic,kjcb->ijab', Wvoov, t2)`
void contract_SparseTensor4d_1302_wrapper(pytensor_4d W, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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
void contract_SparseTensor4d_1202_wrapper(pytensor_4d W, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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
void contract_SparseTensor4d_1303_wrapper(pytensor_4d W, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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
void contract_SparseTensor4d_1203_wrapper(pytensor_4d W, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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

// O^4V^2 `Wklij += lib.einsum('kcld,ijcd->klij', eris_ovov, t2)`
void contract_SparseTensor4d_1323_wrapper(pytensor_4d ovov, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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
              output(k, l, i, j) += ovov(k, c, l, d) * value;
            }
          }
        }
      }
    }
  }
}

// O^3V^3 `Wakic -= 0.5*lib.einsum('ldkc,ilda->akic', eris_ovov, t2)`
void contract_SparseTensor4d_0112_wrapper(pytensor_4d ovov, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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

// O^3V^3 `Wakic -= 0.5*lib.einsum('lckd,ilad->akic', eris_ovov, t2)`
void contract_SparseTensor4d_0313_wrapper(pytensor_4d ovov, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic, 16) collapse(2)
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
#pragma unroll 4
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

// O^3V^3 `Wakic += lib.einsum("ldkc,ilad->akic", eris_ovov, t2)`
void contract_SparseTensor4d_0113_wrapper(pytensor_4d ovov, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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
              output(a, k, i, c) += ovov(l, d, k, c) * value;
            }
          }
        }
      }
    }
  }
}

// O^3V^3 `Wakci -= 0.5*lib.einsum('lckd,ilda->akci', eris_ovov, t2)`
void contract_SparseTensor4d_0312_wrapper(pytensor_4d ovov, SparseTensor4d &T,
                                          pytensor_4d output) {
  const size_t no = T.dimension(0);
  const size_t nv = T.dimension(2);
  const size_t sp_size = T.size();

#pragma omp parallel for schedule(dynamic) collapse(2)
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

// Main way to interact with contraction kernels
void contract_SparseTensor4d_wrapper(py::array_t<double> W_raw,
                                     SparseTensor4d &T,
                                     py::array_t<double> output_raw,
                                     const std::string term) {
  auto W = W_raw.mutable_unchecked<4>();
  auto output = output_raw.mutable_unchecked<4>();

  if (term == "0101") {
    contract_SparseTensor4d_0101_wrapper(W, T, output);
  } else if (term == "2323") {
    contract_SparseTensor4d_2323_wrapper(W, T, output);
  } else if (term == "1302") {
    contract_SparseTensor4d_1302_wrapper(W, T, output);
  } else if (term == "1202") {
    contract_SparseTensor4d_1202_wrapper(W, T, output);
  } else if (term == "1303") {
    contract_SparseTensor4d_1303_wrapper(W, T, output);
  } else if (term == "1203") {
    contract_SparseTensor4d_1203_wrapper(W, T, output);
  } else if (term == "1323") {
    contract_SparseTensor4d_1323_wrapper(W, T, output);
  } else if (term == "0112") {
    contract_SparseTensor4d_0112_wrapper(W, T, output);
  } else if (term == "0113") {
    contract_SparseTensor4d_0113_wrapper(W, T, output);
  } else if (term == "0313") {
    contract_SparseTensor4d_0313_wrapper(W, T, output);
  } else if (term == "0312") {
    contract_SparseTensor4d_0312_wrapper(W, T, output);
  } else {
    std::cerr << "CASE NOT IMPLEMENTED" << std::endl;
    exit(EXIT_FAILURE);
  }
}