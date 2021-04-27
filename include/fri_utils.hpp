#ifndef FRI_UTILS_HPP
#define FRI_UTILS_HPP
#include <algorithm>  // because of std::sort
#include <execution>
#include <fricc.hpp>

using VecXST = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

void argsort(Eigen::Ref<Eigen::VectorXd> array, Eigen::Ref<VecXST> sorted_idx);
void get_m_largest(Eigen::Ref<Eigen::VectorXd> v, const size_t m,
                   Eigen::Ref<VecXST> v_largest_idx);

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

#endif