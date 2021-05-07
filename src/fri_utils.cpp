#include <algorithm>
#include <execution>
#include <fri_utils.hpp>

#include "tbb/parallel_sort.h"

/**
 * @brief Mimics numpy.argsort and populates a vector of the indices that sort
 * the vector array.
 *
 * @param array The vector we want to sort.
 * @param sorted_idx The vector of indices that sorts array.
 */
void argsort(const Eigen::Ref<Eigen::VectorXd>& array,
             Eigen::Ref<VecXST> sorted_idx) {
  // std::iota(sorted_idx.data(), sorted_idx.data() + sorted_idx.size(), 0);
  auto t_iota = std::chrono::steady_clock::now();
#pragma omp parallel for simd
  for (size_t i = 0; i < sorted_idx.size(); i++) {
    sorted_idx[i] = i;
  }
  log_timing("IOTA Time", t_iota);

  std::sort(std::execution::par_unseq, sorted_idx.data(),
            sorted_idx.data() + sorted_idx.size(),
            [&array](const size_t& left, const size_t& right) -> bool {
              // sort indices according to corresponding array element in
              // DESCENDING ORDER
              return abs(array[left]) > abs(array[right]);
            });
}

// /**
//  * @brief Get the indices of the m largest largest elements in v.
//  *
//  * @param v Vector that we want to know the largest elements of.
//  * @param m The number of largest elements we want.
//  * @param v_largest_idx The indices for the m largest elements in v.
//  */
void get_m_largest(const Eigen::Ref<Eigen::VectorXd>& v, const size_t m,
                   Eigen::Ref<VecXST> v_largest_idx) {
  auto t_alloc = std::chrono::steady_clock::now();
  VecXST sorted_idx(v.size());
  log_timing("Allocation time", t_alloc);

  auto t_argsort = std::chrono::steady_clock::now();
  argsort(v, sorted_idx);
  log_timing("Argsort time", t_argsort);

  auto t_copy = std::chrono::steady_clock::now();
  std::copy(std::execution::par_unseq, sorted_idx.data(), sorted_idx.data() + m,
            v_largest_idx.data());
  log_timing("Copy time", t_copy);
}

void my_parallel_sort(Eigen::Ref<Eigen::VectorXd>& v) {
  std::sort(std::execution::par_unseq, v.begin(), v.end());
}
