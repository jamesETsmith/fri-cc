#include <fri_utils.hpp>

/**
 * @brief Mimics numpy.argsort and populates a vector of the indices that sort
 * the vector array.
 *
 * @param array The vector we want to sort.
 * @param sorted_idx The vector of indices that sorts array.
 */
void argsort(Eigen::Ref<Eigen::VectorXd> array, Eigen::Ref<VecXST> sorted_idx) {
  std::iota(sorted_idx.data(), sorted_idx.data() + sorted_idx.size(), 0);
  std::sort(std::execution::par_unseq, sorted_idx.data(),
            sorted_idx.data() + sorted_idx.size(),
            [&array](int left, int right) -> bool {
              // sort indices according to corresponding array element in
              // DESCENDING ORDER
              return array[left] > array[right];
            });
}

/**
 * @brief Get the indices of the m largest largest elements in v.
 *
 * @param v Vector that we want to know the largest elements of.
 * @param m The number of largest elements we want.
 * @param v_largest_idx The indices for the m largest elements in v.
 */
void get_m_largest(Eigen::Ref<Eigen::VectorXd> v, const size_t m,
                   Eigen::Ref<VecXST> v_largest_idx) {
  VecXST sorted_idx(v.size());
  argsort(v, sorted_idx);
  std::copy(std::execution::par_unseq, sorted_idx.data(), sorted_idx.data() + m,
            v_largest_idx.data());
}