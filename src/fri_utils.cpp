#include <fri_utils.hpp>

void argsort(Eigen::Ref<Eigen::VectorXd> array, Eigen::Ref<VecXST> sorted_idx) {
  std::iota(sorted_idx.data(), sorted_idx.data() + sorted_idx.size(), 0);
  std::sort(std::execution::par_unseq, sorted_idx.data(),
            sorted_idx.data() + sorted_idx.size(),
            [&array](int left, int right) -> bool {
              // sort indices according to corresponding array element in
              // DESCENDING ORDER
              return array[left] < array[right];
            });
}

void get_m_largest(Eigen::Ref<Eigen::VectorXd> v, const size_t m,
                   Eigen::Ref<VecXST> v_largest_idx) {
  //   sorted_idx.resize(v.size());
  std::cout << "Starting get_m_largest" << std::endl;
  VecXST sorted_idx(v.size());
  argsort(v, sorted_idx);
  std::copy(std::execution::par_unseq, sorted_idx.data(), sorted_idx.data() + m,
            v_largest_idx.data());
}