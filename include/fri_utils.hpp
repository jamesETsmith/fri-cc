#ifndef FRI_UTILS_HPP
#define FRI_UTILS_HPP
#include <fricc.hpp>

// Sorting utilities
void argsort(Eigen::Ref<Eigen::VectorXd> array, Eigen::Ref<VecXST> sorted_idx);
void get_m_largest(Eigen::Ref<Eigen::VectorXd> v, const size_t m,
                   Eigen::Ref<VecXST> v_largest_idx);
void my_parallel_sort(Eigen::Ref<Eigen::VectorXd>& v);

#endif