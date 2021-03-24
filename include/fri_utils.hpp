#ifndef FRI_UTILS_HPP
#define FRI_UTILS_HPP
#include <algorithm>  // because of std::sort
#include <execution>
#include <fricc.hpp>

using VecXST = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;

void argsort(Eigen::Ref<Eigen::VectorXd> array, Eigen::Ref<VecXST> sorted_idx);
void get_m_largest(Eigen::Ref<Eigen::VectorXd> v, const size_t m,
                   Eigen::Ref<VecXST> v_largest_idx);
#endif