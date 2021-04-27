#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

//

#include <fri_utils.hpp>
#include <sparse_tensor.hpp>

TEST_CASE("LOG_TIMING") {
  auto start = std::chrono::steady_clock::now();
  log_timing("Testing Timing", start);
}

TEST_CASE("M=10 N=100") {
  const size_t M = 10, N = 100;
  Eigen::VectorXd v = Eigen::VectorXd::Random(N);
  Eigen::VectorXd v_sorted = v;
  std::sort(v_sorted.data(), v_sorted.data() + v_sorted.size(),
            std::greater<>());

  VecXST v_largest_idx(M);
  get_m_largest(v, M, v_largest_idx);

  for (size_t i = 0; i < M; i++) {
    CHECK(v[v_largest_idx[i]] == v_sorted[i]);
    // std::cout << v_sorted[i] << std::endl;
  }
}
