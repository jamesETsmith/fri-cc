#include "tbb/parallel_sort.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <vector>

template <typename Clock>
void log_timing(std::string msg, std::chrono::time_point<Clock> start) {
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;  // in seconds
  printf("%-24s %4.2f (s)\n", msg.c_str(), elapsed.count());
}

void fill_vector(std::vector<double>& v1) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < v1.size(); i++) {
    v1[i] = pow(-1., i % 2) * i;
  }
}

int main() {
  //
  const size_t N = 1e9;
  std::vector<double> v(N);
  std::vector<double> v2(N);
  std::vector<double> v3(N);
  fill_vector(v);
  std::copy(std::execution::par_unseq, v.begin(), v.end(), v2.begin());
  std::copy(std::execution::par_unseq, v.begin(), v.end(), v3.begin());

  auto tbb_time = std::chrono::steady_clock::now();
  tbb::parallel_sort(v.begin(), v.end());
  log_timing("TBB Parallel Sort", tbb_time);

  auto p_time = std::chrono::steady_clock::now();
  std::sort(std::execution::par_unseq, v3.begin(), v3.end());
  log_timing("Parallel Sort", p_time);

  return 0;
}