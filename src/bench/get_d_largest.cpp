#include "fri_utils.hpp"

int main() {
  // Seed RNG
  srand(20);

  // Setup random vector
  size_t N = 100000000;
  std::vector<double> x(N);
  std::generate(x.begin(), x.end(), rand);

  // Get the largest d elements
  size_t n_sample = N / 1e2;
  auto [D, remaining_norm] = get_d_largest(x, n_sample);

  return 0;
}