#ifndef FRI_UTILS_HPP
#define FRI_UTILS_HPP
#include <fricc.hpp>
#include <utility>

// Sorting utilities

/**
 * @brief Partially sort and the array and return the first m indices necessary
 * to sort it.
 *
 * @param array
 * @param m The number of sorted values we want to return.
 * @return std::vector<size_t>
 */
std::vector<size_t> partial_argsort_paired(const std::vector<double>& array,
                                           const size_t m);
std::vector<size_t> partial_argsort(const Eigen::Ref<Eigen::VectorXd>& array,
                                    const size_t m);

//
// Sampling Functions
//
std::vector<size_t> sample_pivotal(const size_t& n_sample,
                                   const std::vector<double>& probs);
std::vector<size_t> sample_systematic(const size_t& n_sample,
                                      const std::vector<double>& probs);

std::vector<size_t> parallel_sample(const size_t& n_sample,
                                    const std::vector<double>& vector);

/**
 * @brief A helper class for some of the sorting utilities to reduce cache
 * misses and keep the indices and values closer to eachother.
 *
 */
class valuePair {
 public:
  size_t idx;
  double value;
  // Default construction necessary to use with stl containers
  valuePair() {
    idx = 0;
    value = 0.0;
  }
  /**
   * @brief Construct a new value Pair object
   *
   * @param idx The index to store.
   * @param value The value hold onto for later comparison.
   */
  valuePair(const size_t idx, const double value) : idx(idx), value(value) {}
};

// Making probability vector
std::vector<double> make_probability_vector(const std::vector<double>& x,
                                            const size_t n_sample,
                                            const std::vector<size_t>& D,
                                            const double remaining_norm);

std::pair<std::vector<size_t>, double> get_d_largest(
    const std::vector<double>& x, const size_t n_sample);

std::pair<std::vector<size_t>, std::vector<double>> fri_compression(
    const std::vector<double>& x, const size_t n_sample,
    const std::string sampling_method = "pivotal", const bool verbose = false);

#endif