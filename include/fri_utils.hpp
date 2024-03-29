#ifndef FRI_UTILS_HPP
#define FRI_UTILS_HPP
#include <fricc.hpp>
#include <utility>

// Sorting utilities

std::vector<size_t> argsort(std::span<double const> array, const size_t m);

//
// Sampling Functions
//
std::vector<size_t> sample_pivotal(const size_t& n_sample,
                                   const std::vector<double>& probs);
std::vector<size_t> sample_systematic(const size_t& n_sample,
                                      const std::vector<double>& probs);

std::vector<size_t> parallel_sample(const size_t& n_sample,
                                    std::span<double const> vector);

// Making probability vector
std::vector<double> make_probability_vector(std::span<double const> x,
                                            const size_t n_sample,
                                            const std::vector<size_t>& D,
                                            const double remaining_norm);

std::pair<std::vector<size_t>, double> get_d_largest(
    std::span<double const> x, const size_t n_sample);

std::pair<std::vector<size_t>, std::vector<double>> fri_compression(
    std::span<double const> x, const size_t n_sample,
    const std::string sampling_method = "pivotal", const bool verbose = false);

#endif