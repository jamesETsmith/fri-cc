#include <omp.h>

// algo has to come before execution
#include <algorithm>
#include <cstdlib>
#include <fri_utils.hpp>
#include <iomanip>
#include <numeric>
#include <sorting_sandbox/mergesort.hpp>  // For parallel mergesort implementation

std::vector<size_t> argsort(const std::vector<double>& array, const size_t m) {
  // Create pairs
  std::vector<std::pair<double, size_t>> index_pairs(array.size());

#pragma omp parallel for simd schedule(static, 16)
  for (size_t i = 0; i < array.size(); i++) {
    index_pairs[i] = std::make_pair(array[i], i);
  }

  // Sort pairs

  merge_sort(index_pairs, "taskyield", [](const auto& left, const auto& right) {
    return abs(left.first) > abs(right.first);
  });

  // Copy indices of largest magnitude to sorted_index and return
  std::vector<size_t> sorted_idx(m);

#pragma omp parallel for simd schedule(static, 16)
  for (size_t i = 0; i < sorted_idx.size(); i++) {
    sorted_idx[i] = index_pairs[i].second;
  }

  return sorted_idx;
}

//
// Sampling functions
//

// Used only in
// sampling_systematic(), generates a
// linearly spaced vector and adds
std::vector<double> lin_space_add_const(const double start, const double stop,
                                        size_t num, bool inclusive = false) {
  const double rn = (double)rand() / RAND_MAX;
  // std::cout << "SYSTEMATIC RANDOM
  // NUMBER " << rn << std::endl;
  std::vector<double> v(num);
  const double step =
      inclusive ? (stop - start) / (num - 1.) : (stop - start) / num;

#pragma omp parallel for simd
  for (size_t i = 0; i < num; i++) {
    v[i] = i * step + rn;
  }

  return v;
}

//
//
//
std::vector<size_t> sample_pivotal(const size_t& n_sample,
                                   const std::vector<double>& probs) {
  // Out list of sampled indices
  std::vector<size_t> S;  //(n_sample);
  S.reserve(n_sample);

  size_t i = 0;
  double a = probs[i];

  for (size_t j = 1; j < probs.size(); j++) {
    a += probs[j];
    double rn = (double)rand() / RAND_MAX;

    if (a < 1.) {
      if (rn < probs[j] / a) {
        i = j;
      }

    } else {
      a -= 1.;
      double prob_accept = (1. - probs[j]) / (1. - a);
      if (prob_accept > 1 || prob_accept < 0) {
        std::cerr << "ERROR IN PIVOTAL "
                     "SAMPLING"
                  << std::endl;
        std::cerr << "Pivotal acceptance "
                     "probability is "
                  << prob_accept << std::endl;
        std::cerr << "i = " << i << " j = " << j << std::endl;
        std::cerr << "a = " << a << " probs[j] = " << probs[j] << std::endl;
        exit(EXIT_FAILURE);
      }
      if (rn < prob_accept) {
        S.push_back(i);
        i = j;
      } else {
        S.push_back(j);
      }
    }

    if (S.size() == n_sample) {
      break;
    }
    if (j == (probs.size() - 1)) {
      if (abs(a - 1.) < 1e-5) {
        S.push_back(i);
      } else {
        std::cerr << "ERROR IN PIVOTAL "
                     "SAMPLING"
                  << std::endl;
        std::cerr << "a - 1 = " << abs(a - 1.) << std::endl;
        std::cerr << "Length of S " << S.size() << std::endl;
        std::cerr << "S should be " << n_sample << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  if (S.size() != n_sample) {
    std::cerr << "Wrong number of elements" << std::endl;
    exit(EXIT_FAILURE);
  }

  return S;
}  // End sample_pivotal()

std::vector<size_t> sample_systematic(const size_t& n_sample,
                                      const std::vector<double>& probs) {
  // Out list of sampled indices
  std::vector<size_t> S(n_sample);

  // Generate the locations of all
  // our randomly sampled locations
  std::vector<double> uk = lin_space_add_const(0, n_sample, n_sample);

  // Divide our vector of
  // probabilities (probs) into
  // intervals so we can see what
  // intervals our {uk} fell into
  std::vector<double> intervals(probs.size() + 1);
  std::partial_sum(probs.begin(), probs.end(), intervals.begin() + 1);

  // Figure out where our samples
  // "landed" and save the index to S
  size_t last_interval = 0;

  for (size_t i = 0; i < uk.size(); i++) {
    for (size_t j = last_interval; j < intervals.size() - 1; j++) {
      if (uk[i] >= intervals[j] && uk[i] < intervals[j + 1]) {
        S[i] = j;
        last_interval = j;
        break;
      }
    }
  }
  return S;
}  // End of sample_systematic

//
//
// DUMB WAY: TODO FIX ME
double one_norm(const std::vector<double>& q) {
  double q_norm = 0.0;
#pragma omp parallel for simd reduction(+ : q_norm)
  for (int i = 0; i < q.size(); i++) {
    q_norm += abs(q[i]);
  }
  return q_norm;
}

//
//
//
std::vector<size_t> parallel_sample(const size_t& n_sample,
                                    const std::vector<double>& q) {
  // std::cout << "Starting parallel
  // sample" << std::endl;
  std::vector<size_t> S(n_sample);

  // Thread shorthand
  const int n_threads = omp_get_max_threads();
  const size_t step = q.size() / n_threads;
  const size_t rem = q.size() % n_threads;
  // std::cout << "Step and remainder
  // " << step << " " << rem <<
  // std::endl;

  // DUMB WAY: TODO FIX ME
  double q_norm = one_norm(q);
  std::vector<double> t(n_threads);
  std::vector<int> g(n_threads);
  std::vector<double> a(n_threads);
  std::vector<double> qj_norms(n_threads);

  std::vector<std::vector<double>> qj(n_threads,
                                      std::vector<double>(step, 0.0));
  qj[n_threads - 1].resize(step + rem);
  // for (const auto& qji : qj[0])
  // std::cout << qji << std::endl;

  //
  // Setup and divide work among
  // threads
  //
  // std::cout << "Dividing work
  // among threads" << std::endl;

  size_t c = n_sample;

  auto t_partition = std::chrono::steady_clock::now();
  // #pragma omp parallel for
  // reduction(+ : c)
  for (int j = 0; j < n_threads; j++) {
    size_t ub = (j + 1) * step;
    if (j + 1 == n_threads) {
      ub = (j + 1) * step + rem;
    }
    std::copy(q.begin() + j * step, q.begin() + ub, qj[j].begin());
    // std::cout << "Done copy" <<
    // std::endl;

    qj_norms[j] = one_norm(qj[j]);

    // std::cout << "Done one norm "
    // << qj_norms[j] << std::endl;

    //
    a[j] = n_sample * qj_norms[j] / q_norm;
    t[j] = a[j] - floor(a[j]);

    g[j] = floor(a[j]);
    c -= g[j];
    // std::cout << "Done with a t g
    // c" << std::endl;
  }
  log_timing("Time to partition work", t_partition);
  // for (const auto& qji : qj[0])
  // std::cout << qji << std::endl;
  // double t_sum =
  // std::accumulate(t.begin(),
  // t.end(), 0.0); double
  // qj_norm_sum =
  // std::accumulate(qj_norms.begin(),
  // qj_norms.end(), 0.0); double
  // g_sum =
  // std::accumulate(g.begin(),
  // g.end(), 0.0); std::cout <<
  // std::setprecision(12); std::cout
  // << "T_SUM " << t_sum << "
  // QJ_NORM_SUM " << qj_norm_sum <<
  // " G sum "
  //           << g_sum << std::endl;
  // std::cout << "C " << c <<
  // std::endl;

  // std::cout << c << std::endl;
  // for (const auto& tj : t)
  // std::cout << tj << std::endl;
  // std::cout << "Sampling which
  // processes to increase the load
  // of" << std::endl;
  std::vector<size_t> s_prime = sample_pivotal(c, t);
  // std::cout << "S'" << std::endl;
  for (const size_t& s_prime_i : s_prime) {
    // std::cout << s_prime_i <<
    // std::endl;

    g[s_prime_i] += 1;
  }
  // std::cout << "Done with sampling
  // helpers" << std::endl;

  // Where to put the sampled indices
  std::vector<double> chunk_idx(g.begin(), g.end());
  std::partial_sum(chunk_idx.begin(), chunk_idx.end(), chunk_idx.begin());
  // std::cout << "Chunk sizes" <<
  // std::endl; for (const double&
  // idx : chunk_idx) {
  //   std::cout << idx << std::endl;
  // }

  // std::cout << "t  a  g  qj_norm"
  // << std::endl; for (int i = 0; i
  // < n_threads; i++) {
  //   std::cout << t[i] << " " <<
  //   a[i] << " " << g[i] << " " <<
  //   qj_norms[i]
  //             << std::endl;
  // }

  auto t_parallel = std::chrono::steady_clock::now();
#pragma omp parallel
  {
    const int j = omp_get_thread_num();
    double sj = qj_norms[j];
    // #pragma omp critical
    //     { std::cout << "Hello from
    //     thread " << j <<
    //     std::endl; }

    if (g[j] > a[j]) {
      for (int i = 0; i < qj[j].size(); i++) {
        double yji = std::min(1., qj[j][i] / t[j]);
        sj += yji - qj[j][i];
        qj[j][i] = yji;
        if (sj >= g[j]) {
          qj[j][i] = yji + g[j] - sj;
          break;
        }
      }
    } else if (g[j] < a[j]) {
      for (int i = 0; i < qj[j].size(); i++) {
        double yji = std::max(0., (qj[j][i] - t[j]) / (1. - t[j]));
        sj += yji - qj[j][i];
        qj[j][i] = yji;
        if (sj <= g[j]) {
          qj[j][i] = yji + g[j] - sj;
          break;
        }
      }
    } else {
      // std::cout << "DIDN'T HAVE TO
      // ADJUST ANYTHING" <<
      // std::endl;
    }

    //
    // Sample
    //
    int lb = 0;
    if (j > 0) {
      lb = chunk_idx[j - 1];
    }
    const int ub = chunk_idx[j];

    // std::cout << "qj[j]" <<
    // std::endl; for (const auto&
    // qji : qj[j]) std::cout << qji
    // << std::endl; std::cout <<
    // std::endl;

    auto t_sample = std::chrono::steady_clock::now();
    std::vector<size_t> Sj = sample_pivotal(g[j], qj[j]);
    log_timing("Time for parallel sampling", t_sample);

    // std::cout << "S[j]" <<
    // std::endl; for (const auto& sj
    // : Sj) std::cout << sj <<
    // std::endl;

    const size_t incr = j * step;
    std::transform(Sj.begin(), Sj.end(), Sj.begin(),
                   [&](const size_t& Sji) { return Sji + incr; });
    std::copy(Sj.begin(), Sj.end(), S.begin() + lb);
  }

  log_timing("Time for parallel work", t_parallel);

  return S;
}

//
//
//
std::vector<double> make_probability_vector(const std::vector<double>& x,
                                            const size_t n_sample,
                                            const std::vector<size_t>& D,
                                            const double remaining_norm) {
  std::vector<double> p(x.size());

#pragma omp parallel for simd schedule(static, 16)
  for (size_t i = 0; i < p.size(); i++) {
    p[i] = abs(x[i]) / remaining_norm * n_sample;
  }

#pragma omp parallel for simd schedule(static, 16)
  for (size_t i = 0; i < D.size(); i++) {
    p[D[i]] = 0.0;
  }

  for (size_t i = 0; i < p.size(); i++) {
    if (p[i] > 1.) {
      std::cerr << "ERROR MAKING "
                   "PROBABILITY VECTOR"
                << std::endl;
      std::cerr << "p[i] = " << p[i] << std::endl;
      std::cerr << "i = " << i << " abs(x[i]) = " << abs(x[i])
                << " remaining_norm = " << remaining_norm
                << " n_sample = " << n_sample << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  return p;
}

std::pair<std::vector<size_t>, double> get_d_largest(
    const std::vector<double>& x, const size_t n_sample) {
  double remaining_norm = one_norm(x);
  // std::cout << "FRI-C++: 1-NORM OF
  // X (t2) " << remaining_norm <<
  // std::endl;
  std::vector<size_t> D;
  D.reserve(n_sample);

  std::vector<size_t> sort_idx = argsort(x, n_sample);

  for (size_t i = 0; i < n_sample; i++) {
    auto idx = sort_idx[i];
    auto d = D.size();
    auto xi = abs(x[idx]);
    if ((n_sample - d) * xi >= remaining_norm - xi && remaining_norm > 1e-14) {
      D.push_back(idx);
      remaining_norm -= xi;
    } else {
      break;
    }
  }

  // for (int i = 0; i < 10; i++) {
  //   std::cout << i << " th largest
  //   element with value " <<
  //   x[sort_idx[i]]
  //             << std::endl;
  // }

  // if (remaining_norm < 1e-12) {
  //   std::cout << "REMAINING NORM
  //   IS VERY SMALL " <<
  //   remaining_norm << std::endl;
  // }

  return std::make_pair(D, remaining_norm);
}

std::pair<std::vector<size_t>, std::vector<double>> fri_compression(
    const std::vector<double>& x, const size_t n_sample,
    const std::string sampling_method, const bool verbose) {
  // Input checking
  if (!sampling_method.compare("pivotal") &&
      !sampling_method.compare("systematic")) {
    std::cerr << "ERROR";
    std::cerr << "\tThe sampling method "
                 "you chose ("
              << sampling_method << ") isn't supported ";
    std::cerr << "sampling_method "
                 "must be 'pivotal' "
                 "or 'systematic'"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  auto _t_total = std::chrono::steady_clock::now();

  // Setup outputs
  std::vector<size_t> compressed_idx(n_sample);
  std::vector<double> compressed_vals(n_sample);

  // Determine the largest elements
  // within budget Time: O(N*log(N))
  // Memory: O(2*N)
  auto _t_get_d_largest = std::chrono::steady_clock::now();
  auto [D, remaining_norm] = get_d_largest(x, n_sample);
  auto t_get_d_largest = get_timing(_t_get_d_largest);
  // std::cout << "Size of D " <<
  // D.size() << std::endl;

  // Calculate a vector of
  // probabilities for sampling each
  // element Time: O(N) Memory: O(N)
  auto _t_p_vector = std::chrono::steady_clock::now();
  std::vector<double> p =
      make_probability_vector(x, n_sample - D.size(), D, remaining_norm);
  auto t_p_vector = get_timing(_t_p_vector);

  // Sample the remaining number of
  // elements we want Time: O(N)
  // Memory: O(m)
  auto _t_sample = std::chrono::steady_clock::now();

  std::vector<size_t> S;
  if (!sampling_method.compare("pivotal")) {
    S = sample_pivotal(n_sample - D.size(), p);
  } else if (!sampling_method.compare("systematic")) {
    S = sample_systematic(n_sample - D.size(), p);
  }
  auto t_sample = get_timing(_t_sample);

  // Move indices to compressed_idx
  auto _t_wrap_up = std::chrono::steady_clock::now();
  std::copy(D.begin(), D.end(), compressed_idx.begin());
  std::copy(S.begin(), S.end(), compressed_idx.begin() + D.size());

  // Calculate values
  for (size_t i = 0; i < D.size(); i++) {
    compressed_vals[i] = x[compressed_idx[i]];
  }

  for (size_t i = D.size(); i < compressed_idx.size(); i++) {
    compressed_vals[i] = x[compressed_idx[i]] / p[compressed_idx[i]];
  }
  auto t_wrap_up = get_timing(_t_wrap_up);

  // Timing summary
  auto t_total = get_timing(_t_total);
  if (verbose) {
    // clang-format off
    printf("Subsection    Fraction of Total    Total Time\n");
    printf("==========    =================    ==========\n");
    printf("Sorting       %6.4f               %6.4f (s)\n", t_get_d_largest / t_total, t_get_d_largest);
    printf("P vector      %6.4f               %6.4f (s)\n", t_p_vector / t_total, t_p_vector);
    printf("Sampling      %6.4f               %6.4f (s)\n", t_sample / t_total, t_sample);
    printf("Wrap up       %6.4f               %6.4f (s)\n", t_wrap_up / t_total, t_wrap_up);
    printf("Total         %6.4f               %6.4f (s)\n", t_total / t_total, t_total);
    // clang-format on
  }

  return std::make_pair(compressed_idx, compressed_vals);
}