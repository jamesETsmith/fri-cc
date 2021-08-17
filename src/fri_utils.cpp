#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <execution>
#include <fri_utils.hpp>
#include <iomanip>
#include <numeric>

#include "tbb/parallel_sort.h"

//
// Sorting routines
//

//
// Partially argsort an array using a special paired datatype to reduce jumping
// around in memory. The special type (valuePair) is described in fri_utils.hpp
// and keeps the index and value "next" to each other in memory.
//
std::vector<size_t> partial_argsort_paired(
    const Eigen::Ref<Eigen::VectorXd>& array, const size_t m) {
  std::vector<valuePair> pairs_full(array.size());

#pragma omp parallel for simd
  for (size_t i = 0; i < pairs_full.size(); i++) {
    pairs_full[i] = valuePair(i, abs(array[i]));
  }

  // auto t_psort = std::chrono::steady_clock::now();

  std::partial_sort(std::execution::par_unseq, pairs_full.begin(),
                    pairs_full.begin() + m, pairs_full.end(),
                    [](const valuePair& left, const valuePair& right) -> bool {
                      return left.value > right.value;
                    });

  // log_timing("Sorting time", t_psort);

  std::vector<size_t> sorted_idx(m);
#pragma omp parallel for simd
  for (size_t i = 0; i < m; i++) {
    sorted_idx[i] = pairs_full[i].idx;
  }
  return sorted_idx;
}

//
// A less efficient way to partially sort an array than
// partial_argsort_paired().
//
std::vector<size_t> partial_argsort(const Eigen::Ref<Eigen::VectorXd>& array,
                                    const size_t m) {
  std::vector<size_t> indices_full(array.size());
  std::vector<size_t> sorted_idx(m);

#pragma omp parallel for simd
  for (size_t i = 0; i < indices_full.size(); i++) {
    indices_full[i] = i;
  }

  std::partial_sort_copy(
      std::execution::par, indices_full.begin(), indices_full.end(),
      sorted_idx.begin(), sorted_idx.end(),
      [&array](const size_t& left, const size_t& right) -> bool {
        // sort indices according to corresponding array element in
        // DESCENDING ORDER
        return abs(array[left]) > abs(array[right]);
      });

  return sorted_idx;
}

//
// A thingly wrapped version of std::partial_sort() used only to compare
// performance of argsort functions.
//
std::vector<double> my_partial_sort(Eigen::Ref<Eigen::VectorXd>& v,
                                    const size_t m) {
  std::vector<double> sorted(m);
  std::partial_sort_copy(std::execution::par_unseq, v.begin(), v.end(),
                         sorted.begin(), sorted.end());
  return sorted;
}

//
// Sampling functions
//

// Used only in sampling_systematic(), generates a linearly spaced vector and
// adds
std::vector<double> lin_space_add_const(const double start, const double stop,
                                        size_t num, bool inclusive = false) {
  const double rn = (double)rand() / RAND_MAX;
  // std::cout << "SYSTEMATIC RANDOM NUMBER " << rn << std::endl;
  std::vector<double> v(num);
  const double step =
      inclusive ? (stop - start) / (num - 1.) : (stop - start) / num;

#pragma omp simd
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
        std::cout << "ERROR IN PIVOTAL SAMPLING" << std::endl;
        std::cout << "Pivotal acceptance probability is " << prob_accept
                  << std::endl;
        throw "BAD PROBABILITY";
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
    if (j == probs.size() - 1) {
      if (abs(a - 1.) < 1e-10) {
        S.push_back(i);
      } else {
        std::cout << "ERROR IN PIVOTAL SAMPLING" << std::endl;
        std::cout << "a - 1 = " << a - 1 << std::endl;
        std::cout << "Length of S " << S.size() << std::endl;
        throw "a isn't close enough to 1. ";
      }
    }
  }

  if (S.size() != n_sample) {
    std::cout << "Wrong number of elements" << std::endl;
    throw "S is the wrong size";
  }

  return S;
}  // End sample_pivotal()

std::vector<size_t> sample_systematic(const size_t& n_sample,
                                      const std::vector<double>& probs) {
  // Out list of sampled indices
  std::vector<size_t> S(n_sample);

  // Generate the locations of all our randomly sampled locations
  std::vector<double> uk = lin_space_add_const(0, n_sample, n_sample);

  // Divide our vector of probabilities (probs) into intervals so we can see
  // what intervals our {uk} fell into
  std::vector<double> intervals(probs.size() + 1);
  std::partial_sum(probs.begin(), probs.end(), intervals.begin() + 1);

  // Figure out where our samples "landed" and save the index to S
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
#pragma omp parallel for reduction(+ : q_norm)
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
  // std::cout << "Starting parallel sample" << std::endl;
  std::vector<size_t> S(n_sample);

  // Thread shorthand
  const int n_threads = omp_get_max_threads();
  const size_t step = q.size() / n_threads;
  const size_t rem = q.size() % n_threads;
  // std::cout << "Step and remainder " << step << " " << rem << std::endl;

  // DUMB WAY: TODO FIX ME
  double q_norm = one_norm(q);
  std::vector<double> t(n_threads);
  std::vector<int> g(n_threads);
  std::vector<double> a(n_threads);
  std::vector<double> qj_norms(n_threads);

  std::vector<std::vector<double>> qj(n_threads,
                                      std::vector<double>(step, 0.0));
  qj[n_threads - 1].resize(step + rem);
  // for (const auto& qji : qj[0]) std::cout << qji << std::endl;

  //
  // Setup and divide work among threads
  //
  // std::cout << "Dividing work among threads" << std::endl;

  size_t c = n_sample;

  auto t_partition = std::chrono::steady_clock::now();
  // #pragma omp parallel for reduction(+ : c)
  for (int j = 0; j < n_threads; j++) {
    size_t ub = (j + 1) * step;
    if (j + 1 == n_threads) {
      ub = (j + 1) * step + rem;
    }
    std::copy(q.begin() + j * step, q.begin() + ub, qj[j].begin());
    // std::cout << "Done copy" << std::endl;

    qj_norms[j] = one_norm(qj[j]);

    // std::cout << "Done one norm " << qj_norms[j] << std::endl;

    //
    a[j] = n_sample * qj_norms[j] / q_norm;
    t[j] = a[j] - floor(a[j]);

    g[j] = floor(a[j]);
    c -= g[j];
    // std::cout << "Done with a t g c" << std::endl;
  }
  log_timing("Time to partition work", t_partition);
  // for (const auto& qji : qj[0]) std::cout << qji << std::endl;
  // double t_sum = std::accumulate(t.begin(), t.end(), 0.0);
  // double qj_norm_sum = std::accumulate(qj_norms.begin(), qj_norms.end(),
  // 0.0); double g_sum = std::accumulate(g.begin(), g.end(), 0.0); std::cout <<
  // std::setprecision(12); std::cout << "T_SUM " << t_sum << " QJ_NORM_SUM " <<
  // qj_norm_sum << " G sum "
  //           << g_sum << std::endl;
  // std::cout << "C " << c << std::endl;

  // std::cout << c << std::endl;
  // for (const auto& tj : t) std::cout << tj << std::endl;
  // std::cout << "Sampling which processes to increase the load of" <<
  // std::endl;
  std::vector<size_t> s_prime = sample_pivotal(c, t);
  // std::cout << "S'" << std::endl;
  for (const size_t& s_prime_i : s_prime) {
    // std::cout << s_prime_i << std::endl;

    g[s_prime_i] += 1;
  }
  // std::cout << "Done with sampling helpers" << std::endl;

  // Where to put the sampled indices
  std::vector<double> chunk_idx(g.begin(), g.end());
  std::partial_sum(chunk_idx.begin(), chunk_idx.end(), chunk_idx.begin());
  // std::cout << "Chunk sizes" << std::endl;
  // for (const double& idx : chunk_idx) {
  //   std::cout << idx << std::endl;
  // }

  // std::cout << "t  a  g  qj_norm" << std::endl;
  // for (int i = 0; i < n_threads; i++) {
  //   std::cout << t[i] << " " << a[i] << " " << g[i] << " " << qj_norms[i]
  //             << std::endl;
  // }

  auto t_parallel = std::chrono::steady_clock::now();
#pragma omp parallel
  {
    const int j = omp_get_thread_num();
    double sj = qj_norms[j];
    // #pragma omp critical
    //     { std::cout << "Hello from thread " << j << std::endl; }

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
      // std::cout << "DIDN'T HAVE TO ADJUST ANYTHING" << std::endl;
    }

    //
    // Sample
    //
    int lb = 0;
    if (j > 0) {
      lb = chunk_idx[j - 1];
    }
    const int ub = chunk_idx[j];

    // std::cout << "qj[j]" << std::endl;
    // for (const auto& qji : qj[j]) std::cout << qji << std::endl;
    // std::cout << std::endl;

    auto t_sample = std::chrono::steady_clock::now();
    std::vector<size_t> Sj = sample_pivotal(g[j], qj[j]);
    log_timing("Time for parallel sampling", t_sample);

    // std::cout << "S[j]" << std::endl;
    // for (const auto& sj : Sj) std::cout << sj << std::endl;

    const size_t incr = j * step;
    std::transform(Sj.begin(), Sj.end(), Sj.begin(),
                   [&](const size_t& Sji) { return Sji + incr; });
    std::copy(Sj.begin(), Sj.end(), S.begin() + lb);
  }

  log_timing("Time for parallel work", t_parallel);

  return S;
}