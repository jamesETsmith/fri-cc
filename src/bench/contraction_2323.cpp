#include "sparse_tensor.hpp"

int main(int argc, char** argv) {
  // Set up problem
  srand(20);
  size_t t2_size, n_sample;
  int nocc, nvirt;
  double frac;

  // Read commandline args (if they're there)
  if (argc != 4) {
    nocc = 40;
    nvirt = 80;
    frac = 1e-2;
    // Default settings originally took 4.0293 (s)
  } else {
    std::cout << "Setting args from command line" << std::endl;
    std::string _nocc = argv[1], _nvirt = argv[2], _frac = argv[3];
    nocc = std::stoi(_nocc);
    nvirt = std::stoi(_nvirt);
    frac = std::stod(_frac);
  }

  // Print what's happening
  t2_size = nocc * nocc * nvirt * nvirt;
  n_sample = t2_size * frac;
  std::cout << "|T_ijab| = " << t2_size << std::endl;
  std::cout << "|Wvvvv| = " << nvirt * nvirt * nvirt * nvirt << std::endl;
  std::cout << "|T_sparse| = " << n_sample << std::endl;

  // Populate data structures
  std::vector<double> t2_old(t2_size);
  Eigen::VectorXd Wvvvv =
      Eigen::VectorXd::Random(nvirt * nvirt * nvirt * nvirt);
  Eigen::VectorXd t2_new = Eigen::VectorXd::Zero(t2_size);
  std::generate(t2_old.begin(), t2_old.end(),
                [] { return (double)rand() / RAND_MAX; });
  SparseTensor4d t2_compressed(t2_old, {nocc, nocc, nvirt, nvirt}, n_sample,
                               "largest");

  Eigen::Ref<Eigen::VectorXd> W_ref = Wvvvv;
  Eigen::Ref<Eigen::VectorXd> t_ref = t2_new;
  auto t_contract = std::chrono::steady_clock::now();
  contract_SparseTensor4d_wrapper(W_ref, t2_compressed, t_ref, "2323");
  log_timing("2323 contraction timing", t_contract);

  return 0;
}