#include <chrono>

#include "tick/random/test_rand.h"
#include "tick/solver/sdca.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/prox/prox_zero.h"

int main(int argc, char *argv[]) {

  ulong n_samples = 10000;

  if (argc == 2) {
    std::istringstream ss(argv[1]);
    if (!(ss >> n_samples))
      std::cerr << "Invalid number for n_samples: " << argv[1] << '\n';
  }

  const size_t seed = 1933;
  const ulong n_features = 1000;
  const double l_l2sq = 1e-1;
  const size_t n_iter = 100;

  // generate random data
  const auto sample = test_uniform(n_samples * n_features, seed);
  ArrayDouble2d sample2d(n_samples, n_features, sample->data());
  const auto features = SArrayDouble2d::new_ptr(sample2d);

  const auto int_sample = test_uniform_int(0, 4, n_samples, seed);
  SArrayDoublePtr labels = SArrayDouble::new_ptr(n_samples);

  // Set up model and solver
  ulong non_zero_label = 0;
  for (size_t i = 0; i < n_samples; ++i) {
    (*labels)[i] = (*int_sample)[i];
    if ((*labels)[i] != 0) {
      non_zero_label++;
    }
  }

  auto model = std::make_shared<ModelPoisReg>(features, labels, LinkType::identity, false);

  auto sdca = SDCA(l_l2sq, n_samples, 0);
  sdca.set_rand_max(non_zero_label);

  sdca.set_model(model);
  sdca.set_prox(std::make_shared<TProxZero<double> >(0, 0, 1));

  // run solver
  using milli = std::chrono::microseconds;
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t j = 0; j < n_iter; ++j) {
    sdca.solve();
  }

  auto finish = std::chrono::high_resolution_clock::now();

  const char *filename = argv[0];
  std::cout << filename << " ";
  std::cout << std::chrono::duration_cast<milli>(finish - start).count() / 1e6 << std::endl;

  return 0;
}