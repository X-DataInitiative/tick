#include <chrono>

#include "test_rand.h"
#include "sgd.h"
#include "logreg.h"
#include "prox_zero.h"

int main(int argc, char *argv[]) {

  ulong n_samples = 100000;

  if (argc == 2) {
    std::istringstream ss(argv[1]);
    if (!(ss >> n_samples))
      std::cerr << "Invalid number for n_samples: " << argv[1] << '\n';
  }

  const int seed = 1933;
  const ulong n_features = 100;
  const int n_iter = 10;

  // generate random data
  const auto sample = test_uniform(n_samples * n_features, seed);
  ArrayDouble2d sample2d(n_samples, n_features, sample->data());
  const auto features = SArrayDouble2d::new_ptr(sample2d);

  const auto int_sample = test_uniform_int(0, 2, n_samples, seed);
  SArrayDoublePtr labels = SArrayDouble::new_ptr(n_samples);

  // Set up model and solver
  for (int i = 0; i < n_samples; ++i) {
    (*labels)[i] = (*int_sample)[i] - 1;
  }

  auto model = std::make_shared<ModelLogReg>(features, labels, false);

  auto sgd = SGD(n_samples, 0, RandType::seq);
  sgd.set_jump_sec(1);
  sgd.set_rand_max(n_samples);

  sgd.set_model(model);
  sgd.set_prox(std::make_shared<ProxZero>(0, 0, 1));

  // run solver
  using milli = std::chrono::microseconds;
  auto start = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < n_iter; ++j) {
    sgd.solve();
  }

  auto finish = std::chrono::high_resolution_clock::now();

  const char *filename = argv[0];
  std::cout << filename << " ";
  std::cout << std::chrono::duration_cast<milli>(finish - start).count() / 1e6 << std::endl;

  return 0;
}