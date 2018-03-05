#include <chrono>

#include "tick/linear_model/model_logreg.h"
#include "tick/random/test_rand.h"

int main(int nargs, char *args[]) {
  ulong num_threads = 1;
  ulong n_samples = 20000;

  if (nargs > 1) num_threads = std::stoul(args[1]);

  if (nargs > 2) n_samples = std::stoul(args[1]);

  const int seed = 1933;
  ulong num_runs = 5;

  const ulong n_features = 10000;
  const int num_iterations = 10;

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

  for (ulong run_i = 0; run_i < num_runs; ++run_i) {
    auto model =
        std::make_shared<ModelLogReg>(features, labels, false, num_threads);

    using milli = std::chrono::microseconds;
    const auto start = std::chrono::system_clock::now();

    ArrayDouble coeffs = view_row(*features, 0);

    volatile double loss = 0;
    for (int j = 0; j < num_iterations; ++j) {
      loss = model->loss(coeffs);
    }
    const auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << elapsed_seconds.count() << '\t' << num_iterations << '\t'
              << num_threads << '\t' << n_samples << '\t' << n_features << '\t'
              << args[0] << '\t' << std::endl;
  }

  return 0;
}