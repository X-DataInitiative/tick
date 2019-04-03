#include "tick/base/base.h"
#include "tick/random/test_rand.h"


//
// Benchmark matrix vector dot products performances
// The command lines arguments are the following
// n_samples : number of observations (number of rows)
// n_features : number variables per observation (number of columns)
// n_threads : number of threads used
// num_runs : number of run for each timing
// num_iterations : number of timings
//
// Example
// Then run a matrix vector product with 20000 rows, 10000 colums, 2 threads,  5 runs and 10 iterations
// threads
// ./matrix_vector_product 20000 10000 2 5 10
//

#include "tick/linear_model/model_logreg.h"

int main(int nargs, char *args[]) {

  ulong n_samples = 20000;
  if (nargs > 1) n_samples = std::stoul(args[1]);

  ulong n_features = 10000;
  if (nargs > 2) n_features = std::stoul(args[2]);

  ulong num_threads = 1;
  if (nargs > 3) num_threads = std::stoul(args[3]);

  ulong num_runs = 5;
  if (nargs > 4) num_runs = std::stoul(args[4]);

  ulong num_iterations = 10;
  if (nargs > 5) num_iterations = std::stoul(args[5]);

  const int seed = 1933;

  // generate random data
  const auto sample = test_uniform(n_samples * n_features, seed);
  ArrayDouble2d sample2d(n_samples, n_features, sample->data());
  const auto features = SArrayDouble2d::new_ptr(sample2d);

  const auto int_sample = test_uniform_int(0, 2, n_samples, seed);
  SArrayDoublePtr labels = SArrayDouble::new_ptr(n_samples);

  // Set up model and solver
  for (size_t i = 0; i < n_samples; ++i) {
    (*labels)[i] = (*int_sample)[i] - 1;
  }

  for (ulong run_i = 0; run_i < num_runs; ++run_i) {
    auto model =
        std::make_shared<ModelLogReg>(features, labels, false, num_threads);

    using milli = std::chrono::microseconds;
    const auto start = std::chrono::system_clock::now();

    ArrayDouble coeffs = view_row(*features, 0);

    volatile double loss = 0;
    for (size_t j = 0; j < num_iterations; ++j) {
      loss = model->loss(coeffs);
    }
    const auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << elapsed_seconds.count() << '\t' << num_iterations << '\t'
              << num_threads << '\t' << n_samples << '\t' << n_features << '\t'
              << args[0] << '\t' << std::endl;
  }
}