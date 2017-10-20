#include <chrono>

#include "test_rand.h"
#include "linreg.h"
#include "logreg.h"

int main(int argc, char *argv[]) {

  ulong n_samples = 20000;

  if (argc == 2) {
    std::istringstream ss(argv[1]);
    if (!(ss >> n_samples))
      std::cerr << "Invalid number for n_samples: " << argv[1] << '\n';
  }

  const int seed = 1933;
  const ulong n_features = 1000;
  const int n_iter = 200;

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

  int n_threads = 4;
//  mkl_set_num_threads_local(4);
  auto model = std::make_shared<ModelLinReg>(features, labels, false, n_threads);

  // run solver
  using milli = std::chrono::microseconds;
  auto start = std::chrono::high_resolution_clock::now();

  ArrayDouble coeffs = view_row(*features, 0);
//  ArrayDouble out(n_samples);

  volatile double loss = 0;
  for (int j = 0; j < n_iter; ++j) {
//    model->grad2(coeffs, out);
    loss = model->loss2(coeffs);
  }

  std::cout << "loss = " << loss << std::endl;

//  out.print();
  auto finish = std::chrono::high_resolution_clock::now();

  const char *filename = argv[0];
  std::cout << filename << " ";
  std::cout << std::chrono::duration_cast<milli>(finish - start).count() / 1e6 << std::endl;

  return 0;
}