#include <chrono>

#include "tick/array/serializer.h"
#include "tick/random/test_rand.h"
#include "tick/optim/solver/saga.h"
#include "tick/optim/model/logreg.h"
#include "tick/optim/model/linreg.h"
#include "tick/optim/prox/prox_zero.h"

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/os.hpp"
#endif

const constexpr size_t SEED       = 1933;
const constexpr size_t N_ITER     = 30;
const constexpr size_t N_FEATURES = 200;

int main(int argc, char *argv[]) {
  ulong n_samples = 750000;
  if (argc == 2) {
    std::istringstream ss(argv[1]);
    if (!(ss >> n_samples))
      std::cerr << "Invalid number for n_samples: " << argv[1] << '\n';
  }

  {
    n_samples = 750000;
    const auto sample = test_uniform(n_samples * N_FEATURES, SEED);
    ArrayDouble2d sample2d(n_samples, N_FEATURES, sample->data());
    const auto features = SArrayDouble2d::new_ptr(sample2d);
    const auto int_sample = test_uniform_int(0, 2, n_samples, SEED);
    SArrayDoublePtr labels = SArrayDouble::new_ptr(n_samples);
    for (int i = 0; i < n_samples; ++i) (*labels)[i] = (*int_sample)[i] - 1;

    using milli = std::chrono::microseconds;
    {
      auto model = std::make_shared<ModelLogReg>(features, labels, false);
      SAGA saga(n_samples, 0, RandType::unif, 1e-3);
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      saga.set_prox(std::make_shared<ProxZero>(0, 0, 1));
      auto start = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < N_ITER; ++j) saga.solve();
      auto finish = std::chrono::high_resolution_clock::now();
      std::cout << argv[0] << " "
                << std::chrono::duration_cast<milli>(finish - start).count() / 1e6 
                << std::endl;
    }
    {
      auto model = std::make_shared<ModelLinReg>(features, labels, false);
      SAGA saga(n_samples, 0, RandType::unif, 1e-3);
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      saga.set_prox(std::make_shared<ProxZero>(0, 0, 1));
      auto start = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < N_ITER; ++j) saga.solve();
      auto finish = std::chrono::high_resolution_clock::now();
      std::cout << argv[0] << " "
                << std::chrono::duration_cast<milli>(finish - start).count() / 1e6 
                << std::endl;
    }
  }

  return 0;
}