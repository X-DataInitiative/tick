

#include "test_rand.h"
#include "sdca.h"
#include "poisreg.h"
#include "prox_zero.h"

int main() {
  const int seed = 1933;
  const ulong n_samples = 10000;
  const ulong n_features = 1000;
  const double l_l2sq = 1e-1;
  const int n_iter = 300;

  const auto sample = test_uniform(n_samples * n_features, seed);
  ArrayDouble2d sample2d(n_samples, n_features, sample->data());
  const auto features = SArrayDouble2d::new_ptr(sample2d);

  const auto int_sample = test_uniform_int(0, 4, n_samples, seed);
  SArrayDoublePtr labels = SArrayDouble::new_ptr(n_samples);

  ulong non_zero_label = 0;
  for (int i = 0; i < n_samples; ++i) {
    (*labels)[i] = (*int_sample)[i];
    if ((*labels)[i] != 0) {
      non_zero_label++;
    }
  }

  auto model = std::make_shared<ModelPoisReg>(features, labels, LinkType::identity, false);

  auto sdca = SDCA(l_l2sq, n_samples, 0, RandType::unif, BatchSize::two, 2, 12);
  sdca.set_rand_max(non_zero_label);

  sdca.set_model(model);
  sdca.set_prox(std::make_shared<ProxZero>(0, 0, 1));

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  for (int j = 0; j < n_iter; ++j) {
    sdca.solve();
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double dif = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-9;

//  sdca.get_primal_vector()->print();
  printf("Elasped time is %lf seconds. Second value = %lf\n",
         dif, (*sdca.get_primal_vector())[1]);

  return 0;
}