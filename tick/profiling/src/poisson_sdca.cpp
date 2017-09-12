

#include "test_rand.h"
#include "sdca.h"
#include "poisreg.h"
#include "prox_zero.h"

int main() {
  const int seed = 1933;
  const ulong n_samples = 30000;
  const ulong n_features = 1000;
  const double l_l2sq = 1e-1;
  const int n_iter = 100;

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

  auto sdca = SDCA(l_l2sq, n_samples, 0);
  sdca.set_rand_max(non_zero_label);

  sdca.set_model(model);
  sdca.set_prox(std::make_shared<ProxZero>(0, 0, 1));

  for (int j = 0; j < n_iter; ++j) {
    sdca.solve();
  }

  sdca.get_primal_vector()->print();

  return 0;
}