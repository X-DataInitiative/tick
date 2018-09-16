#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include "tick/linear_model/model_linreg.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/solver/svrg.h"
#include "toy_dataset.ipp"

TEST(SVRG, test_convergence) {
  SArrayDoublePtr labels_ptr = get_labels();
  SArrayDouble2dPtr features_ptr = get_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model =
      std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  SVRG svrg(n_samples, 0, RandType::unif, model->get_lip_max() / 100, 1309);
  svrg.set_rand_max(n_samples);
  svrg.set_model(model);
  svrg.set_prox(std::make_shared<ProxL2Sq>(1e-3, false));

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    svrg.solve();
  }
  svrg.get_iterate(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 30; ++j) {
    svrg.solve();
  }
  svrg.get_iterate(out_iterate60);

  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
