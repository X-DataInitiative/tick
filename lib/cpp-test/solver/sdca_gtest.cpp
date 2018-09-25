#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>
#include <cereal/types/memory.hpp>
#include <lib/include/tick/prox/prox_l2sq.h>

#include "tick/linear_model/model_linreg.h"
#include "tick/prox/prox_zero.h"
#include "tick/solver/sdca.h"
#include "tick/solver/asdca.h"
#include "toy_dataset.ipp"

TEST(SDCA, test_sdca_dense_convergence) {
  SArrayDoublePtr labels_ptr = get_labels();
  SArrayDouble2dPtr features_ptr = get_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  auto prox = std::make_shared<TProxZero<double>>(false);

  double l_l2sq = 1e-1;
  auto objective_prox = std::make_shared<TProxL2Sq<double>>(l_l2sq, false);

  SDCA sdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
  sdca.set_rand_max(n_samples);
  sdca.set_model(model);
  sdca.set_prox(prox);

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    sdca.solve();
  }
  sdca.get_iterate(out_iterate30);
  double objective30 = model->loss(out_iterate30) + objective_prox->value(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 300; ++j) {
    sdca.solve();
  }
  sdca.get_iterate(out_iterate60);
  double objective60 = model->loss(out_iterate60) + objective_prox->value(out_iterate60);


  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
  EXPECT_LE(objective60 - objective30, 0.);
  EXPECT_LE(objective30 - objective60, 0.1);
}

TEST(SDCA, test_sdca_sparse_convergence) {
  SArrayDoublePtr labels_ptr = get_labels();
  SBaseArrayDouble2dPtr features_ptr = get_sparse_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  auto prox = std::make_shared<TProxZero<double>>(false);

  double l_l2sq = 1e-1;
  auto objective_prox = std::make_shared<TProxL2Sq<double>>(l_l2sq, false);

  SDCA sdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
  sdca.set_rand_max(n_samples);
  sdca.set_model(model);
  sdca.set_prox(prox);

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    sdca.solve();
  }
  sdca.get_iterate(out_iterate30);
  double objective30 = model->loss(out_iterate30) + objective_prox->value(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 270; ++j) {
    sdca.solve();
  }
  sdca.get_iterate(out_iterate60);
  double objective300 = model->loss(out_iterate60) + objective_prox->value(out_iterate60);

  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
  EXPECT_LE(objective300 - objective30, 0.);
  EXPECT_LE(objective30 - objective300, 0.1);

  auto atomic_model = std::make_shared<TModelLinReg<double, std::atomic<double>>>(
      features_ptr, labels_ptr, false, 1);
  auto atomic_prox = std::make_shared<TProxZero<double, std::atomic<double>>>(false);

  ASDCA asdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 2);
  asdca.set_rand_max(n_samples);
  asdca.set_model(atomic_model);
  asdca.set_prox(atomic_prox);

  asdca.solve(300);
  auto iterate_sdca = sdca.get_iterate_history().back();
  auto iterate_asdca = asdca.get_iterate_history().back();
  const auto objective_asdca = model->loss(*iterate_asdca) + objective_prox->value(*iterate_asdca);
  EXPECT_LE(objective_asdca - objective300, 0.0001);
  EXPECT_LE(objective300 - objective_asdca, 0.0001);

  ASDCA asdca_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 2);
  asdca_batch.set_rand_max(n_samples);
  asdca_batch.set_model(atomic_model);
  asdca_batch.set_prox(atomic_prox);

  asdca_batch.solve_batch(300, 3);
  auto iterate_asdca_batch = asdca_batch.get_iterate_history().back();
  const auto objective_asdca_batch = model->loss(*iterate_asdca_batch) + objective_prox->value(*iterate_asdca_batch);

  EXPECT_LE(objective_asdca_batch - objective300, 0.0001);
  EXPECT_LE(objective300 - objective_asdca_batch, 0.0001);
}


#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
