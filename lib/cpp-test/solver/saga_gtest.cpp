#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>

#include "tick/linear_model/model_linreg.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/solver/saga.h"
#include "toy_dataset.ipp"

TEST(SAGA, test_saga_dense_convergence) {
  SArrayDoublePtr labels_ptr = get_linreg_labels();
  SBaseArrayDouble2dPtr features_ptr = get_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  auto prox = std::make_shared<ProxL2Sq>(1e-1, false);

  SAGA saga(n_samples, 0, RandType::unif, model->get_lip_max() / 300, 1309);
  saga.set_rand_max(n_samples);
  saga.set_model(model);
  saga.set_prox(prox);

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    saga.solve();
  }
  saga.get_iterate(out_iterate30);
  double objective30 = model->loss(out_iterate30) + prox->value(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 30; ++j) {
    saga.solve();
  }
  saga.get_iterate(out_iterate60);
  double objective60 = model->loss(out_iterate60) + prox->value(out_iterate60);

  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
  EXPECT_LE(objective60 - objective30, 0.);
  EXPECT_LE(objective30 - objective60, 0.1);
}

TEST(SAGA, test_saga_sparse_convergence) {
  SArrayDoublePtr labels_ptr = get_linreg_labels();
  SBaseArrayDouble2dPtr features_ptr = get_sparse_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  auto prox = std::make_shared<ProxL2Sq>(1e-1, false);

  SAGA saga(n_samples, 0, RandType::unif, model->get_lip_max() / 300, 1309);
  saga.set_rand_max(n_samples);
  saga.set_model(model);
  saga.set_prox(prox);

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    saga.solve();
  }
  saga.get_iterate(out_iterate30);
  double objective30 = model->loss(out_iterate30) + prox->value(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 270; ++j) {
    saga.solve();
  }
  saga.get_iterate(out_iterate60);
  double objective300 = model->loss(out_iterate60) + prox->value(out_iterate60);

  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
  EXPECT_LE(objective300 - objective30, 0.);
  EXPECT_LE(objective30 - objective300, 0.1);

  ASAGA asaga(n_samples, 0, RandType::unif, model->get_lip_max() / 300, 50, 1309);
  asaga.set_rand_max(n_samples);
  asaga.set_model(model);
  asaga.set_prox(prox);

  asaga.solve(300);
  auto iterate_asaga = asaga.get_iterate_history().back();
  const auto objective_asaga = model->loss(*iterate_asaga) + prox->value(*iterate_asaga);
  EXPECT_LE(objective_asaga - objective300, 0.0001);
}

TEST(SAGA, test_saga_serialization) {
  SArrayDoublePtr labels_ptr = get_linreg_labels();
  SBaseArrayDouble2dPtr features_ptr = get_features();

  ulong n_samples = features_ptr->n_rows();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  auto prox = std::make_shared<ProxL2Sq>(1e-1, false);

  SAGA saga(n_samples, 0, RandType::unif, model->get_lip_max() / 300, 1309);
  saga.set_rand_max(n_samples);
  saga.set_model(model);
  saga.set_prox(prox);
  for (int j = 0; j < 10; ++j) {
    saga.solve();
  }

  std::stringstream os;
  {
    cereal::PortableBinaryOutputArchive outputArchive(os);
    outputArchive(saga);
  }

  {
    cereal::PortableBinaryInputArchive inputArchive(os);

    SAGA restored_saga;
    inputArchive(restored_saga);

    ASSERT_TRUE(saga == restored_saga);
  }
}

TEST(SAGA, test_asaga_serialization) {
  SArrayDoublePtr labels_ptr = get_linreg_labels();
  SBaseArrayDouble2dPtr features_ptr = get_sparse_features();

  ulong n_samples = features_ptr->n_rows();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  auto prox = std::make_shared<ProxL2Sq>(1e-1, false);

  ASAGA asaga(n_samples, 0, RandType::unif, model->get_lip_max() / 300, 1309);
  asaga.set_rand_max(n_samples);
  asaga.set_model(model);
  asaga.set_prox(prox);
  asaga.solve();

  ArrayDouble iterate1(asaga.get_model()->get_n_coeffs());
  asaga.get_iterate(iterate1);

  std::stringstream os;
  {
    cereal::PortableBinaryOutputArchive outputArchive(os);
    outputArchive(asaga);
  }
  {
    cereal::PortableBinaryInputArchive inputArchive(os);

    ASAGA restored_asaga;
    inputArchive(restored_asaga);

    ArrayDouble iterate(restored_asaga.get_model()->get_n_coeffs());
    restored_asaga.get_iterate(iterate);

    ASSERT_TRUE(asaga == restored_asaga);
  }
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
