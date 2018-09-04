#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include "tick/linear_model/model_linreg.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/solver/sgd.h"

SArrayDoublePtr get_labels() {
  ArrayDouble labels{-1.76, 2.6, -0.7, -1.84, -1.88, -1.78, 2.52};
  return labels.as_sarray_ptr();
}

SSparseArrayDouble2dPtr get_sparse_features() {
  ulong n_samples = 7;
  ulong n_features = 5;
  // no need to free, it will be done by sparse array
  double* sparse_data = new double[22]{
      0.55, 2.04, 0.78,  -2.62, 0.31,  -0.64, 0.94, 0.55, -0.14, 0.93, -2.39,
      1.13, 0.05, -1.50, -0.50, -1.41, 1.41,  1.10, 0.12, -1.33, 0.85, 3.03};
  INDICE_TYPE* sparse_indices = new INDICE_TYPE[22]{
      0, 1, 2, 1, 4, 0, 1, 3, 4, 0, 4, 0, 1, 2, 3, 4, 0, 1, 3, 1, 3, 4};
  INDICE_TYPE* sparse_indptr = new INDICE_TYPE[8]{0, 3, 5, 9, 11, 16, 19, 22};

  SSparseArrayDouble2dPtr sparse_features =
      SSparseArrayDouble2d::new_ptr(0, 0, 0);
  sparse_features->set_data_indices_rowindices(
      sparse_data, sparse_indices, sparse_indptr, n_samples, n_features);
  return sparse_features;
}

TEST(SGD, test_convergence) {
  SArrayDoublePtr labels_ptr = get_labels();
  SSparseArrayDouble2dPtr features_ptr = get_sparse_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model =
      std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  SGD sgd(n_samples, 0, RandType::unif, model->get_lip_max() / 100, 1309);
  sgd.set_rand_max(n_samples);
  sgd.set_model(model);
  auto prox = std::make_shared<TProxL2Sq<double> >(1e-3, false);
  sgd.set_prox(prox);

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    sgd.solve();
  }
  sgd.get_iterate(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 30; ++j) {
    sgd.solve();
  }
  sgd.get_iterate(out_iterate60);

  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
}

TEST(SGD, test_convergence_batch) {
  SArrayDoublePtr labels_ptr = get_labels();
  SSparseArrayDouble2dPtr features_ptr = get_sparse_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model =
      std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
  SGD sgd(n_samples, 0, RandType::unif, model->get_lip_max() / 100, 1309);
  sgd.set_rand_max(n_samples);
  sgd.set_model(model);
  auto prox = std::make_shared<TProxL2Sq<double> >(1e-3, false);
  sgd.set_prox(prox);

  ArrayDouble out_iterate30(n_features);
  for (int j = 0; j < 30; ++j) {
    sgd.solve_sparse_batch(1);
  }
  sgd.get_iterate(out_iterate30);

  ArrayDouble out_iterate60(n_features);
  for (int j = 0; j < 30; ++j) {
    sgd.solve_sparse_batch(1);
  }
  sgd.get_iterate(out_iterate60);

  out_iterate60.mult_incr(out_iterate30, -1);
  EXPECT_LE(out_iterate60.norm_sq() / n_features, 0.1);
}



#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
