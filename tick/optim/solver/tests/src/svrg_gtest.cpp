#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include <svrg.h>
#include <prox_l2sq.h>
#include <prox_l2sq.h>
#include "linreg.h"

SArrayDoublePtr get_labels() {
  ArrayDouble labels{-1.76, 2.6, -0.7, -1.84, -1.88, -1.78, 2.52};
  return labels.as_sarray_ptr();
}

SArrayDouble2dPtr get_features() {
  ulong n_samples = 7;
  ulong n_features = 5;

  ArrayDouble features_data{0.55, 2.04, 0.78, -0.00, 0.00,
                            -0.00, -2.62, -0.00, 0.00, 0.31,
                            -0.64, 0.94, 0.00, 0.55, -0.14,
                            0.93, 0.00, 0.00, -0.00, -2.39,
                            1.13, 0.05, -1.50, -0.50, -1.41,
                            1.41, 1.10, -0.00, 0.12, 0.00,
                            -0.00, -1.33, -0.00, 0.85, 3.03};

  ArrayDouble2d features(n_samples, n_features);
  for (int i = 0; i < features_data.size(); ++i) {
    features[i] = features_data[i];
  }
  return features.as_sarray2d_ptr();
}

SSparseArrayDouble2dPtr get_sparse_features() {
  ulong n_samples = 7;
  ulong n_features = 5;
  // no need to free, it will be done by sparse array
  double* sparse_data = new double[22]{0.55, 2.04, 0.78, -2.62, 0.31, -0.64, 0.94, 0.55, -0.14, 0.93, -2.39,
                                       1.13, 0.05, -1.50, -0.50, -1.41, 1.41, 1.10, 0.12, -1.33, 0.85, 3.03};
  INDICE_TYPE* sparse_indices = new INDICE_TYPE[22] {0, 1, 2, 1, 4, 0, 1, 3, 4, 0, 4, 0, 1, 2, 3,
                                                     4, 0, 1, 3, 1, 3, 4};
  INDICE_TYPE* sparse_indptr = new INDICE_TYPE[8] {0, 3, 5, 9, 11, 16, 19, 22};

  SSparseArrayDouble2dPtr sparse_features = SSparseArrayDouble2d::new_ptr(0, 0, 0);
  sparse_features->set_data_indices_rowindices(sparse_data, sparse_indices, sparse_indptr,
                                               n_samples, n_features);
  return sparse_features;
}

TEST(SVRG, test_convergence) {
  SArrayDoublePtr labels_ptr = get_labels();
  SArrayDouble2dPtr features_ptr = get_features();

  ulong n_samples = features_ptr->n_rows();
  ulong n_features = features_ptr->n_cols();

  auto model = std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1);
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
