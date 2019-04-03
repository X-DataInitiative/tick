#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>

#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_poisreg.h"

#include "tick/prox/prox_zero.h"
#include "tick/prox/prox_l1.h"
#include "tick/prox/prox_l2sq.h"

#include "tick/solver/sdca.h"

#include "toy_dataset.ipp"

template <class T, class K>
double run_and_get_objective(TBaseSDCA<T, K> &sdca, std::shared_ptr<TModel<T> > model,
                             std::shared_ptr<TProx<T> > prox, size_t n_epochs, ulong batch_size = 1) {
  const auto objective_prox = std::make_shared<TProxL2Sq<T>>(sdca.get_l_l2sq(), false);
  Array<T> out_iterate(model->get_n_coeffs());

  if (batch_size == 1) sdca.solve(n_epochs);
  else sdca.solve_batch(n_epochs, batch_size);

  sdca.get_iterate(out_iterate);
  return model->loss(out_iterate) + prox->value(out_iterate) + objective_prox->value(out_iterate);
};
TEST(SDCA, test_sdca_sparse) {

  SBaseArrayDouble2dPtr sparse_features_ptr = get_sparse_features();
  SBaseArrayDouble2dPtr dense_features_ptr = get_features();

  ulong n_samples = dense_features_ptr->n_rows();
  const double l_l2sq = 0.1;
  const ulong n_epochs = 400;

  for (auto fit_intercept : std::vector<bool>{true, false}) {

    // TESTED MODELS
    std::vector<ModelPtr> dense_models;
    std::vector<ModelAtomicPtr> dense_atomic_models;
    std::vector<ModelPtr> sparse_models;
    std::vector<ModelAtomicPtr> sparse_atomic_models;
    for (auto is_sparse : std::vector<bool>{false, true}) {
      auto *models = is_sparse ? &sparse_models : &dense_models;
      auto *atomic_models = is_sparse ? &sparse_atomic_models : &dense_atomic_models;

      auto features_ptr = is_sparse ? sparse_features_ptr : dense_features_ptr;

      models->push_back(std::make_shared<ModelLinReg>(
          features_ptr, get_linreg_labels(), fit_intercept, 1));
      atomic_models->push_back(std::make_shared<ModelLinRegAtomic>(
          features_ptr, get_linreg_labels(), fit_intercept, 1));
      models->push_back(std::make_shared<ModelLogReg>(
          features_ptr, get_logreg_labels(), fit_intercept, 1));
      atomic_models->push_back(std::make_shared<ModelLogRegAtomic>(
          features_ptr, get_logreg_labels(), fit_intercept, 1));
      models->push_back(std::make_shared<ModelPoisReg>(
          features_ptr, get_poisson_labels(), LinkType::exponential, fit_intercept, 1));
      atomic_models->push_back(std::make_shared<ModelPoisRegAtomic>(
          features_ptr, get_poisson_labels(), LinkType::exponential, fit_intercept, 1));
      models->push_back(std::make_shared<ModelPoisReg>(
          features_ptr, get_poisson_labels(), LinkType::identity, fit_intercept, 1));
      atomic_models->push_back(std::make_shared<ModelPoisRegAtomic>(
          features_ptr, get_poisson_labels(), LinkType::identity, fit_intercept, 1));
    }

    // TESTED PROXS
    std::vector<ProxPtr> proxs;
    std::vector<ProxAtomicPtr> atomic_proxs;
    if (fit_intercept) {
      const auto n_features = dense_features_ptr->n_cols();
      proxs.push_back(std::make_shared<TProxZero<double>>(false));
      atomic_proxs.push_back(std::make_shared<TProxZero<double, std::atomic<double>>>(false));
      proxs.push_back(std::make_shared<TProxL1<double> >(0.002, 0, n_features, false));
      atomic_proxs.push_back(std::make_shared<TProxL1<double, std::atomic<double>>>(0.002,
                                                                                    0,
                                                                                    n_features,
                                                                                    false));
    } else {
      proxs.push_back(std::make_shared<TProxZero<double>>(false));
      atomic_proxs.push_back(std::make_shared<TProxZero<double, std::atomic<double>>>(false));
      proxs.push_back(std::make_shared<TProxL1<double> >(0.002, false));
      atomic_proxs.push_back(std::make_shared<TProxL1<double, std::atomic<double>>>(0.002, false));
    }

    for (size_t i = 0; i < dense_models.size(); ++i) {
      for (size_t j = 0; j < proxs.size(); ++j) {

        auto dense_model = dense_models[i];
        auto dense_atomic_model = dense_atomic_models[i];
        auto sparse_model = sparse_models[i];
        auto sparse_atomic_model = sparse_atomic_models[i];

        auto prox = proxs[j];
        auto atomic_prox = atomic_proxs[j];

        ulong rand_max = dense_model->get_sdca_index_map() == nullptr ?
                         n_samples : dense_model->get_sdca_index_map()->size();

        // Check Sparse and dense equivalence after one epoch
        SCOPED_TRACE("sparse / dense equivalence with"
                     " fit_intercept: " + std::to_string(fit_intercept)
                         + ", model: " + dense_model->get_class_name()
                         + ", prox: " + prox->get_class_name());

        SDCA dense_sdca_one_epoch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        dense_sdca_one_epoch.set_rand_max(rand_max);
        dense_sdca_one_epoch.set_model(dense_model);
        dense_sdca_one_epoch.set_prox(prox);

        SDCA sparse_sdca_one_epoch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sparse_sdca_one_epoch.set_rand_max(rand_max);
        sparse_sdca_one_epoch.set_model(sparse_model);
        sparse_sdca_one_epoch.set_prox(prox);

        ArrayDouble dense_out_iterate(dense_model->get_n_coeffs());
        dense_sdca_one_epoch.solve(2);
        dense_sdca_one_epoch.get_iterate(dense_out_iterate);

        ArrayDouble sparse_out_iterate(sparse_model->get_n_coeffs());
        sparse_sdca_one_epoch.solve(2);
        sparse_sdca_one_epoch.get_iterate(sparse_out_iterate);
        for (ulong k = 0; k < dense_model->get_n_coeffs(); ++k) {
          EXPECT_FLOAT_EQ(dense_out_iterate[k], sparse_out_iterate[k]);
        }


        // Check convergence
        SCOPED_TRACE("convergence with"
                     " fit_intercept: " + std::to_string(fit_intercept)
                         + ", model: " + dense_model->get_class_name()
                         + ", prox: " + prox->get_class_name());

        SDCA dense_sdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        dense_sdca.set_rand_max(rand_max);
        dense_sdca.set_model(dense_model);
        dense_sdca.set_prox(prox);

        auto objective_dense_sdca_20 = run_and_get_objective(dense_sdca, dense_model, prox, 20);
        auto objective_dense_sdca =
            run_and_get_objective(dense_sdca, dense_model, prox, n_epochs - 20);

        // Check it is converging
        EXPECT_LE(objective_dense_sdca - objective_dense_sdca_20, 1e-13);
        EXPECT_LE(objective_dense_sdca_20 - objective_dense_sdca, 0.3);

        SDCA dense_sdca_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        dense_sdca_batch.set_rand_max(rand_max);
        dense_sdca_batch.set_model(dense_model);
        dense_sdca_batch.set_prox(prox);

        auto objective_dense_sdca_batch =
            run_and_get_objective(dense_sdca_batch, dense_model, prox, n_epochs, 3);

        // Check it reaches the same objective
        EXPECT_LE(objective_dense_sdca_batch - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_dense_sdca_batch, 0.0001);

        AtomicSDCA dense_sdca_atomic(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 3);
        dense_sdca_atomic.set_rand_max(rand_max);
        dense_sdca_atomic.set_model(dense_atomic_model);
        dense_sdca_atomic.set_prox(atomic_prox);
        auto objective_dense_sdca_atomic =
            run_and_get_objective(dense_sdca_atomic, dense_model, prox, n_epochs);

        // Check it reaches the same objective
        EXPECT_LE(objective_dense_sdca_atomic - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_dense_sdca_atomic, 0.0001);

        AtomicSDCA dense_sdca_atomic_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 2);
        dense_sdca_atomic_batch.set_rand_max(rand_max);
        dense_sdca_atomic_batch.set_model(dense_atomic_model);
        dense_sdca_atomic_batch.set_prox(atomic_prox);

        auto objective_dense_sdca_atomic_batch = run_and_get_objective(
            dense_sdca_atomic_batch, dense_model, prox, n_epochs, 2);

        // Check it reaches the same objective
        EXPECT_LE(objective_dense_sdca_atomic_batch - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_dense_sdca_atomic_batch, 0.0001);

        /////////////////
        //    SPARSE   //
        /////////////////
        SDCA sparse_sdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sparse_sdca.set_rand_max(rand_max);
        sparse_sdca.set_model(sparse_model);
        sparse_sdca.set_prox(prox);
        auto objective_sparse_sdca =
            run_and_get_objective(sparse_sdca, sparse_model, prox, n_epochs, 3);
        // Check it reaches the same objective
        EXPECT_LE(objective_sparse_sdca - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_sparse_sdca, 0.0001);

        SDCA sparse_sdca_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sparse_sdca_batch.set_rand_max(rand_max);
        sparse_sdca_batch.set_model(sparse_model);
        sparse_sdca_batch.set_prox(prox);

        auto objective_sparse_sdca_batch =
            run_and_get_objective(sparse_sdca_batch, sparse_model, prox, n_epochs, 3);

        // Check it reaches the same objective
        EXPECT_LE(objective_sparse_sdca_batch - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_sparse_sdca_batch, 0.0001);

        AtomicSDCA sparse_sdca_atomic(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 3);
        sparse_sdca_atomic.set_rand_max(rand_max);
        sparse_sdca_atomic.set_model(sparse_atomic_model);
        sparse_sdca_atomic.set_prox(atomic_prox);
        auto objective_sparse_sdca_atomic =
            run_and_get_objective(sparse_sdca_atomic, sparse_model, prox, n_epochs);

        // Check it reaches the same objective
        EXPECT_LE(objective_sparse_sdca_atomic - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_sparse_sdca_atomic, 0.0001);

        AtomicSDCA sparse_sdca_atomic_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 2);
        sparse_sdca_atomic_batch.set_rand_max(rand_max);
        sparse_sdca_atomic_batch.set_model(sparse_atomic_model);
        sparse_sdca_atomic_batch.set_prox(atomic_prox);

        auto objective_sparse_sdca_atomic_batch = run_and_get_objective(
            sparse_sdca_atomic_batch, sparse_model, prox, n_epochs, 2);

        // Check it reaches the same objective
        EXPECT_LE(objective_sparse_sdca_atomic_batch - objective_dense_sdca, 0.0001);
        EXPECT_LE(objective_dense_sdca - objective_sparse_sdca_atomic_batch, 0.0001);
      }
    }
  }
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
