#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>
#include <lib/include/tick/prox/prox_l2sq.h>

#include "tick/linear_model/model_linreg.h"
#include "tick/prox/prox_zero.h"
#include "tick/prox/prox_l1.h"
#include "tick/solver/sdca.h"
#include "tick/solver/asdca.h"
#include "toy_dataset.ipp"

template <class T, class K>
double run_and_get_objective(TBaseSDCA<T, K> &sdca, std::shared_ptr<TModel<T> > model,
                             std::shared_ptr<TProx<T> > prox, int n_epochs, ulong batch_size=1) {
  const auto objective_prox = std::make_shared<TProxL2Sq<T>>(sdca.get_l_l2sq(), false);
  Array<T> out_iterate(model->get_n_coeffs());

  if (batch_size == 1) sdca.solve(n_epochs);
  else sdca.solve_batch(n_epochs, batch_size);

  sdca.get_iterate(out_iterate);
  return model->loss(out_iterate) + prox->value(out_iterate) + objective_prox->value(out_iterate);
};

TEST(SDCA, test_sdca_sparse) {
  SArrayDoublePtr labels_ptr = get_labels();

  for (auto is_sparse : std::vector<bool> {false, true}) {

    SBaseArrayDouble2dPtr features_ptr = is_sparse? get_sparse_features() : get_features();

    ulong n_samples = features_ptr->n_rows();
    const double l_l2sq = 0.1;

    // TESTED MODELS
    std::vector<ModelPtr> models;
    std::vector<ModelAtomicPtr> atomic_models;
    models.push_back(std::make_shared<ModelLinReg>(features_ptr, labels_ptr, false, 1));
    atomic_models.push_back(std::make_shared<ModelLinRegAtomic>(features_ptr, labels_ptr, false, 1));

    // TESTED PROXS
    std::vector<ProxPtr> proxs;
    std::vector<ProxAtomicPtr> atomic_proxs;
    proxs.push_back(std::make_shared<TProxZero<double>>(false));
    atomic_proxs.push_back(std::make_shared<TProxZero<double, std::atomic<double>>>(false));
    proxs.push_back(std::make_shared<TProxL1<double> >(0.8, false));
    atomic_proxs.push_back(std::make_shared<TProxL1<double, std::atomic<double>>>(0.8, false));

    for (int i = 0; i < models.size(); ++i) {
      for (int j = 0; j < proxs.size(); ++j) {
        auto model = models[i];
        auto atomic_model = atomic_models[i];
        auto prox = proxs[j];
        auto atomic_prox = atomic_proxs[j];

        SCOPED_TRACE(
            "is sparse: " + std::to_string(is_sparse) +
                ", model: " + model->get_class_name()
                + ", prox: " + prox->get_class_name());

        SDCA sdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sdca.set_rand_max(n_samples);
        sdca.set_model(model);
        sdca.set_prox(prox);

        auto objective_saga_30 = run_and_get_objective(sdca, model, prox, 30);
        auto objective_saga = run_and_get_objective(sdca, model, prox, 70);

        // Check it is converging
        EXPECT_LE(objective_saga - objective_saga_30, 0.);
        EXPECT_LE(objective_saga_30 - objective_saga, 0.1);

        SDCA sdca_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sdca.set_rand_max(n_samples);
        sdca.set_model(model);
        sdca.set_prox(prox);

        auto objective_saga_batch = run_and_get_objective(sdca, model, prox, 70, 3);

        // Check it reaches the same objective
        EXPECT_LE(objective_saga_batch - objective_saga, 0.0001);
        EXPECT_LE(objective_saga - objective_saga_batch, 0.0001);

        AtomicSDCA sdca_atomic(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 4);
        sdca_atomic.set_rand_max(n_samples);
        sdca_atomic.set_model(atomic_model);
        sdca_atomic.set_prox(atomic_prox);

        auto objective_saga_atomic = run_and_get_objective(sdca_atomic, model, prox, 100);

        // Check it reaches the same objective
        EXPECT_LE(objective_saga_atomic - objective_saga, 0.0001);
        EXPECT_LE(objective_saga - objective_saga_atomic, 0.0001);

        AtomicSDCA sdca_atomic_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 2);
        sdca_atomic_batch.set_rand_max(n_samples);
        sdca_atomic_batch.set_model(atomic_model);
        sdca_atomic_batch.set_prox(atomic_prox);

        auto objective_saga_atomic_batch = run_and_get_objective(
            sdca_atomic_batch, model, prox, 100, 2);

        // Check it reaches the same objective
        EXPECT_LE(objective_saga_atomic_batch - objective_saga, 0.0001);
        EXPECT_LE(objective_saga - objective_saga_atomic_batch, 0.0001);
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
