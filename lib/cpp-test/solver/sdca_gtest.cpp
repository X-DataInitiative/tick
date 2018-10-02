#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include <cereal/archives/json.hpp>

#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_poisreg.h"

#include "tick/prox/prox_zero.h"
#include "tick/prox/prox_l1.h"
#include "lib/include/tick/prox/prox_l2sq.h"

#include "tick/solver/sdca.h"

#include "toy_dataset.ipp"

template <class T, class K>
double run_and_get_objective(TBaseSDCA<T, K> &sdca, std::shared_ptr<TModel<T> > model,
                             std::shared_ptr<TProx<T> > prox, int n_epochs, ulong batch_size=1) {
  const auto objective_prox = std::make_shared<TProxL2Sq<T>>(sdca.get_l_l2sq(), false);
  Array<T> out_iterate(model->get_n_coeffs());

  if (batch_size == 1) sdca.solve(n_epochs);
  else sdca.solve_batch(n_epochs, batch_size);

  sdca.get_iterate(out_iterate);
  out_iterate.print();
  return model->loss(out_iterate) + prox->value(out_iterate) + objective_prox->value(out_iterate);
};

TEST(SDCA, test_sdca_sparse) {
  for (auto is_sparse : std::vector<bool> {false, true}) {

    SBaseArrayDouble2dPtr features_ptr = is_sparse? get_sparse_features() : get_features();

    ulong n_samples = features_ptr->n_rows();
    const double l_l2sq = 0.1;
    const ulong n_epochs = 200;

    // TESTED MODELS
    std::vector<ModelPtr> models;
    std::vector<ModelAtomicPtr> atomic_models;
//    models.push_back(std::make_shared<ModelLinReg>(
//        features_ptr, get_linreg_labels(), false, 1));
//    atomic_models.push_back(std::make_shared<ModelLinRegAtomic>(
//        features_ptr, get_linreg_labels(), false, 1));
//    models.push_back(std::make_shared<ModelLogReg>(
//        features_ptr, get_logreg_labels(), false, 1));
//    atomic_models.push_back(std::make_shared<ModelLogRegAtomic>(
//        features_ptr, get_logreg_labels(), false, 1));
//    models.push_back(std::make_shared<ModelPoisReg>(
//        features_ptr, get_poisson_labels(), LinkType::exponential, false, 1));
//    atomic_models.push_back(std::make_shared<ModelPoisRegAtomic>(
//        features_ptr, get_poisson_labels(), LinkType::exponential, false, 1));
    models.push_back(std::make_shared<ModelPoisReg>(
        features_ptr, get_poisson_labels(), LinkType::identity, false, 1));
    atomic_models.push_back(std::make_shared<ModelPoisRegAtomic>(
        features_ptr, get_poisson_labels(), LinkType::identity, false, 1));


    // TESTED PROXS
    std::vector<ProxPtr> proxs;
    std::vector<ProxAtomicPtr> atomic_proxs;
    proxs.push_back(std::make_shared<TProxZero<double>>(false));
    atomic_proxs.push_back(std::make_shared<TProxZero<double, std::atomic<double>>>(false));
    proxs.push_back(std::make_shared<TProxL1<double> >(0.2, false));
    atomic_proxs.push_back(std::make_shared<TProxL1<double, std::atomic<double>>>(0.2, false));

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

        ulong rand_max = model->get_sdca_index_map() == nullptr?
          n_samples: model->get_sdca_index_map()->size();

        SDCA sdca(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sdca.set_rand_max(rand_max);
        sdca.set_model(model);
        sdca.set_prox(prox);

        auto objective_sdca_20 = run_and_get_objective(sdca, model, prox, 20);
        auto objective_sdca = run_and_get_objective(sdca, model, prox, n_epochs - 20);

        // Check it is converging
        EXPECT_LE(objective_sdca - objective_sdca_20, 1e-13);
        EXPECT_LE(objective_sdca_20 - objective_sdca, 0.1);

        SDCA sdca_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309);
        sdca_batch.set_rand_max(rand_max);
        sdca_batch.set_model(model);
        sdca_batch.set_prox(prox);

        auto objective_sdca_batch = run_and_get_objective(sdca_batch, model, prox, n_epochs, 3);

        // Check it reaches the same objective
        EXPECT_LE(objective_sdca_batch - objective_sdca, 0.0001);
        EXPECT_LE(objective_sdca - objective_sdca_batch, 0.0001);

        AtomicSDCA sdca_atomic(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 3);
        sdca_atomic.set_rand_max(rand_max);
        sdca_atomic.set_model(atomic_model);
        sdca_atomic.set_prox(atomic_prox);
        auto objective_sdca_atomic = run_and_get_objective(sdca_atomic, model, prox, n_epochs);

        // Check it reaches the same objective
        EXPECT_LE(objective_sdca_atomic - objective_sdca, 0.0001);
        EXPECT_LE(objective_sdca - objective_sdca_atomic, 0.0001);

        AtomicSDCA sdca_atomic_batch(l_l2sq, n_samples, 0, RandType::unif, 1, 1309, 2);
        sdca_atomic_batch.set_rand_max(rand_max);
        sdca_atomic_batch.set_model(atomic_model);
        sdca_atomic_batch.set_prox(atomic_prox);

        auto objective_sdca_atomic_batch = run_and_get_objective(
            sdca_atomic_batch, model, prox, n_epochs, 2);

        // Check it reaches the same objective
        EXPECT_LE(objective_sdca_atomic_batch - objective_sdca, 0.0001);
        EXPECT_LE(objective_sdca - objective_sdca_atomic_batch, 0.0001);
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
