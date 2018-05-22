#include <chrono>

#include "tick/array/serializer.h"
#include "tick/random/test_rand.h"
#include "tick/solver/saga.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_linreg.h"

#include "tick/prox/prox_zero.h"
#include "tick/prox/prox_elasticnet.h"

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/os.hpp"
#endif

#define NOW std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

const constexpr size_t SEED       = 1933;
const constexpr size_t N_ITER     = 200;

constexpr ulong n_samples = 196000;
// constexpr ulong n_samples = 20000;

constexpr auto ALPHA = 1. / n_samples;
constexpr auto BETA  = 1e-10;
constexpr auto STRENGTH = ALPHA + BETA;
constexpr auto RATIO = BETA / STRENGTH;

int main(int argc, char *argv[]) {

  std::string features_s("../url.features.cereal");
  std::string labels_s("../url.labels.cereal");
#ifdef _MKN_WITH_MKN_KUL_
  kul::File features_f(features_s);
  kul::File labels_f(labels_s);
  if(!features_f){
    features_s = "url.features.cereal";
    labels_s = "url.labels.cereal";
  }
#endif

  std::vector<int> range;//{ 12}; //, 4, 6, 8, 10, 12, 14, 16 };
  if(argc == 1) return 0;
  range.push_back(std::stol(argv[1]));

  for(auto n_threads : range){

    auto features(tick_double_sparse2d_from_file(features_s));

    std::cout << "features.indices() "  << features->indices() << std::endl;
    std::cout << "features.indices()[-1] "  << features->indices()[features->size_sparse() - 1] << std::endl;

    std::cout << "features.n_rows() "  << features->n_rows() << std::endl;
    std::cout << "features.size_sparse() "  << features->size_sparse() << std::endl;
    auto labels(tick_double_array_from_file(labels_s));
    // using milli = std::chrono::microseconds;
    {
      auto model = std::make_shared<TModelLogReg<double, std::atomic<double> > >(features, labels, false);
      Array<std::atomic<double>> minimizer(model->get_n_coeffs());
      AtomicSAGA<double> saga(
        n_samples, N_ITER / n_threads,
        0,
        RandType::unif,
        0.00257480411965, //1e-3,
        -1,
        SAGA_VarianceReductionMethod::Last,
        n_threads
      );
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      auto prox = std::make_shared<TProxElasticNet<double, std::atomic<double> >>(STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
      saga.set_prox(prox);
      // size_t total = 0;
      // auto start = NOW;
      // for (int j = 0; j < N_ITER; ++j) {
        // auto start_iter = NOW;
        saga.solve();
        // total += (NOW - start_iter);
        // saga.get_atomic_minimizer(minimizer);
        // double objective = model->loss(minimizer) + prox->value(minimizer, prox->get_start(), prox->get_end());
      // }
      const auto &history = saga.get_history();
      const auto &objective = saga.get_objective();
      for(size_t i = 0; i < N_ITER / n_threads; i++)
        std::cout << n_threads << " " << (i * n_threads) << " " << history[i] << " " << objective[i] << std::endl;
      // auto finish = NOW;
      // std::cout << argv[0] << " with n_threads " << std::to_string(n_threads) << " "
      //           << (finish - start) / 1e6
      //           << std::endl;
    }
    // {
    //   auto model = std::make_shared<TModelLinReg<double, std::atomic<double> > >(features, labels, false);
    //   Array<std::atomic<double>> minimizer(model->get_n_coeffs());
    //   AtomicSAGA<double> saga(
    //     n_samples,
    //     0,
    //     RandType::unif,
    //     1e-3,
    //     -1,
    //     SAGA_VarianceReductionMethod::Last,
    //     n_threads
    //   );
    //   saga.set_rand_max(n_samples);
    //   saga.set_model(model);
    //   auto prox = std::make_shared<TProxElasticNet<double, std::atomic<double> >>(STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
    //   saga.set_prox(prox);
    //   size_t total = 0;
    //   auto start = NOW;
    //   for (int j = 0; j < N_ITER; ++j) {
    //     auto start_iter = NOW;
    //     saga.solve();
    //     total += NOW - start_iter;
    //     saga.get_atomic_minimizer(minimizer);
    //     double objective = model->loss(minimizer) + prox->value(minimizer, prox->get_start(), prox->get_end());
    //     std::cout << "LinReg : " << j << " : time : " << total << " : objective: " << objective << std::endl;
    //   }
    //   auto finish = NOW;
    //   std::cout << argv[0] << " with n_threads " << std::to_string(n_threads) << " "
    //             << (finish - start) / 1e6
    //             << std::endl;
    // }
  }

  return 0;
}
