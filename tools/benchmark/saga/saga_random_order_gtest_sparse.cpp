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

// constexpr ulong n_samples = 196000;
constexpr ulong n_samples = 20000;

constexpr auto ALPHA = 1. / n_samples;
constexpr auto BETA  = 1e-10;
constexpr auto STRENGTH = ALPHA + BETA;
constexpr auto RATIO = BETA / STRENGTH;

int main(int argc, char *argv[]) {

  {
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
    auto features(tick_double_sparse2d_from_file(features_s));
    std::cout << "features.n_rows() "  << features->n_rows() << std::endl;
    auto labels(tick_double_array_from_file(labels_s));
    using milli = std::chrono::microseconds;
    {
      auto model = std::make_shared<ModelLogReg>(features, labels, false);
      Array<double> minimizer(model->get_n_coeffs());
      TSAGA<double, double> saga(n_samples, 0, RandType::unif, 0.00257480411965);
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      auto prox = std::make_shared<TProxElasticNet<double, double>>(STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
      saga.set_prox(prox);
      size_t total = 0;
      auto start = NOW;
      for (size_t j = 0; j < N_ITER; ++j) {
        auto start_iter = NOW;
        saga.solve();
        total += (NOW - start_iter);
        saga.get_minimizer(minimizer);
        double objective = model->loss(minimizer) + prox->value(minimizer, prox->get_start(), prox->get_end());
        std::cout << "LogReg : " << j << " : time : " << total << " : objective: " << objective << std::endl;
      }
      auto finish = NOW;
      std::cout << argv[0] << " with n_threads " << std::to_string(1) << " "
                << (finish - start) / 1e6
                << std::endl;
    }
  }

  return 0;
}