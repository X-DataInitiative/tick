#include <chrono>

#include "tick/array/serializer.h"
#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/random/test_rand.h"
#include "tick/solver/saga.h"

#include "tick/prox/prox_elasticnet.h"
#include "tick/prox/prox_zero.h"

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/os.hpp"
#endif

#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()

const constexpr size_t SEED = 1933;
const constexpr size_t N_ITER = 25;

double Variance(std::vector<double> samples) {
  int size = samples.size();

  double variance = 0;
  double t = samples[0];
  for (int i = 1; i < size; i++) {
    t += samples[i];
    double diff = ((i + 1) * samples[i]) - t;
    variance += (diff * diff) / ((i + 1.0) * i);
  }

  return variance / (size - 1);
}

double StandardDeviation(std::vector<double> samples) {
  return std::sqrt(Variance(samples));
}

int main(int argc, char *argv[]) {
  std::string features_s("../url.features.cereal");
  std::string labels_s("../url.labels.cereal");
#ifdef _MKN_WITH_MKN_KUL_
  kul::File features_f(features_s);
  kul::File labels_f(labels_s);
  if (!features_f) {
    features_s = "url.features.cereal";
    labels_s = "url.labels.cereal";
  }
#endif
  std::vector<int> range;  //{ 12}; //, 4, 6, 8, 10, 12, 14, 16 };
  if (argc == 1) return 0;
  range.push_back(std::stol(argv[1]));

  auto features(tick_double_sparse2d_from_file(features_s));
  auto labels(tick_double_array_from_file(labels_s));
  ulong n_samples = features->n_rows();
  const auto ALPHA = 1.0 / n_samples;
  const auto BETA = 1e-10;
  const auto STRENGTH = ALPHA + BETA;
  const auto RATIO = BETA / STRENGTH;

  auto model =
      std::make_shared<TModelLogReg<double, double>>(features, labels, false);
  auto prox = std::make_shared<TProxElasticNet<double, double>>(
      STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
  for (auto n_threads : range) {
    std::vector<double> samples;
    for (int tries = 0; tries < 5; ++tries) {
      Array<std::atomic<double>> minimizer(model->get_n_coeffs());
      AtomicSAGA<double> saga(n_samples, N_ITER, 0, RandType::unif,
                              0.00257480411965,  // 1e-3,
                              -1, n_threads);
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      saga.set_prox(prox);
      auto start = NOW;
      saga.solve();
      auto finish = NOW - start;
      const auto &objective = saga.get_computed_objective();
      samples.push_back(finish);
      auto rel =
          (objective[objective.size() - 2] - objective[objective.size() - 1]) /
          objective[objective.size() - 1];
      std::cout << "LAST " << objective[objective.size() - 1] << std::endl;
      std::cout << "Relative " << rel << std::endl;
    }
    double min = *std::min_element(samples.begin(), samples.end());
    double mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double ci_half_width =
        1.96 * StandardDeviation(samples) / std::sqrt(samples.size());

    std::cout << "\n"
              << "Min: " << min << "\n Mean: " << mean << " +/- "
              << ci_half_width << std::endl;
  }

  return 0;
}
