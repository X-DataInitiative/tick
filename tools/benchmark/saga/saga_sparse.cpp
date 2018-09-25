#include <vector>

#include "tick/solver/saga.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/prox/prox_elasticnet.h"

#include "shared_saga.ipp"

const constexpr int SEED = 42;

std::tuple<std::vector<double>, std::vector<double>> run_asaga_solver(
    SBaseArrayDouble2dPtr features, SArrayDoublePtr labels, ulong n_iter, int n_threads,
    int record_every, double strength, double ratio) {
  const auto n_samples = features->n_rows();

  auto model = std::make_shared<TModelLogReg<double> >(features, labels, false);
  TSAGA<double> saga(
      n_samples, 0,
      RandType::unif,
      1. / model->get_lip_max(),
      record_every,
      SEED
  );
  saga.set_rand_max(n_samples);
  saga.set_model(model);

  auto prox = std::make_shared<TProxElasticNet<double> >(
      strength, ratio, 0, model->get_n_coeffs(), 0);

  saga.set_prox(prox);
  saga.solve((int) n_iter); // single solve call as iterations happen within threads
  const auto &history = saga.get_time_history();

  const auto &iterates = saga.get_iterate_history();

  std::vector<double> objectives(iterates.size());
  for (int i = 0; i < iterates.size(); ++i) {
    objectives[i] = model->loss(*iterates[i]) + prox->value(*iterates[i]);
  }

  return std::make_tuple(history, objectives);
}

int main(int argc, char *argv[]) {
  submain(argc, argv, run_asaga_solver);
  return 0;
}

