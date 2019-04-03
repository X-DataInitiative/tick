#include <vector>

#include "tick/solver/asaga.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/prox/prox_elasticnet.h"

#include "shared_saga.ipp"

//
// Benchmark asaga performances
// The command lines arguments are the following
// dataset : a sparse dataset (for example generated with benchmark_utils.py
// n_threads : the number of threads to be used
// n_iter : the number of passes on the data
// record_every : how often metrics are computed
// verbose : verbose results on the fly if true, only summary if false
//
// Example
// First get the data ready
// python -c "from benchmark_util import save_url_dataset_for_cpp_benchmarks; save_url_dataset_for_cpp_benchmarks(5)"
// Then run asaga on url dataset with 5 days, 1 thread, 25 iterations, record every 5 and no verbose
// ./tick_asaga_sparse url.5 1 25 5 0
//

const constexpr int SEED = 42;

std::tuple<std::vector<double>, std::vector<double>> run_asaga_solver(
    SBaseArrayDouble2dPtr features, SArrayDoublePtr labels, ulong n_iter, int n_threads,
    int record_every, double strength, double ratio) {
  const auto n_samples = features->n_rows();

  auto model = std::make_shared<TModelLogReg<double> >(features, labels, false);
  AtomicSAGA<double> asaga(
      n_samples, 0,
      RandType::unif,
      1. / model->get_lip_max(),
      record_every,
      SEED,
      n_threads
  );
  asaga.set_rand_max(n_samples);
  asaga.set_model(model);

  auto prox = std::make_shared<TProxElasticNet<double> >(
      strength, ratio, 0, model->get_n_coeffs(), 0);

  asaga.set_prox(prox);
  asaga.solve((int) n_iter); // single solve call as iterations happen within threads
  const auto &history = asaga.get_time_history();

  const auto &iterates = asaga.get_iterate_history();

  std::vector<double> objectives(iterates.size());
  for (size_t i = 0; i < iterates.size(); ++i) {
    objectives[i] = model->loss(*iterates[i]) + prox->value(*iterates[i]);
  }

  return std::make_tuple(history, objectives);
}

int main(int argc, char *argv[]) {
  std::cout << "ASAGA" << std::endl;
  submain(argc, argv, run_asaga_solver);
  return 0;
}

