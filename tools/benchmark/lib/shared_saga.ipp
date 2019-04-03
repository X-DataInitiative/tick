#include <vector>

#include "tick/array/serializer.h"

double Variance(std::vector<double> samples) {
  int size = samples.size();

  double variance = 0;
  double t = samples[0];
  for (size_t i = 1; i < size; i++) {
    t += samples[i];
    double diff = ((i + 1) * samples[i]) - t;
    variance += (diff * diff) / ((i + 1.0) * i);
  }

  return variance / (size - 1);
}

double StandardDeviation(std::vector<double> samples) {
  return std::sqrt(Variance(samples));
}

void submain(
    int argc, char *argv[],
    std::function<std::tuple<std::vector<double>, std::vector<double>>
                      (SBaseArrayDouble2dPtr,
                       SArrayDoublePtr,
                       ulong,
                       int,
                       int,
                       double,
                       double)> run_solver) {

  std::string file_path = __FILE__;
  std::string dir_path = file_path.substr(0, file_path.rfind("/"));

  std::string dataset;
  if (argc <= 1) dataset = "adult";
  else dataset = std::string(argv[1]);

  std::string features_s(dir_path + "/../data/" + dataset + ".features.cereal");
  std::string labels_s(dir_path + "/../data/" + dataset + ".labels.cereal");

  std::vector<int> range;
  if (argc < 2) range.push_back(1);
  else range.push_back(std::stoi(argv[2]));

  ulong n_iter;
  if (argc <= 3) n_iter = 25;
  else n_iter = std::stoul(argv[3]);

  int record_every;
  if (argc <= 4) record_every = 4;
  else record_every = std::stoi(argv[4]);

  bool verbose;
  if (argc <= 5) verbose = false;
  else verbose = (bool) std::stoi(argv[5]);

  auto features(tick_double_sparse2d_from_file(features_s));

  if (verbose) {
    std::cout << "features.n_rows() " << features->n_rows() << std::endl;
    std::cout << "features.size_sparse() " << features->size_sparse() << std::endl;
  }
  auto labels(tick_double_array_from_file(labels_s));

  const auto n_samples = features->n_rows();
  const auto ALPHA = 100. / n_samples;
  const auto BETA = 1e-10;
  const auto STRENGTH = ALPHA + BETA;
  const auto RATIO = BETA / STRENGTH;

  for (auto n_threads : range) {

    double min_objective;
    std::vector<double> samples;
    for (size_t tries = 0; tries < 5; ++tries) {

      std::vector<double> history, objective;
      std::tie(history, objective) = run_solver(features, labels, n_iter, n_threads, record_every,
                                                STRENGTH, RATIO);

      min_objective = *std::min_element(std::begin(objective), std::end(objective));

      if (verbose) {
        for (ulong i = 1; i < objective.size(); i++) {
          auto log_dist = objective[i] == min_objective ? 0 : log10(objective[i] - min_objective);
          std::cout << n_threads << " " << i * record_every << " " << history[i] << " "
                    << "1e" << log_dist << std::endl;
        }
      }

      samples.push_back(history.back());
    }

    double min = *std::min_element(samples.begin(), samples.end());
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double ci_half_width = 1.96 * StandardDeviation(samples) / std::sqrt(samples.size());

    std::cout << "\n"
              << "min_objective : " << min_objective << "\n"
              << "Min: " << min
              << "\n Mean: " << mean << " +/- " << ci_half_width
              << std::endl;
  }
}