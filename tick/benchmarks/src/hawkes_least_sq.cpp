

#include "hawkes_fixed_sumexpkern_leastsq.h"

#include "random"

int main(int nargs, char** args) {
  ulong num_threads = 1;

  if (nargs > 1)
    num_threads = std::stoul(args[1]);

  const ulong num_iterations = 100;
  const ulong num_events_per_node = 1000;
  const ulong num_nodes = 20;
  const ulong num_decays = 10;
  const ulong num_baselines = 1;

  const ulong num_runs = 1;

  std::mt19937 gen(1337);
  std::uniform_real_distribution<> unif{};

  ArrayDouble decays(num_decays);

  SArrayDoublePtrList1D timestamps;
  timestamps = SArrayDoublePtrList1D(0);

  // Test will fail if process array is not sorted
  double end_time = 0.0;
  for (ulong j = 0; j < decays.size(); ++j) {
    decays[j] = j + 1.0;

    double t = 0.0;

    ArrayDouble timestamps_0(num_events_per_node);
    for (ulong k = 0; k < timestamps_0.size(); ++k) {
      t += unif(gen);

      timestamps_0[k] = t;
    }

    timestamps.push_back(timestamps_0.as_sarray_ptr());

    end_time = std::max(end_time, t);
  }

  for (ulong run_i = 0; run_i < num_runs; ++run_i) {


    ModelHawkesFixedSumExpKernLeastSq model(decays, num_baselines, end_time, num_threads);
    model.set_data(timestamps, end_time);


    ArrayDouble coeffs = ArrayDouble(num_nodes * num_baselines + num_nodes * num_nodes * decays.size());

    const auto start = std::chrono::system_clock::now();
    for (ulong i = 0; i < num_iterations; ++i) {
      model.compute_weights();
//      model.loss_and_grad(coeffs, coeffs);
    }
    const auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << elapsed_seconds.count() * 1000.0 << '\t'
              << num_iterations << '\t'
              << num_threads << '\t'
              << num_nodes << '\t'
              << args[0] << '\t'
              << std::endl;
  }
}