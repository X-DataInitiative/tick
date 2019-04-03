#include <vector>

#include "tick/solver/asaga.h"

#include "shared_saga.ipp"

//
// Benchmark Hawkes least squares weights computing performances
// The command lines arguments are the following
// num_nodes : number of nodes in the Hawkes process
// num_events_per_node : number of events simulated in each node
// num_baselines : number of baselines used per node (non constant baselines)
// num_decays : number of decays used in the sum of exponential kernels
// num_iterations : number of iterations of the same experiment
// num_threads : number of threads used
//
// Example
// To run hawkes with 5 nodes, 100 events per node, 1 baseline 3 decays, 100 iterations on 3 threads
// ./tick_hawkes_least_squares_weights 5 100 1 3 100 3
//

#include "tick/hawkes/model/model_hawkes_sumexpkern_leastsq_single.h"

SArrayDoublePtrList1D generate_data(ulong num_nodes, ulong num_events_per_node) {

  std::mt19937 gen(1337);
  std::uniform_real_distribution<> unif{};

  SArrayDoublePtrList1D timestamps;
  timestamps = SArrayDoublePtrList1D(0);

  // Test will fail if process array is not sorted
  for (ulong j = 0; j < num_nodes; ++j) {
    double t = 0.0;

    ArrayDouble timestamps_0(num_events_per_node);
    for (ulong k = 0; k < timestamps_0.size(); ++k) {
      t += unif(gen);

      timestamps_0[k] = t;
    }

    timestamps.push_back(timestamps_0.as_sarray_ptr());
  }

  return timestamps;
}

ArrayDouble generate_decays(ulong num_decays) {
  ArrayDouble decays(num_decays);

  for (ulong j = 0; j < decays.size(); ++j) {
    decays[j] = j + 1.0;
  }

  return decays;
}

int main(int nargs, char **args) {

  ulong num_nodes = 40;
  if (nargs > 1) num_nodes = std::stoul(args[1]);

  ulong num_events_per_node = 30;
  if (nargs > 2) num_events_per_node = std::stoul(args[2]);

  ulong num_baselines = 1;
  if (nargs > 3) num_baselines = std::stoul(args[3]);

  ulong num_decays = 10;
  if (nargs > 4) num_decays = std::stoul(args[4]);

  unsigned int num_iterations = 100;
  if (nargs > 5) num_iterations = std::stoul(args[5]);

  unsigned int num_threads = 1;
  if (nargs > 6) num_threads = std::stoul(args[6]);

  const ulong num_runs = 5;

  auto decays = generate_decays(num_decays);

  for (ulong run_i = 0; run_i < num_runs; ++run_i) {
    auto timestamps = generate_data(num_nodes, num_events_per_node);

    double end_time = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
      end_time = std::max(end_time, timestamps[i]->last());
    }

    ModelHawkesSumExpKernLeastSqSingle model(decays, num_baselines, end_time, num_threads);
    model.set_data(timestamps, end_time);

    ArrayDouble coeffs = ArrayDouble(num_nodes * num_baselines +
        num_nodes * num_nodes * decays.size());

    const auto start = std::chrono::system_clock::now();
    for (ulong i = 0; i < num_iterations; ++i) {
      model.compute_weights();
    }
    const auto end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << elapsed_seconds.count() << '\t' << num_iterations << '\t'
              << num_threads << '\t' << num_nodes << '\t' << args[0] << '\t'
              << std::endl;
  }
}