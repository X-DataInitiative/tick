#include <ctime>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <sys/time.h>

#include "hawkes.h"
#include "hawkes_sdca_loglik_kern.h"

int main() {

  const int seed = 1933;
  Rand rand(seed);

  unsigned int n_nodes = 4;
  const double decay = 3.;

  Hawkes hawkes(n_nodes);

  for (unsigned int i = 0; i < n_nodes; ++i) {
    hawkes.set_baseline(i, 0.04);
    for (unsigned int j = 0; j < n_nodes; ++j) {
      double intensity = rand.uniform(0, 0.5 / (n_nodes * n_nodes));
      std::shared_ptr<HawkesKernel> kernel = std::make_shared<HawkesKernelExp>(intensity, decay);
      hawkes.set_kernel(i, j, kernel);
    }
  }

  ulong n_points = 1000000;
  hawkes.simulate(n_points);

  for (int k = 0; k < n_nodes; ++k) {
    hawkes.timestamps[k]->print();
  }
  SArrayDoublePtrList2D timestamps_list(1);
  timestamps_list[0] = SArrayDoublePtrList1D(n_nodes);
  double end_time = 0;
  for (int i = 0; i < n_nodes; ++i) {
    timestamps_list[0][i] = hawkes.timestamps[i];
    end_time = std::max(end_time, hawkes.timestamps[i]->last());
  }
  VArrayDoublePtr end_times = VArrayDouble::new_ptr(0);
  end_times->append1(end_time);

  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  double l_l2sq = 0.1;

  ArrayDouble decays {decay};

  HawkesSDCALoglikKern hawkes_dual(decays, l_l2sq, 4);
  hawkes_dual.set_data(timestamps_list, end_times);
  for (int t = 0; t < 50; ++t) {
    hawkes_dual.solve();
  }

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  double dif = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() * 1e-9;
  printf("Elasped time is %lf seconds.\n", dif);

  hawkes_dual.get_iterate()->print();

  return 0;
}