// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/simulation/hawkes.h"

TEST(SimuHawkesTest, constant_baseline) {
  Hawkes hawkes(1);
  hawkes.set_baseline(0, 5.);
  const double simu_time = 10;
  hawkes.simulate(simu_time);
  EXPECT_EQ(hawkes.get_time(), simu_time);
  EXPECT_GT(hawkes.get_n_total_jumps(), 1);
}

TEST(SimuHawkesTest, tuple_baseline) {
  ArrayDouble t_values {1., 2., 4., 5.3};
  ArrayDouble y_values {1., 3., 2., 0.};
  Hawkes hawkes(1);
  hawkes.set_baseline(0, t_values, y_values);
  const double simu_time = 100;
  hawkes.simulate(simu_time);
  EXPECT_EQ(hawkes.get_time(), simu_time);
  EXPECT_GT(hawkes.get_n_total_jumps(), 1);
  // Check that intensity TimeFunction is cycled
  EXPECT_GT(hawkes.timestamps[0]->last(), 10);
}
