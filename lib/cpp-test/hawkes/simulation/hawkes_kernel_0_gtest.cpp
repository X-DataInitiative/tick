// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_0.h"

class HawkesKernel0Test : public ::testing::Test {
 protected:
  HawkesKernel0 hawkes_kernel_time_func;
  ArrayDouble timestamps;

  std::array<double, 6> test_times{{1., 2., 3.5, 5., 8., 100.}};

  void SetUp() override {
    timestamps = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
  }
};

TEST_F(HawkesKernel0Test, is_zero) {
  EXPECT_TRUE(hawkes_kernel_time_func.is_zero());
}

TEST_F(HawkesKernel0Test, get_value) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_value(-3), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_value(0), 0);
  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_value(test_time), 0);
  }
}

TEST_F(HawkesKernel0Test, get_future_max) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_future_max(-3, 0), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_future_max(0, 0), 0);
  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_future_max(test_time, 0), 0);
  }
}

TEST_F(HawkesKernel0Test, get_norm) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func.get_norm(), 0);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
