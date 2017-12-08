// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_time_func.h"

class HawkesKernelTimeFuncTest : public ::testing::Test {
 protected:
  // we need to use a unique_ptr as HawkesKernelTimeFunc has no default constructor
  std::unique_ptr<HawkesKernelTimeFunc> hawkes_kernel_time_func;

  void SetUp() override {
    ArrayDouble t_axis {1, 2, 3, 4};
    ArrayDouble y_axis {1, 1, 3, 2};
    hawkes_kernel_time_func = std::unique_ptr<HawkesKernelTimeFunc>(
        new HawkesKernelTimeFunc(t_axis, y_axis));
  }
};

TEST_F(HawkesKernelTimeFuncTest, is_zero) {
  EXPECT_FALSE(hawkes_kernel_time_func->is_zero());
}

TEST_F(HawkesKernelTimeFuncTest, get_value) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(-3), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(0), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(1), 1);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(1.5), 1);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(3.5), 2.5);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(4), 2.);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_value(4.1), 0.);
}

TEST_F(HawkesKernelTimeFuncTest, get_future_max) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_future_max(-3, 0), 3);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_future_max(1, 0), 3);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_future_max(3.5, 2.5), 2.5);
  EXPECT_DOUBLE_EQ(hawkes_kernel_time_func->get_future_max(4.1, 0), 0);
}

TEST_F(HawkesKernelTimeFuncTest, get_norm) {
  EXPECT_NEAR(hawkes_kernel_time_func->get_norm(), 5.5, 1e-3);
}

TEST_F(HawkesKernelTimeFuncTest, get_support) {
  EXPECT_GE(hawkes_kernel_time_func->get_support(), 4);
}


#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
