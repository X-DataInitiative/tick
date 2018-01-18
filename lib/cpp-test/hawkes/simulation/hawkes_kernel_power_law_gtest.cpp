// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_power_law.h"

class HawkesKernelPowerLawTest : public ::testing::Test {
 protected:
  double multiplier;
  double cutoff;
  double exponent;
  HawkesKernelPowerLaw hawkes_kernel_power_law;
  ArrayDouble timestamps;

  std::array<double, 6> test_times{{1., 2., 3.5, 5., 8., 100.}};

  HawkesKernelPowerLawTest() : hawkes_kernel_power_law(0, 0, 0) {};

  void SetUp() override {
    multiplier = 0.1;
    cutoff = 0.01;
    exponent = 1.2;
    hawkes_kernel_power_law = HawkesKernelPowerLaw(multiplier, cutoff, exponent);

    timestamps = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
  }
};

TEST_F(HawkesKernelPowerLawTest, is_zero) {
  EXPECT_FALSE(hawkes_kernel_power_law.is_zero());
}

TEST_F(HawkesKernelPowerLawTest, get_value) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_power_law.get_value(-3), 0);

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_power_law.get_value(test_time),
                     multiplier * pow(test_time + cutoff, -exponent));
  }
}

TEST_F(HawkesKernelPowerLawTest, get_norm) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_power_law.get_norm(),
                   1.2096372793483503);
}

TEST_F(HawkesKernelPowerLawTest, invalid_constructor_parameters) {
  EXPECT_THROW(HawkesKernelPowerLaw(multiplier, cutoff, exponent, -1, -1), std::invalid_argument);
  EXPECT_THROW(HawkesKernelPowerLaw(multiplier, cutoff, exponent, -1, 0), std::invalid_argument);
  EXPECT_THROW(HawkesKernelPowerLaw(multiplier, cutoff, exponent, 0, -1), std::invalid_argument);
  EXPECT_THROW(HawkesKernelPowerLaw(multiplier, cutoff, exponent, 0, 0), std::invalid_argument);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
