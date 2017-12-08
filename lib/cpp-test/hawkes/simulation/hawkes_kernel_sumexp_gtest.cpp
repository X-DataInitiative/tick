// License: BSD 3 clause


#include <gtest/gtest.h>
#include "tick/base/base.h"
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_sum_exp.h"

double compute_sumexpkernel_get_value(ArrayDouble &decays,
                                      ArrayDouble &intensities,
                                      double test_time) {
  double kernel_sum{0.};
  for (ulong i = 0; i < decays.size(); ++i) {
    kernel_sum += intensities[i] * decays[i] * exp(-decays[i] * test_time);
  }
  return kernel_sum;
}

double compute_sumexpkernel_convolution(ArrayDouble &decays,
                                        ArrayDouble &intensities,
                                        ArrayDouble &timestamps,
                                        double time) {
  double kernel_sum{0.};
  for (ulong i = 0; i < timestamps.size(); ++i) {
    double t_i = timestamps[i];
    if (time >= t_i) {
      kernel_sum += compute_sumexpkernel_get_value(decays, intensities, time - t_i);
    }
  }
  return kernel_sum;
}

double compute_sumexpkernel_convolution_bound(ArrayDouble &decays,
                                              ArrayDouble &intensities,
                                              ArrayDouble &timestamps,
                                              double time) {
  double kernel_sum{0.};
  for (ulong i = 0; i < timestamps.size(); ++i) {
    double t_i = timestamps[i];
    if (time >= t_i) {
      kernel_sum += compute_sumexpkernel_get_value(decays, intensities, time - t_i);
    }
  }
  return kernel_sum;
}

class HawkesKernelSumExpTest : public ::testing::Test {
 protected:
  ArrayDouble intensities;
  ArrayDouble decays;

  // we need to use a unique_ptr as HawkesKernelSumExp has no default constructor
  std::unique_ptr<HawkesKernelSumExp> hawkes_kernel_sum_exp;
  ArrayDouble timestamps;

  // Test will fail if test_times array is not sorted
  std::array<double, 8> test_times{{0.93, 1., 2., 2.32, 3.5, 5., 8., 100.}};
  
  void SetUp() override {
    intensities = ArrayDouble {0.15, 0.45, 1.3};
    decays = ArrayDouble {0.75, 4.2, 2.3};
    hawkes_kernel_sum_exp = std::unique_ptr<HawkesKernelSumExp>(
        new HawkesKernelSumExp(intensities, decays));

    // Test will fail if process array is not sorted
    timestamps = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
  }
};

TEST_F(HawkesKernelSumExpTest, is_zero) {
  EXPECT_FALSE(hawkes_kernel_sum_exp->is_zero());
}

TEST_F(HawkesKernelSumExpTest, constructor_negative_decays_error) {
  decays[1] = -1;
  ASSERT_THROW(HawkesKernelSumExp(intensities, decays), std::invalid_argument);
}

TEST_F(HawkesKernelSumExpTest, constructor_wrong_size_error) {
  decays = ArrayDouble {1, 0.4};
  ASSERT_THROW(HawkesKernelSumExp(intensities, decays), std::invalid_argument);
}

TEST_F(HawkesKernelSumExpTest, constructor_zero_length_error) {
  decays = ArrayDouble(0);
  intensities = ArrayDouble(0);
  ASSERT_THROW(HawkesKernelSumExp(intensities, decays), std::invalid_argument);
}

TEST_F(HawkesKernelSumExpTest, get_value) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_value(-3), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_value(0), intensities.dot(decays));
  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_value(test_time),
                     compute_sumexpkernel_get_value(decays, intensities, test_time));
  }
}

TEST_F(HawkesKernelSumExpTest, get_future_max) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_future_max(-3, 0), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_future_max(0, intensities.dot(decays)),
                                                        intensities.dot(decays));
  for (double test_time : test_times) {
    double value_at_test_time = compute_sumexpkernel_get_value(decays, intensities, test_time);
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_future_max(test_time, value_at_test_time),
                     value_at_test_time);
  }
}

TEST_F(HawkesKernelSumExpTest, get_norm) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_norm(), intensities.sum());
}

TEST_F(HawkesKernelSumExpTest, get_convolution_value) {
  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(time0 - 0.1, timestamps, nullptr), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(time0, timestamps, nullptr),
                   intensities.dot(decays));

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(test_time, timestamps, nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, timestamps, test_time));
  }
}


// This edge case happen when we are simulating Hawkes processes
TEST_F(HawkesKernelSumExpTest, get_convolution_value_while_appending_array) {
  VArrayDouble v_timestamps(0);

  for (int k = 0; k < test_times.size() - 1; ++k) {
    double t_k = test_times[k];
    double t_k_next = test_times[k + 1];
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(t_k, v_timestamps, nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, v_timestamps, t_k));
    v_timestamps.append1(t_k);
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(t_k, v_timestamps, nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, v_timestamps, t_k));
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(t_k_next, v_timestamps, nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, v_timestamps, t_k_next));
    break;
  }
}

TEST_F(HawkesKernelSumExpTest, get_convolution_older_time_error) {
  double last_time = 4.;
  hawkes_kernel_sum_exp->get_convolution(last_time, timestamps, nullptr);

  double old_time = 2.;
  ASSERT_THROW(hawkes_kernel_sum_exp->get_convolution(old_time, timestamps, nullptr),
               std::runtime_error);
}

TEST_F(HawkesKernelSumExpTest, get_convolution_older_time_rewind) {
  double last_time = 4.;
  hawkes_kernel_sum_exp->get_convolution(last_time, timestamps, nullptr);
  hawkes_kernel_sum_exp->rewind();

  double old_time = 2.;
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(old_time, timestamps, nullptr),
                   compute_sumexpkernel_convolution(decays, intensities, timestamps, old_time));
}

TEST_F(HawkesKernelSumExpTest, get_convolution_other_timestamps_rewind) {
  double test_time = 4.;
  hawkes_kernel_sum_exp->get_convolution(test_time, timestamps, nullptr);
  hawkes_kernel_sum_exp->rewind();

  ArrayDouble other_timestamps {0.23, 1.34, 2.17};
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp->get_convolution(test_time, other_timestamps, nullptr),
                   compute_sumexpkernel_convolution(decays, intensities, other_timestamps, test_time));
}

TEST_F(HawkesKernelSumExpTest, get_convolution_bound) {
  double time0 = timestamps[0];

  double bound_exp_kernel{-1.};

  hawkes_kernel_sum_exp->get_convolution(time0 - 0.1, timestamps, &bound_exp_kernel);
  EXPECT_DOUBLE_EQ(bound_exp_kernel, 0);

  hawkes_kernel_sum_exp->get_convolution(time0, timestamps, &bound_exp_kernel);
  EXPECT_DOUBLE_EQ(bound_exp_kernel, intensities.dot(decays));
  for (double test_time : test_times) {
    bound_exp_kernel = -1.;
    hawkes_kernel_sum_exp->get_convolution(test_time, timestamps, &bound_exp_kernel);
    EXPECT_DOUBLE_EQ(bound_exp_kernel,
                     compute_sumexpkernel_convolution_bound(decays, intensities, timestamps,
                                                            test_time));
  }
}

TEST_F(HawkesKernelSumExpTest, get_convolution_after_copy) {
  hawkes_kernel_sum_exp->get_convolution(4., timestamps, nullptr);

  HawkesKernelSumExp hawkes_kernel_sum_exp_copy = *hawkes_kernel_sum_exp;
  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp_copy.get_convolution(time0 - 0.1, timestamps, nullptr), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp_copy.get_convolution(time0, timestamps, nullptr),
                   intensities.dot(decays));

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_sum_exp_copy.get_convolution(test_time, timestamps, nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, timestamps, test_time));
  }
}

TEST_F(HawkesKernelSumExpTest, get_convolution_after_duplicate_if_necessary) {
  auto shared_hawkes_kernel_sum_exp = std::make_shared<HawkesKernelSumExp>(intensities, decays);

  shared_hawkes_kernel_sum_exp->get_convolution(4., timestamps, nullptr);

  auto shared_hawkes_kernel_sum_exp_copy =
      shared_hawkes_kernel_sum_exp->duplicate_if_necessary(shared_hawkes_kernel_sum_exp);

  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_sum_exp_copy->get_convolution(time0 - 0.1, timestamps, nullptr),
                   0);
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_sum_exp_copy->get_convolution(time0, timestamps, nullptr),
                   intensities.dot(decays));

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(shared_hawkes_kernel_sum_exp_copy->get_convolution(test_time, timestamps,
                                                                        nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, timestamps, test_time));
  }
}

TEST_F(HawkesKernelSumExpTest, get_convolution_after_duplicate_if_necessary2) {
  auto shared_hawkes_kernel_sum_exp = std::make_shared<HawkesKernelSumExp>(intensities, decays);

  shared_hawkes_kernel_sum_exp->get_convolution(4., timestamps, nullptr);

  auto shared_hawkes_kernel_sum_exp_copy =
      shared_hawkes_kernel_sum_exp->duplicate_if_necessary();

  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_sum_exp_copy->get_convolution(time0 - 0.1, timestamps, nullptr),
                   0);
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_sum_exp_copy->get_convolution(time0, timestamps, nullptr),
                   intensities.dot(decays));

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(shared_hawkes_kernel_sum_exp_copy->get_convolution(test_time, timestamps,
                                                                        nullptr),
                     compute_sumexpkernel_convolution(decays, intensities, timestamps, test_time));
  }
}


#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
