// License: BSD 3 clause


#include <gtest/gtest.h>
#include "tick/hawkes/simulation/hawkes_kernels/hawkes_kernel_exp.h"

double compute_expkernel_convolution(double decay,
                                     double intensity,
                                     ArrayDouble timestamps,
                                     double time) {
  double kernel_sum{0.};
  for (ulong i = 0; i < timestamps.size(); ++i) {
    double t_i = timestamps[i];
    if (time >= t_i) {
      kernel_sum += intensity * decay * exp(-decay * (time - t_i));
    }
  }
  return kernel_sum;
}

double compute_expkernel_convolution_bound(double decay,
                                           double intensity,
                                           ArrayDouble timestamps,
                                           double time) {
  double kernel_sum{0.};
  for (ulong i = 0; i < timestamps.size(); ++i) {
    double t_i = timestamps[i];
    if (time >= t_i) {
      kernel_sum += intensity * decay * exp(-decay * (time - t_i));
    }
  }
  return kernel_sum;
}

class HawkesKernelExpTest : public ::testing::Test {
 protected:
  double intensity;
  double decay;
  HawkesKernelExp hawkes_kernel_exp;
  ArrayDouble timestamps;

  // Test will fail if test_times array is not sorted
  std::array<double, 8> test_times{{0.93, 1., 2., 2.32, 3.5, 5., 8., 100.}};

  HawkesKernelExpTest() : hawkes_kernel_exp(0, 0) {}

  void SetUp() override {
    intensity = 2.;
    decay = 3.;
    hawkes_kernel_exp = HawkesKernelExp(intensity, decay);

    // Test will fail if process array is not sorted
    timestamps = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
  }
};

TEST_F(HawkesKernelExpTest, is_zero) {
  EXPECT_FALSE(hawkes_kernel_exp.is_zero());
}

TEST_F(HawkesKernelExpTest, get_value) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_value(-3), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_value(0), intensity * decay);
  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_value(test_time),
                     intensity * decay * exp(-decay * test_time));
  }
}

TEST_F(HawkesKernelExpTest, get_future_max) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_future_max(-3, 0), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_future_max(0, intensity * decay), intensity * decay);
  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_future_max(test_time,
                                                      intensity * decay * exp(-decay * test_time)),
                     intensity * decay * exp(-decay * test_time));
  }
}

TEST_F(HawkesKernelExpTest, get_norm) {
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_norm(), intensity);
}

TEST_F(HawkesKernelExpTest, get_convolution_value) {
  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(time0 - 0.1, timestamps, nullptr), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(time0, timestamps, nullptr), intensity * decay);

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(test_time, timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, timestamps, test_time));
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(test_time, timestamps, nullptr),
                     hawkes_kernel_exp.HawkesKernel::get_convolution(test_time, timestamps, nullptr));
  }
}

// This edge case happen when we are simulating Hawkes processes
TEST_F(HawkesKernelExpTest, get_convolution_value_while_appending_array) {
  VArrayDouble v_timestamps(0);

  for (int k = 0; k < test_times.size() - 1; ++k) {
    double t_k = test_times[k];
    double t_k_next = test_times[k + 1];
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(t_k, v_timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, v_timestamps, t_k));
    v_timestamps.append1(t_k);
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(t_k, v_timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, v_timestamps, t_k));
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(t_k_next, v_timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, v_timestamps, t_k_next));
    break;
  }
}

TEST_F(HawkesKernelExpTest, get_convolution_older_time_error) {
  double last_time = 4.;
  hawkes_kernel_exp.get_convolution(last_time, timestamps, nullptr);

  double old_time = 2.;
  ASSERT_THROW(hawkes_kernel_exp.get_convolution(old_time, timestamps, nullptr),
               std::runtime_error);
}

TEST_F(HawkesKernelExpTest, get_convolution_older_time_rewind) {
  double last_time = 4.;
  hawkes_kernel_exp.get_convolution(last_time, timestamps, nullptr);
  hawkes_kernel_exp.rewind();

  double old_time = 2.;
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(old_time, timestamps, nullptr),
                   compute_expkernel_convolution(decay, intensity, timestamps, old_time));
}

TEST_F(HawkesKernelExpTest, get_convolution_other_timestamps_rewind) {
  double test_time = 4.;
  hawkes_kernel_exp.get_convolution(test_time, timestamps, nullptr);
  hawkes_kernel_exp.rewind();

  ArrayDouble other_timestamps {0.23, 1.34, 2.17};
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp.get_convolution(test_time, other_timestamps, nullptr),
                   compute_expkernel_convolution(decay, intensity, other_timestamps, test_time));
}

TEST_F(HawkesKernelExpTest, get_convolution_bound) {
  double time0 = timestamps[0];

  double bound_exp_kernel{-1.};

  hawkes_kernel_exp.get_convolution(time0 - 0.1, timestamps, &bound_exp_kernel);
  EXPECT_DOUBLE_EQ(bound_exp_kernel, 0);

  hawkes_kernel_exp.get_convolution(time0, timestamps, &bound_exp_kernel);
  EXPECT_DOUBLE_EQ(bound_exp_kernel, intensity * decay);
  for (double test_time : test_times) {
    bound_exp_kernel = -1.;
    double bound_original{1.};

    hawkes_kernel_exp.get_convolution(test_time, timestamps, &bound_exp_kernel);
    hawkes_kernel_exp.HawkesKernel::get_convolution(test_time, timestamps, &bound_original);
    EXPECT_DOUBLE_EQ(bound_exp_kernel, bound_original);
    EXPECT_DOUBLE_EQ(bound_exp_kernel,
                     compute_expkernel_convolution_bound(decay, intensity, timestamps, test_time));
  }
}

TEST_F(HawkesKernelExpTest, get_convolution_bound_negative_intensity) {
  intensity = -2.;
  hawkes_kernel_exp = HawkesKernelExp(intensity, decay);

  double time0 = timestamps[0];

  double bound_exp_kernel{-1.};

  hawkes_kernel_exp.get_convolution(time0 - 0.1, timestamps, &bound_exp_kernel);
  EXPECT_DOUBLE_EQ(bound_exp_kernel, 0);
  bound_exp_kernel = -1;
  hawkes_kernel_exp.get_convolution(time0, timestamps, &bound_exp_kernel);
  EXPECT_DOUBLE_EQ(bound_exp_kernel, 0);

  for (double test_time : test_times) {
    bound_exp_kernel = -1.;
    hawkes_kernel_exp.get_convolution(test_time, timestamps, &bound_exp_kernel);
    EXPECT_DOUBLE_EQ(bound_exp_kernel, 0);
  }
}

TEST_F(HawkesKernelExpTest, get_convolution_after_copy) {
  hawkes_kernel_exp.get_convolution(4., timestamps, nullptr);

  HawkesKernelExp hawkes_kernel_exp_copy = hawkes_kernel_exp;
  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp_copy.get_convolution(time0 - 0.1, timestamps, nullptr), 0);
  EXPECT_DOUBLE_EQ(hawkes_kernel_exp_copy.get_convolution(time0, timestamps, nullptr),
                   intensity * decay);

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(hawkes_kernel_exp_copy.get_convolution(test_time, timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, timestamps, test_time));
  }
}

TEST_F(HawkesKernelExpTest, get_convolution_after_duplicate_if_necessary) {
  auto shared_hawkes_kernel_exp = std::make_shared<HawkesKernelExp>(intensity, decay);

  shared_hawkes_kernel_exp->get_convolution(4., timestamps, nullptr);

  auto shared_hawkes_kernel_exp_copy =
      shared_hawkes_kernel_exp->duplicate_if_necessary(shared_hawkes_kernel_exp);

  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_exp_copy->get_convolution(time0 - 0.1, timestamps, nullptr),
                   0);
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_exp_copy->get_convolution(time0, timestamps, nullptr),
                   intensity * decay);

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(shared_hawkes_kernel_exp_copy->get_convolution(test_time, timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, timestamps, test_time));
  }
}

TEST_F(HawkesKernelExpTest, get_convolution_after_duplicate_if_necessary2) {
  auto shared_hawkes_kernel_exp = std::make_shared<HawkesKernelExp>(intensity, decay);

  shared_hawkes_kernel_exp->get_convolution(4., timestamps, nullptr);

  auto shared_hawkes_kernel_exp_copy =
      shared_hawkes_kernel_exp->duplicate_if_necessary();

  double time0 = timestamps[0];
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_exp_copy->get_convolution(time0 - 0.1, timestamps, nullptr),
                   0);
  EXPECT_DOUBLE_EQ(shared_hawkes_kernel_exp_copy->get_convolution(time0, timestamps, nullptr),
                   intensity * decay);

  for (double test_time : test_times) {
    EXPECT_DOUBLE_EQ(shared_hawkes_kernel_exp_copy->get_convolution(test_time, timestamps, nullptr),
                     compute_expkernel_convolution(decay, intensity, timestamps, test_time));
  }
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
