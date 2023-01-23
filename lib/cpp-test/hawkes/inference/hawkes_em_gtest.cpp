// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/inference/hawkes_em.h"

#define EPSILON 1e-64

class HawkesEMTest : public ::testing::Test {
 protected:
  ulong n_nodes = 2;
  double kernel_support = 1.;
  ulong kernel_size = 10;
  double dt = kernel_support / kernel_size;
  double t0 = .0;
  ArrayDouble kernel_discretization;
  ArrayDouble mu;
  ArrayDouble2d kernels;
  HawkesEM em{kernel_support, kernel_size, 1};
  void SetUp() override {
    kernel_discretization = ArrayDouble{.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.};
    em.set_n_nodes(n_nodes);
    mu = ArrayDouble{.05, .05};
    kernels = ArrayDouble2d(n_nodes, n_nodes * kernel_size);
    kernels[0] = .0;
    kernels[1] = .1;
    kernels[2] = .2;
    kernels[3] = .3;
    kernels[4] = .4;
    kernels[5] = .1;
    kernels[6] = .1;
    kernels[7] = .1;
    kernels[8] = .1;
    kernels[9] = .0;
    kernels[10] = .5;
    kernels[11] = .4;
    kernels[12] = .3;
    kernels[13] = .2;
    kernels[14] = .1;
    kernels[15] = .0;
    kernels[16] = .1;
    kernels[17] = .2;
    kernels[18] = .3;
    kernels[19] = .0;
    kernels[20] = .5;
    kernels[21] = .4;
    kernels[22] = .0;
    kernels[23] = .0;
    kernels[24] = .0;
    kernels[25] = .0;
    kernels[26] = .0;
    kernels[27] = .0;
    kernels[28] = .0;
    kernels[29] = .0;
    kernels[30] = .2;
    kernels[31] = .2;
    kernels[32] = .4;
    kernels[33] = .2;
    kernels[34] = .1;
    kernels[35] = .0;
    kernels[36] = .0;
    kernels[37] = .0;
    kernels[38] = .0;
    kernels[39] = .0;
  }
};

TEST_F(HawkesEMTest, can_get_kernel_size) { EXPECT_EQ(em.get_kernel_size(), kernel_size); }

TEST_F(HawkesEMTest, can_get_n_nodes) { EXPECT_EQ(em.get_n_nodes(), n_nodes); }

TEST_F(HawkesEMTest, kernel_dt) {
  EXPECT_DOUBLE_EQ(em.get_kernel_dt(), dt);
  EXPECT_DOUBLE_EQ(em.get_kernel_fixed_dt(), dt);
}

TEST_F(HawkesEMTest, kernel_discretization_test) {
  ArrayDouble ks = *em.get_kernel_discretization();
  EXPECT_EQ(kernel_discretization.size(), ks.size());
  EXPECT_EQ(kernel_size + 1, ks.size());
  EXPECT_DOUBLE_EQ(kernel_discretization[0], em.get_kernel_t0());
  for (ulong t = 0; t < ks.size(); t++) {
    EXPECT_DOUBLE_EQ(ks[t], kernel_discretization[t]);
  }
  em.set_kernel_discretization(kernel_discretization.as_sarray_ptr());
  EXPECT_EQ(em.get_kernel_size(), kernel_size);
  EXPECT_NEAR(kernel_discretization[0], em.get_kernel_t0(), 1e-100);
  EXPECT_DOUBLE_EQ(em.get_kernel_dt(), dt);
}

TEST_F(HawkesEMTest, kernel_time_func_dt) {
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      EXPECT_NEAR(timefunc[uv].get_t0(), kernel_discretization[0], 1e-100);
      EXPECT_DOUBLE_EQ(timefunc[uv].get_t0(), em.get_kernel_t0());
      EXPECT_DOUBLE_EQ(timefunc[uv].get_dt(), dt);
    }
  }
}

TEST_F(HawkesEMTest, kernel_time_func_data) {
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      ArrayDouble data_uv = *(timefunc[uv].get_sampled_y());
      EXPECT_EQ(data_uv.size(), kernel_size);
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        // Test data values
        EXPECT_DOUBLE_EQ(data_uv[k], kernels(u, vk))
            << "Kernel[" << u << ", " << v << "]: "
            << "Value of " << k << "-th sample data  gives a mismatch.";
        // Test abscissa
        EXPECT_DOUBLE_EQ(timefunc[uv].get_t0() + k * timefunc[uv].get_dt(),
                         (*em.get_kernel_discretization())[k])
            << "Kernel[" << u << ", " << v << "]: "
            << "Value of " << k << "-th abscissa point  gives a mismatch.";
      }
    }
  }
}

TEST_F(HawkesEMTest, kernel_time_func_data_with_explicit_abscissa) {
  // Rounding errors in the division T[10] - T[0] can result in
  // the size of `sampled_y` being larger than the size of `Y` by 1
  em.set_kernel_discretization(kernel_discretization.as_sarray_ptr());
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      ArrayDouble data_uv = *(timefunc[uv].get_sampled_y());
      EXPECT_NEAR(data_uv.size(), kernel_size + 1, 1);
      /*
      std::cout << "Kernel[" << u << ", " << v << "]: " << std::endl;
      for (ulong j = 0; j < data_uv.size(); j++) {
        std::cout << "data_uv[" << j << "] = " << data_uv[j] << std::endl;
      }
      */
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        // Test data values
        EXPECT_DOUBLE_EQ(data_uv[k], kernels(u, vk))
            << "Kernel[" << u << ", " << v << "]: " << std::endl
            << "Value of " << k << "-th sample data  gives a mismatch." << std::endl
            << "Corresponding kernel discretization point : "
            << (*em.get_kernel_discretization())[k] << std::endl;
      }
    }
  }
}

TEST_F(HawkesEMTest, kernel_time_func_values) {
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        double t = t0 + k * dt + .5 * dt;  // kernel_discretization[k] + .5 * dt;
        // EXPECT_DOUBLE_EQ(t, kernel_discretization[k] + .5 * dt);
        EXPECT_DOUBLE_EQ(timefunc[uv].value(t), kernels(u, vk))
            << "Kernel[" << u << ", " << v << "]: "
            << "Value at time t = " << t << " gives a mismatch.\n"
            << "Abscissa position k=" << k << std::endl
            << std::endl;
      }
    }
  }
}

TEST_F(HawkesEMTest, kernel_time_func_values_with_explicit_abscissa) {
  em.set_kernel_discretization(kernel_discretization.as_sarray_ptr());
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        double t = t0 + k * dt + .5 * dt;
        EXPECT_DOUBLE_EQ(timefunc[uv].value(t), kernels(u, vk))
            << "Kernel[" << u << ", " << v << "]: " << std::endl
            << "Value at time t = " << t << " gives a mismatch.\n"
            << "Abscissa position k=" << k << std::endl
            << "Abscissa[k] = " << t0 + k * dt << std::endl
            << std::endl;
      }
    }
  }
}

TEST_F(HawkesEMTest, kernel_time_func_primitive) {
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  ArrayDouble kernel_discretization = *em.get_kernel_discretization();
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      double discrete_integral = .0;
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        double t = t0 + (k + 1) * dt;
        discrete_integral += kernels(u, vk) * dt;
        EXPECT_DOUBLE_EQ(timefunc[uv].primitive(t), discrete_integral)
            << "Kernel[" << u << ", " << v << "]: "
            << "Primitive at time t=" << t << " (k = " << k << ") gives a mismatch.";
      }
    }
  }
}

TEST_F(HawkesEMTest, kernel_time_func_primitive_with_explicit_abscissa) {
  em.set_kernel_discretization(kernel_discretization.as_sarray_ptr());
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ASSERT_FALSE(timefunc.empty());
  ArrayDouble kernel_discretization = *em.get_kernel_discretization();
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      double discrete_integral = .0;
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        double t = t0 + (k + 1) * dt;
        discrete_integral += kernels(u, vk) * dt;
        EXPECT_NEAR(timefunc[uv].primitive(t), discrete_integral, 1e-16)
            << "Kernel[" << u << ", " << v << "]: "
            << "Primitive at time t=" << t << " (k = " << k << ") gives a mismatch.";
      }
    }
  }
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
