// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/inference/hawkes_em.h"

class HawkesEMTest : public ::testing::Test {
 protected:
  ulong n_nodes;
  double kernel_support;
  ulong kernel_size;
  ArrayDouble mu;
  ArrayDouble2d kernels;
  HawkesEM em;
  void SetUp() override {
    n_nodes = 2;
    kernel_support = 1.;
    kernel_size = 10;
    em = HawkesEM(kernel_support, kernel_size, 1);
    mu = ArrayDouble{.05, .05};
    kernels = ArrayDouble2d{
        {.1, .1, .1, .1, .1, .1, .1, .1, .1, .1, .2, .2, .1, .0, .0, .0, .0, .1, .2, .2},
        {.0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .0, .2, .2, .4, .1, .1, .0, .0, .0}};
  }
};

TEST_F(HawkesEMTest, get_kernel_size) { EXPECT_EQ(em.get_kernel_size(), kernel_size); }

TEST_F(HawkesEMTest, init_kernel_time_func) {
  em.init_kernel_time_func(kernels);
  std::vector<TimeFunction>& timefunc = em.get_kernel_time_func();
  ArrayDouble kernel_discretization = *em.get_kernel_discretization();
  for (ulong u = 0; u < n_nodes; ++u) {
    for (ulong v = 0; v < n_nodes; ++v) {
      ulong uv = u * n_nodes + v;
      for (ulong k = 0; k < kernel_size; ++k) {
        ulong vk = v * kernel_size + k;
        double t = kernel_discretization[k];
        EXPECT_DOUBLE_EQ(timefunc[uv].evaluate(t), kernels(u, vk));
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
