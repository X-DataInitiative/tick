// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/hawkes/inference/hawkes_em.h"

class HawkesEMTest : public ::testing::Test {
 protected:
  ulong n_nodes = 2;
  double kernel_support = 1.;
  ulong kernel_size = 10;
  ArrayDouble mu;
  ArrayDouble2d kernels;
  HawkesEM em{kernel_support, kernel_size, 1};
  void SetUp() override {
    mu = ArrayDouble{.05, .05};
    kernels = ArrayDouble2d(n_nodes, n_nodes * kernel_size);
    kernels[0] = .1;
    kernels[1] = .1;
    kernels[2] = .1;
    kernels[3] = .1;
    kernels[4] = .1;
    kernels[5] = .1;
    kernels[7] = .1;
    kernels[8] = .1;
    kernels[9] = .1;
    kernels[10] = .2;
    kernels[11] = .2;
    kernels[12] = .1;
    kernels[13] = .0;
    kernels[14] = .0;
    kernels[15] = .0;
    kernels[17] = .1;
    kernels[18] = .2;
    kernels[19] = .2;
    kernels[20] = .0;
    kernels[21] = .0;
    kernels[22] = .0;
    kernels[23] = .0;
    kernels[24] = .0;
    kernels[25] = .0;
    kernels[27] = .0;
    kernels[28] = .0;
    kernels[29] = .0;
    kernels[30] = .2;
    kernels[31] = .2;
    kernels[32] = .4;
    kernels[33] = .1;
    kernels[34] = .1;
    kernels[35] = .0;
    kernels[37] = .0;
    kernels[38] = .0;
    kernels[39] = .0;
  }
};

TEST_F(HawkesEMTest, can_get_kernel_size) { EXPECT_EQ(em.get_kernel_size(), kernel_size); }

/*
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
*/

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
