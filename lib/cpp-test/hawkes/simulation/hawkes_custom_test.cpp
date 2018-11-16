//
// Created by pwu on 1/3/18.
//

#include <gtest/gtest.h>
#include "hawkes_custom.h"
#include "hawkes.h"

class HawkesCustomTest : public ::testing::Test {
protected:
};

TEST_F(HawkesCustomTest, CustomTest) {

    int n_nodes = 2;
    int seed = 10086;
    ulong MaxN_of_f = 5;

    SArrayDoublePtrList1D f_i;
    f_i = SArrayDoublePtrList1D(0);
    // Test will fail if process array is not sorted
    ArrayDouble f_i_0 = ArrayDouble {1.0, 7, 7.7, 6, 3};
    f_i.push_back(f_i_0.as_sarray_ptr());
    ArrayDouble f_i_1 = ArrayDouble {1.0, 0.5, 2, 1, 2};
    f_i.push_back(f_i_1.as_sarray_ptr());

    Hawkes_custom model(n_nodes, seed, MaxN_of_f, f_i);

    model.simulate(10.0);
}

TEST_F(HawkesCustomTest, HawkesTest) {

    int n_nodes = 2;
    int seed = 10086;

    Hawkes model(n_nodes, seed);

    model.simulate(10.0);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "fast";
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN