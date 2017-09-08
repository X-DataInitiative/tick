
#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include <variants/hawkes_fixed_expkern_leastsq_list.h>
#include <variants/hawkes_fixed_expkern_loglik_list.h>
#include "hawkes_sdca_loglik_kern.h"


class HawkesInferenceTest : public ::testing::Test {
 protected:
  SArrayDoublePtrList1D timestamps;
  SArrayDoublePtrList2D timestamps_list;

  VArrayDoublePtr end_times;

  void SetUp() override {
    timestamps = SArrayDoublePtrList1D(0);
    // Test will fail if process array is not sorted
    ArrayDouble timestamps_0 = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
    timestamps.push_back(timestamps_0.as_sarray_ptr());
    ArrayDouble timestamps_1 = ArrayDouble {0.12, 1.19, 2.12, 2.41, 3.35, 4.21};
    timestamps.push_back(timestamps_1.as_sarray_ptr());

    auto timestamps_add = SArrayDoublePtrList1D(0);
    ArrayDouble timestamps_2= ArrayDouble {0.21, 0.43, 1.21, 1.35, 2.25};
    timestamps_add.push_back(timestamps_2.as_sarray_ptr());
    ArrayDouble timestamps_3 = ArrayDouble {0.93, 1.83, 2.33, 2.41, 3.35, 5.71};
    timestamps_add.push_back(timestamps_3.as_sarray_ptr());

    timestamps_list = SArrayDoublePtrList2D(0);
    timestamps_list.push_back(timestamps);
    timestamps_list.push_back(timestamps_add);

    end_times = VArrayDouble::new_ptr(2);
    (*end_times)[0] = 5.65; (*end_times)[1] = 5.87;
  }
};

TEST_F(HawkesInferenceTest, compute_weights) {
  const double decay = 3.;
  const double l_l2sq = 1e-3;

  HawkesSDCALoglikKern hawkes(decay, l_l2sq);
  hawkes.set_data(timestamps_list, end_times);
  hawkes.compute_weights();
}

TEST_F(HawkesInferenceTest, solve) {
  const double decay = 3.;
  const double l_l2sq = 1e-3;

  HawkesSDCALoglikKern hawkes(decay, l_l2sq);
  hawkes.set_data(timestamps_list, end_times);
  hawkes.solve();
}
