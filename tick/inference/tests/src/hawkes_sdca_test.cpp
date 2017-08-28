

#include <gtest/gtest.h>
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

    timestamps_list = SArrayDoublePtrList2D(0);
    timestamps_list.push_back(timestamps);
    timestamps_list.push_back(timestamps);

    end_times = VArrayDouble::new_ptr(2);
    (*end_times)[0] = 5.65; (*end_times)[1] = 5.87;
  }
};


TEST_F(HawkesInferenceTest, constructor) {
  const double decay = 3.;
  const double l_l2sq = 1e-3;

  HawkesSDCALoglikKern hawkes(decay, l_l2sq);
  hawkes.set_data(timestamps_list, end_times);
}

TEST_F(HawkesInferenceTest, compute_weights) {
  const double decay = 3.;
  const double l_l2sq = 1e-3;

  HawkesSDCALoglikKern hawkes(decay, l_l2sq);
  hawkes.set_data(timestamps_list, end_times);
  hawkes.compute_weights();
}