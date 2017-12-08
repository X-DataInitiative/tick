// License: BSD 3 clause

#include <numeric>
#include <algorithm>
#include <complex>

#define DEBUG_COSTLY_THROW 1

#include <gtest/gtest.h>
#include "tick/hawkes/model/hawkes_fixed_sumexpkern_loglik.h"

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

#include "tick/array/array.h"
#include "tick/hawkes/model/hawkes_fixed_expkern_loglik.h"
#include "tick/hawkes/model/hawkes_fixed_sumexpkern_loglik.h"
#include "tick/hawkes/model/hawkes_fixed_expkern_leastsq.h"
#include "tick/hawkes/model/hawkes_fixed_sumexpkern_leastsq.h"

#include "tick/hawkes/model/variants/hawkes_fixed_expkern_leastsq_list.h"
#include "tick/hawkes/model/variants/hawkes_fixed_sumexpkern_leastsq_list.h"
#include "tick/hawkes/model/variants/hawkes_fixed_expkern_loglik_list.h"
#include "tick/hawkes/model/variants/hawkes_fixed_sumexpkern_loglik_list.h"


class HawkesModelTest : public ::testing::Test {
 protected:
  SArrayDoublePtrList1D timestamps;

  void SetUp() override {
    timestamps = SArrayDoublePtrList1D(0);
    // Test will fail if process array is not sorted
    ArrayDouble timestamps_0 = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
    timestamps.push_back(timestamps_0.as_sarray_ptr());
    ArrayDouble timestamps_1 = ArrayDouble {0.12, 1.19, 2.12, 2.41, 3.35, 4.21};
    timestamps.push_back(timestamps_1.as_sarray_ptr());
  }
};

TEST_F(HawkesModelTest, compute_weights_loglikelihood){
  ModelHawkesFixedExpKernLogLik model(2);
  model.set_data(timestamps, 4.25);
  model.compute_weights();
}

TEST_F(HawkesModelTest, compute_loss_loglikelihood){
  ModelHawkesFixedExpKernLogLik model(2);
  model.set_data(timestamps, 4.25);
  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  const double loss = model.loss(coeffs);
  ArrayDouble grad(model.get_n_coeffs());
  model.grad(coeffs, grad);

  EXPECT_DOUBLE_EQ(loss, 2.9434509731246283);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 6);
}

TEST_F(HawkesModelTest, compute_loss_loglikelihood_sparse){
  ModelHawkesFixedExpKernLogLik model(2);
  auto sparse_timestamps = SArrayDoublePtrList1D(0);
  // Test will fail if process array is not sorted
  ArrayDouble timestamps_0 = ArrayDouble {0.31, 0.93, 1.29, 2.32, 4.25};
  sparse_timestamps.push_back(timestamps_0.as_sarray_ptr());
  ArrayDouble timestamps_1 = ArrayDouble(0);
  sparse_timestamps.push_back(timestamps_1.as_sarray_ptr());

  model.set_data(sparse_timestamps, 4.25);
  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  const double loss = model.loss(coeffs);
  ArrayDouble grad(model.get_n_coeffs());
  model.grad(coeffs, grad);

  EXPECT_DOUBLE_EQ(loss, 5.9243119662517856);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 6);
}

TEST_F(HawkesModelTest, check_sto_loglikelihood){
  ModelHawkesFixedExpKernLogLik model(2);
  model.set_data(timestamps, 6.);
  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  const double loss = model.loss(coeffs);

  ArrayDouble grad(model.get_n_coeffs());
  model.grad(coeffs, grad);

  double sum_sto_loss = 0;
  ArrayDouble sto_grad(model.get_n_coeffs());
  sto_grad.init_to_zero();
  ArrayDouble tmp_sto_grad(model.get_n_coeffs());

  for (ulong i = 0; i < model.get_rand_max(); ++i) {
    sum_sto_loss += model.loss_i(i, coeffs) / model.get_rand_max();
    tmp_sto_grad.init_to_zero();
    model.grad_i(i, coeffs, tmp_sto_grad);
    sto_grad.mult_incr(tmp_sto_grad, 1. / model.get_rand_max());
  }

  EXPECT_DOUBLE_EQ(loss, sum_sto_loss);
  for (ulong i  = 0; i < model.get_n_coeffs(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_DOUBLE_EQ(grad[i], sto_grad[i]);
  }
}

TEST_F(HawkesModelTest, compute_loss_loglikelihood_sum_exp_kern){
  ArrayDouble decays {1., 2., 3.};

  const double end_time = 5.65;
  ModelHawkesFixedSumExpKernLogLik model(decays, 1);
  model.set_data(timestamps, end_time);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1., 5., 3., 2., 4., 2., 3., 4., 5.};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 0.43573314143220188);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 8.4919969312665398);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 17.202925821121468);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 14);
}

TEST_F(HawkesModelTest, compute_loss_least_squares){
  ArrayDouble2d decays(2, 2);
  decays.fill(2);

  ModelHawkesFixedExpKernLeastSq model(decays.as_sarray2d_ptr(), 2);
  model.set_data(timestamps, 5.65);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 177.74263433770577);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 300.36718283368231);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 43.46452883376255);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 6);
}

TEST_F(HawkesModelTest, hawkes_least_squares_serialization){

  ArrayDouble2d decays(2, 2);
  decays.fill(2);
  auto sdecays = decays.as_sarray2d_ptr();

  ModelHawkesFixedExpKernLeastSq model(sdecays, 2);
  model.set_data(timestamps, 5.65);

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  std::stringstream os;
  {
    cereal::JSONOutputArchive outputArchive(os);

    outputArchive( model );
  }

  {
    cereal::JSONInputArchive inputArchive(os);

    ModelHawkesFixedExpKernLeastSq restored_model;
    inputArchive( restored_model );

    EXPECT_EQ(restored_model.get_n_nodes(), 2);
    EXPECT_EQ(restored_model.get_end_time(), 5.65);
    EXPECT_EQ(restored_model.get_n_total_jumps(), model.get_n_total_jumps());

    EXPECT_DOUBLE_EQ(restored_model.loss(coeffs), model.loss(coeffs));
  }
}

TEST_F(HawkesModelTest, compute_loss_least_square_sum_exp_kern){
  ArrayDouble decays(2);
  decays.fill(2);

  const double end_time = 5.65;
  ModelHawkesFixedSumExpKernLeastSq model(decays, 1, end_time, 2);
  model.set_data(timestamps, end_time);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1., 5., 3., 2., 4.};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 709.43688360602232);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 1717.7627409202796);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 220.65451132057288);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 10);
}

TEST_F(HawkesModelTest, compute_loss_least_square_sum_exp_varying_baseline){
  ArrayDouble decays(2);
  decays.fill(2);

  const double end_time = 5.87;
  ModelHawkesFixedSumExpKernLeastSq model(decays, 3, 2., 1);
  model.set_data(timestamps, end_time);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 0., 1., 1., 3., 2., 3., 4., 1., 5., 3., 2., 4.};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 754.50509295231836);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 1488.8712825118096);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 203.94330686037526);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 14);
}


TEST_F(HawkesModelTest, hawkes_least_squares_sum_exp_serialization){

  ArrayDouble decays {2., 3.};

  ModelHawkesFixedSumExpKernLeastSq model(decays, 2, 3.);
  model.set_data(timestamps, 5.65);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 0., 1., 1., 3., 2., 3., 4., 1., 5., 3., 2., 4.};

  std::stringstream os;
  {
    cereal::JSONOutputArchive outputArchive(os);

    outputArchive( model );
  }

  {
    cereal::JSONInputArchive inputArchive(os);

    ModelHawkesFixedSumExpKernLeastSq restored_model;
    inputArchive( restored_model );

    EXPECT_EQ(restored_model.get_n_nodes(), 2);
    EXPECT_EQ(restored_model.get_end_time(), 5.65);
    EXPECT_EQ(restored_model.get_n_total_jumps(), model.get_n_total_jumps());

    EXPECT_DOUBLE_EQ(restored_model.loss(coeffs), model.loss(coeffs));
  }
}

TEST_F(HawkesModelTest, compute_loss_least_square_list){
  ArrayDouble2d decays(2, 2);
  decays.fill(2);

  ModelHawkesFixedExpKernLeastSqList model(decays.as_sarray2d_ptr(), 2);

  auto timestamps_list = SArrayDoublePtrList2D(0);
  timestamps_list.push_back(timestamps);
  timestamps_list.push_back(timestamps);

  auto end_times = VArrayDouble::new_ptr(2);
  (*end_times)[0] = 5.65; (*end_times)[1] = 5.87;

  model.set_data(timestamps_list, end_times);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 356.00492335074784);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 603.45311621338624);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 43.611729071097002);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 6);
}

TEST_F(HawkesModelTest, least_square_list_serialization){
  ArrayDouble2d decays(2, 2);
  decays.fill(2);

  ModelHawkesFixedExpKernLeastSqList model(decays.as_sarray2d_ptr(), 2);

  auto timestamps_list = SArrayDoublePtrList2D(0);
  timestamps_list.push_back(timestamps);
  timestamps_list.push_back(timestamps);

  auto end_times = VArrayDouble::new_ptr(2);
  (*end_times)[0] = 5.65; (*end_times)[1] = 5.87;

  model.set_data(timestamps_list, end_times);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  std::stringstream os;
  {
    cereal::BinaryOutputArchive outputArchive(os);

    outputArchive( model );
  }

  {
    cereal::BinaryInputArchive inputArchive(os);

    ModelHawkesFixedExpKernLeastSqList restored_model(nullptr, 0);
    inputArchive( restored_model );

    EXPECT_EQ(restored_model.get_n_nodes(), 2);
    EXPECT_EQ((*restored_model.get_end_times())[1], 5.87);
    EXPECT_EQ(restored_model.get_n_total_jumps(), model.get_n_total_jumps());

    EXPECT_DOUBLE_EQ(restored_model.loss(coeffs), model.loss(coeffs));
  }
}

TEST_F(HawkesModelTest, compute_loss_least_square_sum_exp_list){
  ArrayDouble decays(2);
  decays.fill(2);

  ArrayDouble end_times {5.65, 5.87};
  ModelHawkesFixedSumExpKernLeastSqList model(decays, 1, 1e300, 1);
  model.incremental_set_data(timestamps, end_times[0]);
  model.incremental_set_data(timestamps, end_times[1]);

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1., 5., 3., 2., 4.};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 1419.8117850574868);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 3439.937591029975);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 220.89769891306648);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 10);
}


TEST_F(HawkesModelTest, compute_loss_least_square_sum_exp_list_varying_baseline){
  ArrayDouble decays(2);
  decays.fill(2);

  ArrayDouble end_times {5.65, 5.87};
  ModelHawkesFixedSumExpKernLeastSqList model(decays, 3, 2., 1);
  model.incremental_set_data(timestamps, end_times[0]);
  model.incremental_set_data(timestamps, end_times[1]);

  ArrayDouble coeffs = ArrayDouble {1., 3., 0., 1., 1., 3., 2., 3., 4., 1., 5., 3., 2., 4.};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 1508.7587849870356);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 2973.3304558342033);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 203.73132912823812);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 14);
}


TEST_F(HawkesModelTest, hawkes_least_squares_sum_exp_list_serialization){

  ArrayDouble decays {0.1, 5.};

  ArrayDouble end_times {5.65, 5.87};
  ModelHawkesFixedSumExpKernLeastSqList model(decays, 1, 1e300, 1);
  model.incremental_set_data(timestamps, end_times[0]);
  model.incremental_set_data(timestamps, end_times[1]);

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1., 5., 3., 2., 4.};

  std::stringstream os;
  {
    cereal::JSONOutputArchive outputArchive(os);

    outputArchive( model );
  }

  {
    cereal::JSONInputArchive inputArchive(os);

    ModelHawkesFixedSumExpKernLeastSqList restored_model;
    inputArchive( restored_model );

    EXPECT_EQ(restored_model.get_n_nodes(), 2);
    EXPECT_EQ((*restored_model.get_end_times())[1], 5.87);
    EXPECT_EQ(restored_model.get_n_total_jumps(), model.get_n_total_jumps());

    EXPECT_DOUBLE_EQ(restored_model.loss(coeffs), model.loss(coeffs));
  }
}

TEST_F(HawkesModelTest, compute_loss_loglik_list){
  const double decay = 2.;

  ModelHawkesFixedExpKernLogLikList model(decay, 2);

  auto timestamps_list = SArrayDoublePtrList2D(0);
  timestamps_list.push_back(timestamps);
  timestamps_list.push_back(timestamps);

  auto end_times = VArrayDouble::new_ptr(2);
  (*end_times)[0] = 5.65; (*end_times)[1] = 5.87;

  model.set_data(timestamps_list, end_times);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), -0.68144584020170718);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 1.671674207054755);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 4.1398114444887462);

  ArrayDouble vector = ArrayDouble {1, 3., 3., 7., 8., 1};
  EXPECT_DOUBLE_EQ(model.hessian_norm(coeffs, vector), 2.7963385385715074);
  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 6);
}

TEST_F(HawkesModelTest, check_sto_loglikelihood_list){
  const double decay = 2.;

  ModelHawkesFixedExpKernLogLikList model(decay, 1);

  model.incremental_set_data(timestamps, 5.65);
  model.incremental_set_data(timestamps, 5.87);

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};

  double loss = model.loss(coeffs);
  ArrayDouble grad(model.get_n_coeffs());
  model.grad(coeffs, grad);

  double sum_sto_loss = 0;
  ArrayDouble sto_grad(model.get_n_coeffs());
  sto_grad.init_to_zero();
  ArrayDouble tmp_sto_grad(model.get_n_coeffs());

  for (ulong i = 0; i < model.get_rand_max(); ++i) {
    sum_sto_loss += model.loss_i(i, coeffs) / model.get_rand_max();
    tmp_sto_grad.init_to_zero();
    model.grad_i(i, coeffs, tmp_sto_grad);
    sto_grad.mult_incr(tmp_sto_grad, 1. / model.get_rand_max());
  }

  EXPECT_DOUBLE_EQ(loss, sum_sto_loss);
  for (ulong i  = 0; i < model.get_n_coeffs(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_DOUBLE_EQ(grad[i], sto_grad[i]);
  }
}


TEST_F(HawkesModelTest, compute_loss_loglikelihood_list_sum_exp_kern){
  ArrayDouble decays {1., 2., 3.};

  ModelHawkesFixedSumExpKernLogLikList model(decays, 1);

  model.incremental_set_data(timestamps, 5.65);
  model.incremental_set_data(timestamps, 5.87);

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1., 5., 3., 2., 4., 2., 3., 4., 5.};

  EXPECT_DOUBLE_EQ(model.loss_i(0, coeffs), 0.43573314143220188);
  EXPECT_DOUBLE_EQ(model.loss_i(1, coeffs), 8.4919969312665398);

  EXPECT_DOUBLE_EQ(model.loss(coeffs), 17.270721554384703);

  EXPECT_DOUBLE_EQ(model.get_n_coeffs(), 14);
}

TEST_F(HawkesModelTest, compute_hessian_loglikelihood){
  ModelHawkesFixedExpKernLogLik model(2);
  model.set_data(timestamps, 4.25);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1};
  const int n_nodes = 2;
  ArrayDouble out((1 + n_nodes) * (n_nodes + n_nodes * n_nodes));
  out.init_to_zero();

  model.hessian(coeffs, out);

  // test some random indexes...
  EXPECT_DOUBLE_EQ(out[0], 0.01631907612617749);
  EXPECT_DOUBLE_EQ(out[4], 0.0060622624486514612);
  EXPECT_DOUBLE_EQ(out[6], 0.0070643083720372474);
  EXPECT_DOUBLE_EQ(out[8], 0.0059903489344126197);
  EXPECT_DOUBLE_EQ(out[9], 0.01668440042028695);
}

TEST_F(HawkesModelTest, compute_hessian_sumexp_loglikelihood){
  ArrayDouble decays {1., 2.};

  ModelHawkesFixedSumExpKernLogLik model(decays, 1);
  model.set_data(timestamps, 4.25);
  model.compute_weights();

  ArrayDouble coeffs = ArrayDouble {1., 3., 2., 3., 4., 1., 5., 3., 2., 4., 2., 3., 4., 5.};
  const ulong n_nodes = 2;
  const ulong n_alpha_i = n_nodes * decays.size();
  ArrayDouble out((1 + n_alpha_i) * (n_nodes + n_alpha_i * n_nodes));
  out.init_to_zero();

  model.hessian(coeffs, out);

  EXPECT_DOUBLE_EQ(out[0], 0.0074914657915579903);
  EXPECT_DOUBLE_EQ(out[4], 0.0088873654563159724);
  EXPECT_DOUBLE_EQ(out[6], 0.0021770298651935666);
  EXPECT_DOUBLE_EQ(out[8], 0.0024676761791649136);
  EXPECT_DOUBLE_EQ(out[9], 0.001582373650788027);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "fast";
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN

