// License: BSD 3 clause

#include <numeric>
#include <algorithm>
#include <complex>

#define DEBUG_COSTLY_THROW 1
#define TICK_TEST_DATA_SIZE (1000)

#include <gtest/gtest.h>

#include "tick/array/array.h"
#include "tick/optim/model/linreg.h"

#include <cereal/types/unordered_map.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

SArrayDoublePtr get_labels() {
  ArrayDouble labels{-1.76, 2.6, -0.7, -1.84, -1.88, -1.78, 2.52};
  return labels.as_sarray_ptr();
}

SArrayDouble2dPtr get_features() {
  ulong n_samples = 7;
  ulong n_features = 5;

  ArrayDouble features_data{0.55, 2.04, 0.78, -0.00, 0.00,
                            -0.00, -2.62, -0.00, 0.00, 0.31,
                            -0.64, 0.94, 0.00, 0.55, -0.14,
                            0.93, 0.00, 0.00, -0.00, -2.39,
                            1.13, 0.05, -1.50, -0.50, -1.41,
                            1.41, 1.10, -0.00, 0.12, 0.00,
                            -0.00, -1.33, -0.00, 0.85, 3.03};

  ArrayDouble2d features(n_samples, n_features);
  for (int i = 0; i < features_data.size(); ++i) {
    features[i] = features_data[i];
  }
  return features.as_sarray2d_ptr();
}

TEST(Model, PartialVsFull) {
  ArrayDouble y(3);
  y[0] = -2;
  y[1] = 3;
  y[2] = 1.5;
  ArrayDouble2d x(3, 2);
  x[0] = -2;
  x[1] = 5.2;
  x[2] = 1.8;
  x[3] = 1;
  x[4] = 2.2;
  x[5] = 1.9;

  SArrayDoublePtr labels = y.as_sarray_ptr();
  SArrayDouble2dPtr features = x.as_sarray2d_ptr();

  ModelLinReg model(features, labels, false, 2);

  ArrayDouble coeffs(2);
  coeffs[0] = -2;
  coeffs[1] = 5.2;

  ArrayDouble out_grad(2);
  ArrayDouble sum_grad(2);
  sum_grad.init_to_zero();

  model.grad_i(0, coeffs, out_grad);
  sum_grad.mult_incr(out_grad, 1);

  model.grad_i(1, coeffs, out_grad);
  sum_grad.mult_incr(out_grad, 1);

  model.grad_i(2, coeffs, out_grad);
  sum_grad.mult_incr(out_grad, 1);

  sum_grad /= 3;

  model.grad(coeffs, out_grad);

  for (unsigned int j = 0; j < sum_grad.size(); ++j)
    EXPECT_FLOAT_EQ(sum_grad.data()[j], out_grad.data()[j]);
}

namespace {

template<typename InputArchive, typename OutputArchive>
void TestModelLinRegSerialization() {
  SArrayDoublePtr labels = get_labels();
  SArrayDouble2dPtr features = get_features();

  ModelLinReg model(features, labels, false, 1);

  ArrayDouble coeffs{-2, 5.2, 4., 0., -1};

  ArrayDouble out_grad(model.get_n_coeffs());
  model.grad(coeffs, out_grad);
  const double lip_max = model.get_lip_max();

  std::stringstream os;
  {
    OutputArchive outputArchive(os);

    outputArchive(model);
  }

  {
    InputArchive inputArchive(os);

    ModelLinReg restored_model(nullptr, nullptr, false);
    inputArchive(restored_model);

    ArrayDouble out_grad_restored(model.get_n_coeffs());
    restored_model.grad(coeffs, out_grad_restored);

    for (ulong i = 0; i < out_grad.size(); ++i)
      ASSERT_DOUBLE_EQ(out_grad[i], out_grad_restored[i]);

    const double lip_max_restored = restored_model.get_lip_max();
    EXPECT_DOUBLE_EQ(lip_max, lip_max_restored);
  }
};

template<typename InputArchive, typename OutputArchive>
void TestModelLinRegPtrSerialization() {
  SArrayDoublePtr labels = get_labels();
  SArrayDouble2dPtr features = get_features();

  ModelPtr model = std::make_shared<ModelLinReg>(features, labels, false, 1);

  ArrayDouble coeffs{-2, 5.2, 4., 0., -1};

  ArrayDouble out_grad(model->get_n_coeffs());
  model->grad(coeffs, out_grad);
  const double lip_max = model->get_lip_max();

  std::stringstream os;
  {
    OutputArchive outputArchive(os);

    outputArchive(model);
  }

  {
    InputArchive inputArchive(os);

    ModelPtr restored_model;
    inputArchive(restored_model);

    ArrayDouble out_grad_restored(model->get_n_coeffs());
    restored_model->grad(coeffs, out_grad_restored);

    for (ulong i = 0; i < out_grad.size(); ++i)
      ASSERT_DOUBLE_EQ(out_grad[i], out_grad_restored[i]);

    const double lip_max_restored = restored_model->get_lip_max();
    EXPECT_DOUBLE_EQ(lip_max, lip_max_restored);
    EXPECT_EQ(model->get_n_coeffs(), restored_model->get_n_coeffs());
    EXPECT_EQ(model->get_n_samples(), restored_model->get_n_samples());
  }
};

}  // namespace

TEST(Model, SerializationJSON) {
  SCOPED_TRACE("");
  ::TestModelLinRegSerialization<cereal::JSONInputArchive, cereal::JSONOutputArchive>();
}

TEST(Model, SerializationBinary) {
  SCOPED_TRACE("");
  ::TestModelLinRegSerialization<cereal::BinaryInputArchive, cereal::BinaryOutputArchive>();
}


TEST(ModelPtr, SerializationJSON) {
  SCOPED_TRACE("");
  ::TestModelLinRegPtrSerialization<cereal::JSONInputArchive, cereal::JSONOutputArchive>();
}

TEST(ModelPtr, SerializationBinary) {
  SCOPED_TRACE("");
  ::TestModelLinRegPtrSerialization<cereal::BinaryInputArchive, cereal::BinaryOutputArchive>();
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
