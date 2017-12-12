// License: BSD 3 clause

#include <numeric>
#include <algorithm>
#include <complex>

#define DEBUG_COSTLY_THROW 1
#define TICK_TEST_DATA_SIZE (1000)

#include <gtest/gtest.h>

#include "tick/array/array.h"
#include "tick/linear_model/model_linreg.h"

#include <cereal/types/unordered_map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>
#include <fstream>

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

template <typename InputArchive, typename OutputArchive>
void TestModelLinRegSerialization() {
  ArrayDouble y({-2, 3, 1.5, 1, 0.8});
  ArrayDouble2d x(5, 2);
  x[0] = -2;
  x[1] = 5.2;
  x[2] = 1.8;
  x[3] = 1;
  x[4] = 2.2;
  x[5] = 1.9;
  x[6] = 1;
  x[7] = 2.2;
  x[8] = 1.9;
  x[9] = 1.9;

  SArrayDoublePtr labels = y.as_sarray_ptr();
  SArrayDouble2dPtr features = x.as_sarray2d_ptr();

  ModelLinReg model(features, labels, false, 1);

  ArrayDouble coeffs({-2, 5.2});

  ArrayDouble out_grad(2);
  model.grad(coeffs, out_grad);
  const double lip_max = model.get_lip_max();

  std::stringstream os;
  {
    OutputArchive outputArchive(os);

    outputArchive( model );
  }

  {
    InputArchive inputArchive(os);

    ModelLinReg restored_model(nullptr, nullptr, false);
    inputArchive( restored_model );

    ArrayDouble out_grad_restored(2);
    restored_model.grad(coeffs, out_grad_restored);

    for (ulong i = 0; i < out_grad.size(); ++i) ASSERT_DOUBLE_EQ(out_grad[i], out_grad_restored[i]);

    const double lip_max_restored = restored_model.get_lip_max();
    EXPECT_DOUBLE_EQ(lip_max, lip_max_restored);
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


#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
