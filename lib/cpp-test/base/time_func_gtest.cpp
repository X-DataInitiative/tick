// License: BSD 3 clause

#include <gtest/gtest.h>
#include "tick/base/time_func.h"
#include <iostream>
#include <cmath>

class TimeFunctionTest : public ::testing::Test {
 protected:
  ArrayDouble T;
  ArrayDouble Y;
  double dt = .25;
  double time_horizon = 1.;
  double border_value = .0;
  ulong sample_size = 5;
  void SetUp() override {
    T = ArrayDouble{.0, .25, .5, .75, 1.};
    Y = ArrayDouble{1., 2., 3., 4., 5.};
  }
};

TEST_F(TimeFunctionTest, implicit_abscissa_data) {
  TimeFunction tf(Y, TimeFunction::BorderType::Border0, TimeFunction::InterMode::InterConstRight,
                  dt, border_value);
  EXPECT_DOUBLE_EQ(tf.get_t0(), T[0]);
  SArrayDoublePtr sampled_y = tf.get_sampled_y();
  EXPECT_EQ(tf.get_sample_size(), sample_size);
  EXPECT_EQ(sampled_y->size(), Y.size());
  for (ulong i = 0; i < Y.size(); i++) {
    EXPECT_DOUBLE_EQ((*sampled_y)[i], Y[i]);
  }
}
TEST_F(TimeFunctionTest, explicit_abscissa_data) {
  TimeFunction tf(T, Y, TimeFunction::BorderType::Border0,
                  TimeFunction::InterMode::InterConstRight);
  EXPECT_DOUBLE_EQ(tf.get_t0(), T[0]);
  SArrayDoublePtr sampled_y = tf.get_sampled_y();
  EXPECT_EQ(tf.get_sample_size(), sample_size);
  EXPECT_EQ(sampled_y->size(), Y.size());
  for (ulong i = 0; i < Y.size(); i++) {
    EXPECT_DOUBLE_EQ((*sampled_y)[i], Y[i]);
  }
}

TEST_F(TimeFunctionTest, border0_interconstright_implicit_node_values) {
  TimeFunction tf(Y, TimeFunction::BorderType::Border0, TimeFunction::InterMode::InterConstRight,
                  dt, border_value);
  double s = 0;
  for (int k = 0; k < T.size(); k++) {
    double t_k = T[k];
    double y_k = Y[k];
    EXPECT_DOUBLE_EQ(tf.value(t_k), y_k) << "error at k=" << k << ", t_k=" << t_k << "\n";
    if (k > 0) s += Y[k - 1] * dt;
    EXPECT_DOUBLE_EQ(tf.primitive(t_k), s) << "error at k=" << k << ", t_k=" << t_k << "\n";
  }

  EXPECT_DOUBLE_EQ(tf.get_norm(), tf.primitive(time_horizon));
}

TEST_F(TimeFunctionTest, border0_interconstright_explicit_node_values) {
  TimeFunction tf(T, Y, TimeFunction::BorderType::Border0,
                  TimeFunction::InterMode::InterConstRight);
  double s = 0;
  for (int k = 0; k < T.size(); k++) {
    double t_k = T[k];
    double y_k = Y[k];
    EXPECT_DOUBLE_EQ(tf.value(t_k), y_k) << "error at k=" << k << ", t_k=" << t_k << "\n";
    if (k > 0) s += Y[k - 1] * dt;
    EXPECT_DOUBLE_EQ(tf.primitive(t_k), s) << "error at k=" << k << ", t_k=" << t_k << "\n";
  }

  EXPECT_DOUBLE_EQ(tf.get_norm(), tf.primitive(time_horizon));
}

TEST_F(TimeFunctionTest, border0_interconstleft_implicit_node_values) {
  TimeFunction tf(Y, TimeFunction::BorderType::Border0, TimeFunction::InterMode::InterConstLeft, dt,
                  border_value);
  double s = 0;
  for (int k = 0; k < T.size(); k++) {
    double t_k = T[k];
    double y_k = Y[k];
    EXPECT_DOUBLE_EQ(tf.value(t_k), y_k) << "error at k=" << k << ", t_k=" << t_k << "\n";
    if (k > 0) s += y_k * dt;
    EXPECT_DOUBLE_EQ(tf.primitive(t_k), s) << "error at k=" << k << ", t_k=" << t_k << "\n";
  }
  EXPECT_DOUBLE_EQ(tf.get_norm(), tf.primitive(time_horizon));
}

TEST_F(TimeFunctionTest, border0_interconstleft_explicit_node_values) {
  TimeFunction tf(T, Y, TimeFunction::BorderType::Border0, TimeFunction::InterMode::InterConstLeft);
  double s = 0;
  for (int k = 0; k < T.size(); k++) {
    double t_k = T[k];
    double y_k = Y[k];
    EXPECT_DOUBLE_EQ(tf.value(t_k), y_k) << "error at k=" << k << ", t_k=" << t_k << "\n";
    if (k > 0) s += y_k * dt;
    EXPECT_DOUBLE_EQ(tf.primitive(t_k), s) << "error at k=" << k << ", t_k=" << t_k << "\n";
  }
  EXPECT_DOUBLE_EQ(tf.get_norm(), tf.primitive(time_horizon));
}

TEST_F(TimeFunctionTest, border0_interlinear_implicit_node_values) {
  TimeFunction tf(Y, TimeFunction::BorderType::Border0, TimeFunction::InterMode::InterLinear, dt,
                  border_value);
  double s = 0;
  for (int k = 0; k < T.size(); k++) {
    double t_k = T[k];
    double y_k = Y[k];
    EXPECT_DOUBLE_EQ(tf.value(t_k), y_k) << "error at k=" << k << ", t_k=" << t_k << "\n";
    if (k > 0) s += .5 * (y_k + Y[k - 1]) * dt;
    EXPECT_DOUBLE_EQ(tf.primitive(t_k), s) << "error at k=" << k << ", t_k=" << t_k << "\n";
  }
  EXPECT_DOUBLE_EQ(tf.get_norm(), tf.primitive(time_horizon));
}

TEST_F(TimeFunctionTest, border0_interlinear_explicit_node_values) {
  TimeFunction tf(T, Y, TimeFunction::BorderType::Border0, TimeFunction::InterMode::InterLinear);
  double s = 0;
  for (int k = 0; k < T.size(); k++) {
    double t_k = T[k];
    double y_k = Y[k];
    EXPECT_DOUBLE_EQ(tf.value(t_k), y_k) << "error at k=" << k << ", t_k=" << t_k << "\n";
    if (k > 0) s += .5 * (y_k + Y[k - 1]) * dt;
    EXPECT_DOUBLE_EQ(tf.primitive(t_k), s) << "error at k=" << k << ", t_k=" << t_k << "\n";
  }
  EXPECT_DOUBLE_EQ(tf.get_norm(), tf.primitive(time_horizon));
}
#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
