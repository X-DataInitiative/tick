// License: BSD 3 clause

#include <algorithm>
#include <complex>
#include <fstream>
#include <numeric>

#define DEBUG_COSTLY_THROW 1
#define TICK_TEST_DATA_SIZE (1000)

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>

#include <gtest/gtest.h>

#include "tick/array/array.h"

#include "tick/linear_model/model_hinge.h"
#include "tick/linear_model/model_linreg.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_poisreg.h"
#include "tick/linear_model/model_quadratic_hinge.h"
#include "tick/linear_model/model_smoothed_hinge.h"

#include "tick/robust/model_absolute_regression.h"
#include "tick/robust/model_epsilon_insensitive.h"
#include "tick/robust/model_generalized_linear_with_intercepts.h"
#include "tick/robust/model_huber.h"
#include "tick/robust/model_linreg_with_intercepts.h"
#include "tick/robust/model_modified_huber.h"

#include "tick/prox/prox_binarsity.h"
#include "tick/prox/prox_elasticnet.h"
#include "tick/prox/prox_equality.h"
#include "tick/prox/prox_group_l1.h"
#include "tick/prox/prox_l1.h"
#include "tick/prox/prox_l1w.h"
#include "tick/prox/prox_l2.h"
#include "tick/prox/prox_l2sq.h"
#include "tick/prox/prox_positive.h"
#include "tick/prox/prox_slope.h"
#include "tick/prox/prox_sorted_l1.h"
#include "tick/prox/prox_tv.h"
#include "tick/prox/prox_zero.h"

namespace testing {
namespace internal {
enum GTestColor { COLOR_DEFAULT, COLOR_RED, COLOR_GREEN, COLOR_YELLOW };
#if defined(_WIN32) || defined(__APPLE__)
void ColoredPrintf(GTestColor color, const char *fmt, ...) {}
#else
extern void ColoredPrintf(GTestColor color, const char *fmt, ...);
#endif
}  // namespace internal
}  // namespace testing
#define PRINTF(...)                                                   \
  do {                                                                \
    testing::internal::ColoredPrintf(testing::internal::COLOR_GREEN,  \
                                     "[          ] ");                \
    testing::internal::ColoredPrintf(testing::internal::COLOR_YELLOW, \
                                     __VA_ARGS__);                    \
  } while (0)

// C++ stream interface
class TestCout : public std::stringstream {
 public:
  ~TestCout() { PRINTF("%s", str().c_str()); }
};

#define TEST_COUT TestCout()

namespace {

template <class T>
std::shared_ptr<BaseArray2d<T> > get_features() {
  ulong n_samples = 7;
  ulong n_features = 5;
  Array<T> features_data{
      (T)0.55,  (T)2.04,  (T)0.78,  (T)-0.00, (T)0.00,  (T)-0.00, (T)-2.62,
      (T)-0.00, (T)0.00,  (T)0.31,  (T)-0.64, (T)0.94,  (T)0.00,  (T)0.55,
      (T)-0.14, (T)0.93,  (T)0.00,  (T)0.00,  (T)-0.00, (T)-2.39, (T)1.13,
      (T)0.05,  (T)-1.50, (T)-0.50, (T)-1.41, (T)1.41,  (T)1.10,  (T)-0.00,
      (T)0.12,  (T)0.00,  (T)-0.00, (T)-1.33, (T)-0.00, (T)0.85,  (T)3.03};
  Array2d<T> features(n_samples, n_features);
  for (int i = 0; i < features_data.size(); ++i) features[i] = features_data[i];
  return features.as_sarray2d_ptr();
}

template <class T>
std::shared_ptr<SArray<T> > get_labels() {
  Array<T> labels{(T)-1.76, (T)2.6,   (T)-0.7, (T)-1.84,
                  (T)-1.88, (T)-1.78, (T)2.52};
  return labels.as_sarray_ptr();
}

template <class T>
std::shared_ptr<SArray<T> > get_proxl1w_weights() {
  return Array<T>{(T)1.76, (T)2.6, (T)0.7, (T)1.84, (T)1.88}.as_sarray_ptr();
}

::testing::AssertionResult IsTrue(BoolStrReport &&b) {
  if (!b) return ::testing::AssertionFailure() << b.why();
  return ::testing::AssertionSuccess();
}

template <class T, typename InputArchive, typename OutputArchive>
class Wrapper {
 public:
  template <class SOLVER, class MODEL, class PROX>
  static void TestSerializing(const std::shared_ptr<SOLVER> solver_ptr,
                              const std::shared_ptr<MODEL> model_ptr,
                              const std::shared_ptr<PROX> prox_ptr) {
    Array<T> coeffs({(T)-2, (T)5.2}), out_grad(2);
    model_ptr->grad(coeffs, out_grad);
    double lip_max = 0;
    try {
      lip_max = model_ptr->get_lip_max();
    } catch (const std::runtime_error &e) {
    };
    (void)lip_max;
    solver_ptr->set_model(model_ptr);
    solver_ptr->set_prox(prox_ptr);
    std::stringstream os;
    {
      OutputArchive outputArchive(os);
      outputArchive(*solver_ptr.get());
    }
    {
      InputArchive inputArchive(os);
      auto null_solver_ptr = SOLVER::AS_NULL();
      SOLVER &re_solver = *null_solver_ptr.get();
      inputArchive(re_solver);

      MODEL &re_model(dynamic_cast<MODEL &>(*re_solver.get_model().get()));
      PROX &re_prox(dynamic_cast<PROX &>(*re_solver.get_prox().get()));

      ASSERT_TRUE(IsTrue((*model_ptr.get() == re_model)));
      ASSERT_TRUE(IsTrue((*prox_ptr.get() == re_prox)));
      ASSERT_TRUE(IsTrue((*solver_ptr.get() == re_solver)));

      Array<T> out_grad_restored(2);
      re_model.grad(coeffs, out_grad_restored);
      for (ulong i = 0; i < out_grad.size(); ++i)
        ASSERT_DOUBLE_EQ(out_grad[i], out_grad_restored[i]);
      double lip_max_restored = 0;
      try {
        lip_max_restored = re_model.get_lip_max();
      } catch (const std::runtime_error &e) {
      };
      EXPECT_DOUBLE_EQ(lip_max, lip_max_restored);
    }
    TEST_COUT << "Succeeded with: " << typeid(SOLVER).name() << " : "
              << typeid(MODEL).name() << " : " << typeid(PROX).name()
              << std::endl;
  }
};

/**
 COMPILER TEMPLATE BLACK MAGIC - creates N permutations of templated functions
*/
#define WITH_PROX(...) do_with<T>(solver, model, #__VA_ARGS__, __VA_ARGS__)
template <typename T1, typename H1, class SOLVER, class MODEL>
void do_with(const std::shared_ptr<SOLVER> solver,
             const std::shared_ptr<MODEL> model, const char *label, H1 &&prox) {
  ::Wrapper<T1, cereal::PortableBinaryInputArchive,
            cereal::PortableBinaryOutputArchive>::TestSerializing(solver, model,
                                                        std::forward<H1>(prox));
}
template <typename T1, typename H1, class SOLVER, class MODEL, typename... T>
void do_with(const std::shared_ptr<SOLVER> solver,
             const std::shared_ptr<MODEL> model, const char *label, H1 &&value,
             T &&... rest) {
  const char *pcomma = strchr(label, ',');
  do_with<T1>(solver, model, pcomma + 1, std::forward<H1>(value));
  do_with<T1>(solver, model, pcomma + 1, std::forward<T>(rest)...);
}

}  // namespace
