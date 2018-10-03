// License: BSD 3 clause

#include <cstdlib>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <random>

#include <cereal/archives/json.hpp>
#include <cereal/archives/binary.hpp>

#include <gtest/gtest.h>

#define DEBUG_COSTLY_THROW 1


#include "tick/array/array.h"
#include "tick/array/dot.h"
#include "tick/base/base.h"
#include "tick/base/serialization.h"

#define TICK_TEST_ROW_SIZE (10)
#define TICK_TEST_COLUMN_SIZE (8)
#define TICK_TEST_DATA_SIZE (100)

/**
 * Test arrays a filled with random data ranging between these two values
 *
 * For unsigned integer arrays (of any size) the minimum value is 0.
 */
#define TICK_TEST_DATA_MIN_VALUE -100
#define TICK_TEST_DATA_MAX_VALUE  100

// this is very low but these operations are not always precise
#define TICK_TEST_SINGLE_RELATIVE_ERROR 5e-2
#define TICK_TEST_DOUBLE_RELATIVE_ERROR 1e-11

namespace {

/**
 * Function to get the minimum test data value for signed integers
 */
template<typename T>
constexpr T GetTestMinimum(typename std::enable_if<std::is_signed<T>::value>::type * = 0) { return TICK_TEST_DATA_MIN_VALUE; }

/**
 * Function to get the minimum test data value for unsigned integers
 */
template<typename T>
constexpr T GetTestMinimum(typename std::enable_if<std::is_unsigned<T>::value>::type * = 0) { return 0; }

template <typename T>
constexpr double GetAcceptedRelativeError(typename std::enable_if<std::is_integral<T>::value>::type * = 0) { return 0; }

template <typename T>
constexpr double GetAcceptedRelativeError(typename std::enable_if<std::is_same<T, float>::value>::type * = 0) { return TICK_TEST_SINGLE_RELATIVE_ERROR; }

template <typename T>
constexpr double GetAcceptedRelativeError(typename std::enable_if<std::is_same<T, double>::value>::type * = 0) { return TICK_TEST_DOUBLE_RELATIVE_ERROR; }

#define EXPECT_RELATIVE_ERROR(type, actual, expected)                       \
  {                                                                         \
    const double relE =                                                     \
        std::fabs((expected - actual) /                                     \
                  static_cast<double>(                                      \
                      expected == 0 ? std::numeric_limits<float>::epsilon() \
                                    : expected));                           \
    EXPECT_LE(relE, ::GetAcceptedRelativeError<type>());                    \
  }
#define ASSERT_RELATIVE_ERROR(type, actual, expected)                       \
  {                                                                         \
    const double relE =                                                     \
        std::fabs((expected - actual) /                                     \
                  static_cast<double>(                                      \
                      expected == 0 ? std::numeric_limits<float>::epsilon() \
                                    : expected));                           \
    ASSERT_LE(relE, ::GetAcceptedRelativeError<type>());                    \
  }

std::random_device rd;
std::mt19937 gen(rd());

template<typename ArrType>
ArrType GenerateRandomArray(ulong n = TICK_TEST_DATA_SIZE,
                            typename std::enable_if<std::is_floating_point<typename ArrType::value_type>::value>::type * = 0) {
  ArrType res(n);

  using nl = std::numeric_limits<typename ArrType::value_type>;
  std::uniform_real_distribution<> dis(TICK_TEST_DATA_MIN_VALUE, TICK_TEST_DATA_MAX_VALUE);

  for (ulong i = 0; i < res.size(); ++i) res[i] = dis(gen);

  return res;
}

template<typename ArrType>
ArrType GenerateRandomArray(ulong n = TICK_TEST_DATA_SIZE,
                            typename std::enable_if<std::is_integral<typename ArrType::value_type>::value>::type * = 0) {
  ArrType res(n);

  using value_type = typename ArrType::value_type;
  using nl = std::numeric_limits<value_type>;
  std::uniform_int_distribution<> dis(GetTestMinimum<value_type>(), TICK_TEST_DATA_MAX_VALUE);

  for (ulong i = 0; i < res.size(); ++i) res[i] = dis(gen);

  return res;
}

template<typename Arr2dType>
Arr2dType GenerateRandomArray2d(ulong n_rows = TICK_TEST_ROW_SIZE,
                                ulong n_cols = TICK_TEST_COLUMN_SIZE) {
  Arr2dType random_2d_array = Arr2dType(n_rows, n_cols);

  typedef Array<typename Arr2dType::value_type> Array1dType;
  Array1dType random_array = GenerateRandomArray<Array1dType>(n_rows * n_cols);

  std::copy(random_array.data(), random_array.data() + random_array.size(), random_2d_array.data());

  return random_2d_array;
}

}  // namespace


template <typename ArrType>
class LinearSystemTest : public ::testing::Test {
 public:
  using value_type = typename ArrType::value_type;
};

typedef ::testing::Types<ArrayFloat, ArrayDouble> MyArrayTypes;
TYPED_TEST_CASE(LinearSystemTest, MyArrayTypes);



TYPED_TEST(LinearSystemTest, ToyLinearSystem) {
  using T = typename TypeParam::value_type;

  ulong size = 2;

  Array<T> b {2, 1};
  Array<T> b_copy = b;
  // Array is column major for blas, the system we are solving is:
  // 5 3  *  1  =  2
  // 2 1    -1     1
  Array<T> arrAData {5, 2, 3, 1};
  Array2d<T> arrA(size, size, arrAData.data());
  Array2d<T> arrA_copy = arrA;

  tick::vector_operations<T>{}.solve_linear_system(size, arrA.data(), b.data());

  for (ulong i = 0; i < size; ++i) {
    T b_sol_i = 0;
    for (ulong j = 0; j < size; ++j) {
      // fake a col major array
      b_sol_i += arrA_copy(j, i) * b[j];
    }
    EXPECT_RELATIVE_ERROR(T, b_sol_i, b_copy[i]);
  }

  b = b_copy;
  arrA = arrA_copy;
  tick::detail::vector_operations_unoptimized<T>{}.solve_linear_system(size, arrA.data(), b.data());

  for (ulong i = 0; i < size; ++i) {
    T b_sol_i = 0;
    for (ulong j = 0; j < size; ++j) {
      // fake a col major array
      b_sol_i += arrA_copy(j, i) * b[j];
    }
    EXPECT_RELATIVE_ERROR(T, b_sol_i, b_copy[i]);
  }
}


TYPED_TEST(LinearSystemTest, LinearSystem) {
  using T = typename TypeParam::value_type;

  ulong size = 10;
  Array<T> b = ::GenerateRandomArray<Array<T>>(size);
  Array<T> b_copy = b;
  Array2d<T> arrA = ::GenerateRandomArray2d<Array2d<T>>(size, size);
  Array2d<T> arrA_copy = arrA;

  tick::vector_operations<T>{}.solve_linear_system(size, arrA.data(), b.data());

  for (ulong i = 0; i < size; ++i) {
    T b_sol_i = 0;
    for (ulong j = 0; j < size; ++j) {
      // fake a col major array
      b_sol_i += arrA_copy(j, i) * b[j];
    }
    EXPECT_RELATIVE_ERROR(T, b_sol_i, b_copy[i]);
  }

  b = b_copy;
  arrA = arrA_copy;
  tick::detail::vector_operations_unoptimized<T>{}.solve_linear_system(size, arrA.data(), b.data());

  for (ulong i = 0; i < size; ++i) {
    T b_sol_i = 0;
    for (ulong j = 0; j < size; ++j) {
      // fake a col major array
      b_sol_i += arrA_copy(j, i) * b[j];
    }
    EXPECT_RELATIVE_ERROR(T, b_sol_i, b_copy[i]);
  }
}


TYPED_TEST(LinearSystemTest, SymmetricLinearSystem) {
  using T = typename TypeParam::value_type;

  ulong size = 2;
  Array<T> b = ::GenerateRandomArray<Array<T>>(size);
  Array<T> b_copy = b;
  Array2d<T> sqrt_arrA = ::GenerateRandomArray2d<Array2d<T>>(size, size);
  Array2d<T> arrA(size, size);
  arrA.init_to_zero();
  // make it positive symmetric by computing its square B = A A'
  for (ulong i = 0; i < size; ++i) {
    for (ulong j = 0; j < size; ++j) {
      for (int k = 0; k < size; ++k) {
        arrA(i, j) += sqrt_arrA(i, k) * sqrt_arrA(j, k);
      }
    }
  }
  Array2d<T> arrA_copy = arrA;

  // Solve it with symmetric solver
  tick::vector_operations<T>{}.solve_positive_symmetric_linear_system(
      size, arrA.data(), b.data(), nullptr, 1);

  for (ulong i = 0; i < size; ++i) {
    T b_sol_i = 0;
    for (ulong j = 0; j < size; ++j) {
      // fake a col major array
      b_sol_i += arrA_copy(j, i) * b[j];
    }
    EXPECT_RELATIVE_ERROR(T, b_sol_i, b_copy[i]);
  }


  b = b_copy;
  arrA = arrA_copy;
  // Solve it switching to basic linear system
  tick::vector_operations<T>{}.solve_positive_symmetric_linear_system(
      size, arrA.data(), b.data(), nullptr, size+1);

  for (ulong i = 0; i < size; ++i) {
    T b_sol_i = 0;
    for (ulong j = 0; j < size; ++j) {
      // fake a col major array
      b_sol_i += arrA_copy(j, i) * b[j];
    }
    EXPECT_RELATIVE_ERROR(T, b_sol_i, b_copy[i]);
  }
}

#ifdef ADD_MAIN
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
