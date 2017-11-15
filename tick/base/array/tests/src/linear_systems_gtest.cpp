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


#include <array.h>
#include <dot.h>
#include <base.h>

#define TICK_TEST_ROW_SIZE (10)
#define TICK_TEST_COLUMN_SIZE (8)
#define TICK_TEST_DATA_SIZE (100)

/**
 * Test arrays a filled with random data ranging between these two values
 *
 * For unsigned integer arrays (of any size) the minimum value is 0.
 */
#define TICK_TEST_DATA_MIN_VALUE -10000
#define TICK_TEST_DATA_MAX_VALUE  10000

#define TICK_TEST_SINGLE_RELATIVE_ERROR 1e-4
#define TICK_TEST_DOUBLE_RELATIVE_ERROR 1e-13

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

#if defined(__APPLE__)
TEST(Array2dTest, LinearColumnMajorSystem) {
  ulong size = 2;
//  ArrayDouble b = ::GenerateRandomArray<ArrayDouble>(size);
  ArrayDouble b {2, 1};
  ArrayDouble b_copy = b;
//  ArrayDouble2d arrA = ::GenerateRandomArray2d<ArrayDouble2d>(size, size);
  ArrayDouble2d arrA = ::GenerateRandomArray2d<ArrayDouble2d>(size, size);
  arrA[2] = 3; arrA[3] = 2;
  arrA[0] = 5; arrA[1] = 3;
  ArrayDouble2d arrA_copy = arrA;
  arrA.solve_linear(b);

  b.print();

  for (ulong i = 0; i < size; ++i) {
    double b_sol_i = view_row(arrA_copy, i).dot(b);
    ASSERT_DOUBLE_EQ(b_sol_i, b_copy[i]);
  }
}

TEST(Array2dTest, LinearColumnMajorSystem2) {
  ulong size = 3;
//  ArrayDouble b = ::GenerateRandomArray<ArrayDouble>(size);
  ArrayDouble b {1487.06,5851.89,1590.46};
  ArrayDouble b_copy = b;
  ArrayDouble arrAData {-100.744,-0.597835,-0.714557,
                        -0.597835,-44600.7,-0.66181,
                        -0.714557,-0.66181,-100.82};
  ArrayDouble2d arrA(size, size, arrAData.data());

  TICK_DEBUG();
  arrA.print();
  b.print();


  ArrayDouble2d arrA_copy = arrA;
//  arrA.solve_linear(b);
  tick::vector_operations<double>{}.solve_linear_system(size, arrA.data(), b.data());

  b.print();

  for (ulong i = 0; i < size; ++i) {
    double b_sol_i = view_row(arrA_copy, i).dot(b);
    ASSERT_DOUBLE_EQ(b_sol_i, b_copy[i]);
  }
}

TEST(Array2dTest, SymmetricLinearSystem) {
  ulong size = 2;
  ArrayDouble b = ::GenerateRandomArray<ArrayDouble>(size);
//  ArrayDouble b {2, 1};
  ArrayDouble b_copy = b;
//  ArrayDouble2d arrA = ::GenerateRandomArray2d<ArrayDouble2d>(size, size);
  ArrayDouble2d arrA(size, size);
  arrA[0] = 5; arrA[1] = 3;
  arrA[2] = 3; arrA[3] = 2;

  ArrayDouble2d arrA2d = arrA;
//  ulong c = 0;
  for (ulong i = 0; i < size; ++i) {
    for (ulong j = 0; j < i; ++j) {
      arrA(i, j) = 0;
      arrA2d(i, j) = arrA(j, i);
    }
  }
//  for (ulong i = 0; i < size; ++i) {
//    for (ulong j = 0; j < i; ++j) {
//      arrA2d(i, j) = arrA2d(j, i);
//    }
//  }
//  arrA.print();
//  arrA2d.print();

//  b_copy.print();
//
  tick::vector_operations<double>{}.solve_symmetric_linear_system(size, arrA.data(), b.data());

//  b.print();
  for (ulong i = 0; i < size; ++i) {
    double b_sol_i = view_row(arrA2d, i).dot(b);
    ASSERT_FLOAT_EQ(b_sol_i, b_copy[i]);
  }
}
#endif