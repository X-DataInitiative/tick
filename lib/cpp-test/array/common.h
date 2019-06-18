// License: BSD 3 clause

#include <limits>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <type_traits>

#include <gtest/gtest.h>

#define DEBUG_COSTLY_THROW 1

#define TICK_TEST_HALF_RELATIVE_ERROR 1e-3  // TODO find the real one

#ifndef TICK_TEST_SINGLE_RELATIVE_ERROR
#define TICK_TEST_SINGLE_RELATIVE_ERROR 1e-4
#endif

#ifndef TICK_TEST_DOUBLE_RELATIVE_ERROR
#define TICK_TEST_DOUBLE_RELATIVE_ERROR 1e-13
#endif

#include "tick/array/dot.h"
#include "tick/base/serialization.h"

namespace {

/**
 * Function to get the minimum test data value for signed integers
 */
template <typename T>
constexpr T GetTestMinimum(
    typename std::enable_if<std::is_signed<T>::value>::type * = 0) {
  return TICK_TEST_DATA_MIN_VALUE;
}

/**
 * Function to get the minimum test data value for unsigned integers
 */
template <typename T>
constexpr T GetTestMinimum(
    typename std::enable_if<std::is_unsigned<T>::value>::type * = 0) {
  return 0;
}

template <typename T>
constexpr double GetAcceptedRelativeError(
    typename std::enable_if<std::is_integral<T>::value>::type * = 0) {
  return 0;
}

template <typename T>
constexpr float GetAcceptedRelativeError(
    typename std::enable_if<std::is_same<T, half_float::half>::value>::type * = 0) {
  return TICK_TEST_HALF_RELATIVE_ERROR;
}

template <typename T>
constexpr float GetAcceptedRelativeError(
    typename std::enable_if<std::is_same<T, half_float::detail::expr>::value>::type * = 0) {
  return TICK_TEST_HALF_RELATIVE_ERROR;
}

template <typename T>
constexpr double GetAcceptedRelativeError(
    typename std::enable_if<std::is_same<T, float>::value>::type * = 0) {
  return TICK_TEST_SINGLE_RELATIVE_ERROR;
}

template <typename T>
constexpr double GetAcceptedRelativeError(
    typename std::enable_if<std::is_same<T, double>::value>::type * = 0) {
  return TICK_TEST_DOUBLE_RELATIVE_ERROR;
}

std::random_device rd;
std::mt19937 gen(rd());

template <typename ArrType>
ArrType GenerateRandomArray(
    ulong n = TICK_TEST_DATA_SIZE,
    typename std::enable_if<
        std::is_floating_point<typename ArrType::value_type>::value ||
        std::is_same<typename ArrType::value_type, half_float::half>::value>::type * = 0) {
  ArrType res(n);

  int64_t max = TICK_TEST_DATA_MIN_VALUE, min = TICK_TEST_DATA_MAX_VALUE;
  if (std::is_same<typename ArrType::value_type, half_float::half>()) {
    min = -100;
    max = 100;
  }

  std::uniform_real_distribution<> dis(min, max);

  for (ulong i = 0; i < res.size(); ++i) res[i] = dis(gen);
  return res;
}

template <typename ArrType>
ArrType GenerateRandomArray(
    ulong n = TICK_TEST_DATA_SIZE,
    typename std::enable_if<
        std::is_integral<typename ArrType::value_type>::value>::type * = 0) {
  ArrType res(n);

  using value_type = typename ArrType::value_type;
  std::uniform_int_distribution<> dis(GetTestMinimum<value_type>(),
                                      TICK_TEST_DATA_MAX_VALUE);

  for (ulong i = 0; i < res.size(); ++i) res[i] = dis(gen);

  return res;
}

template <typename Arr2dType>
Arr2dType GenerateRandomArray2d(ulong n_rows = TICK_TEST_ROW_SIZE,
                                ulong n_cols = TICK_TEST_COLUMN_SIZE) {
  Arr2dType random_2d_array = Arr2dType(n_rows, n_cols);

  typedef Array<typename Arr2dType::value_type> Array1dType;
  Array1dType random_array = GenerateRandomArray<Array1dType>(n_rows * n_cols);

  std::copy(random_array.data(), random_array.data() + random_array.size(),
            random_2d_array.data());

  return random_2d_array;
}

}  // namespace

#define EXPECT_RELATIVE_ERROR(type, actual, expected)               \
  {                                                                 \
    float div = static_cast<double>(expected);                      \
    if (expected == 0) div = std::numeric_limits<float>::epsilon(); \
    const double relE = std::fabs((expected - actual) / div);       \
    EXPECT_LE(relE, ::GetAcceptedRelativeError<type>());            \
  }

#define ASSERT_RELATIVE_ERROR(type, actual, expected)               \
  {                                                                 \
    float div = static_cast<double>(expected);                      \
    if (expected == 0) div = std::numeric_limits<float>::epsilon(); \
    const double relE = std::fabs((expected - actual) / div);       \
    ASSERT_LE(relE, ::GetAcceptedRelativeError<type>());            \
  }
