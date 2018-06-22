// License: BSD 3 clause

#include <cstdlib>

#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>

#include <gtest/gtest.h>

#define DEBUG_COSTLY_THROW 1

/*
 * Size of the test arrays
 */
#define TICK_TEST_ROW_SIZE (10)
#define TICK_TEST_COLUMN_SIZE (8)
#define TICK_TEST_DATA_SIZE (100)

/**
 * Test arrays a filled with random data ranging between these two values
 *
 * For unsigned integer arrays (of any size) the minimum value is 0.
 */
#define TICK_TEST_DATA_MIN_VALUE -10000
#define TICK_TEST_DATA_MAX_VALUE 10000

#define TICK_TEST_SINGLE_RELATIVE_ERROR 1e-4
#define TICK_TEST_DOUBLE_RELATIVE_ERROR 1e-13

#include "tick/array/array.h"
#include "tick/array/dot.h"
#include "tick/base/base.h"
#include "tick/base/serialization.h"

namespace {

template <typename T>
Array<T> tick_array_copy_to_nonatomic(const Array<std::atomic<T>> src) {
  Array<T> copy(src.size());
  for (size_t i = 0; i < src.size(); i++)
    copy[i] = src.template get_data_index<T>(i);
  return copy;
}

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

template <typename ArrType, typename NestedType>
ArrType GenerateRandomArray(
    ulong n = TICK_TEST_DATA_SIZE,
    typename std::enable_if<std::is_floating_point<NestedType>::value>::type * =
        0) {
  ArrType res(n);

  using nl = std::numeric_limits<typename ArrType::value_type>;
  std::uniform_real_distribution<> dis(TICK_TEST_DATA_MIN_VALUE,
                                       TICK_TEST_DATA_MAX_VALUE);

  for (ulong i = 0; i < res.size(); ++i) res.set_data_index(i, dis(gen));

  return res;
}

template <typename ArrType, typename NestedType>
ArrType GenerateRandomArray(
    ulong n = TICK_TEST_DATA_SIZE,
    typename std::enable_if<std::is_integral<NestedType>::value>::type * = 0) {
  ArrType res(n);

  using value_type = typename ArrType::value_type;
  using nl = std::numeric_limits<value_type>;
  std::uniform_int_distribution<> dis(GetTestMinimum<value_type>(),
                                      TICK_TEST_DATA_MAX_VALUE);

  for (ulong i = 0; i < res.size(); ++i) res.set_data_index(i, dis(gen));

  return res;
}

template <typename Arr2dType, typename NestedType>
Arr2dType GenerateRandomArray2d(ulong n_rows = TICK_TEST_ROW_SIZE,
                                ulong n_cols = TICK_TEST_COLUMN_SIZE) {
  Arr2dType random_2d_array = Arr2dType(n_rows, n_cols);

  typedef Array<typename Arr2dType::value_type> Array1dType;
  Array1dType random_array =
      GenerateRandomArray<Array1dType, NestedType>(n_rows * n_cols);

  std::copy(random_array.data(), random_array.data() + random_array.size(),
            random_2d_array.data());

  return random_2d_array;
}

}  // namespace

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

TEST(AtomicArrayTestSetup, RelativeErrors) {
  ASSERT_EQ(GetAcceptedRelativeError<int>(), 0);
  ASSERT_DOUBLE_EQ(GetAcceptedRelativeError<float>(),
                   TICK_TEST_SINGLE_RELATIVE_ERROR);
  ASSERT_DOUBLE_EQ(GetAcceptedRelativeError<double>(),
                   TICK_TEST_DOUBLE_RELATIVE_ERROR);
}

template <typename ArrType>
class AtomicArrayTest : public ::testing::Test {
 public:
  using value_type = typename ArrType::value_type;
};

typedef ::testing::Types<Array<float>, Array<double>> MyArrayTypes;
TYPED_TEST_CASE(AtomicArrayTest, MyArrayTypes);

TYPED_TEST(AtomicArrayTest, InitToZero) {
  TypeParam arr{TICK_TEST_DATA_SIZE};

  arr.init_to_zero();

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr.data()[j], 0.0);
}

TYPED_TEST(AtomicArrayTest, Copy) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  AtomicType arr = ::GenerateRandomArray<AtomicType, VT>();
  AtomicType arrCopy = arr;

  EXPECT_EQ(arr.size(), arrCopy.size());
  EXPECT_NE(arr.data(), arrCopy.data());

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr[j], arrCopy[j]);
}

TYPED_TEST(AtomicArrayTest, Move) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  AtomicType arr = ::GenerateRandomArray<AtomicType, VT>();
  const auto *arrDataPtr = arr.data();

  AtomicType arrCopy = arr;
  AtomicType arrMoved = std::move(arr);

  EXPECT_EQ(0, arr.size());
  EXPECT_EQ(arrCopy.size(), arrMoved.size());
  EXPECT_NE(arr.data(), arrMoved.data());
  EXPECT_EQ(arrDataPtr, arrMoved.data());

  for (ulong j = 0; j < arrMoved.size(); ++j)
    ASSERT_DOUBLE_EQ(arrCopy[j], arrMoved[j]);
}

TYPED_TEST(AtomicArrayTest, View) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  AtomicType arr = ::GenerateRandomArray<AtomicType, VT>(100);

  AtomicType arrView = view(arr);

  EXPECT_EQ(100, arr.size());
  EXPECT_EQ(arrView.size(), arr.size());
  EXPECT_EQ(arr.data(), arrView.data());

  for (ulong j = 0; j < arrView.size(); ++j)
    ASSERT_DOUBLE_EQ(arr[j], arrView[j]);
}

TYPED_TEST(AtomicArrayTest, Bounds) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  AtomicType arr = ::GenerateRandomArray<AtomicType, VT>(100);

  EXPECT_THROW(arr[-1], std::out_of_range);
  EXPECT_THROW(arr[100], std::out_of_range);
  EXPECT_THROW(arr[10000], std::out_of_range);
}

TYPED_TEST(AtomicArrayTest, Size) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  AtomicType arr = ::GenerateRandomArray<AtomicType, VT>(100);

  EXPECT_EQ(arr.size(), 100);
  EXPECT_EQ(arr.size_data(), 100);
}

TYPED_TEST(AtomicArrayTest, InitList) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;

  std::array<VT, 6> vals = {{static_cast<VT>(0.0), static_cast<VT>(1.0),
                             static_cast<VT>(2.0), static_cast<VT>(4.0),
                             static_cast<VT>(8.0), static_cast<VT>(16.0)}};

  AtomicType arr{vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]};

  ASSERT_EQ(arr.size(), 6);

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr[j], vals[j]);
}

TYPED_TEST(AtomicArrayTest, Sum) {
  using K = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<K>>;
  AtomicType arrA = ::GenerateRandomArray<AtomicType, K>();

  tick::promote_t<K> sum = 0;
  for (ulong i = 0; i < arrA.size(); ++i) sum += arrA[i].load();

  EXPECT_DOUBLE_EQ(arrA.sum(), sum);
}

TYPED_TEST(AtomicArrayTest, Min) {
  using K = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<K>>;
  AtomicType arrA = ::GenerateRandomArray<AtomicType, K>();

  K min = TICK_TEST_DATA_MAX_VALUE;
  for (ulong i = 0; i < arrA.size(); ++i) min = std::min(arrA[i].load(), min);

  EXPECT_DOUBLE_EQ(arrA.min(), min);
}

TYPED_TEST(AtomicArrayTest, Max) {
  using K = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<K>>;
  AtomicType arrA = ::GenerateRandomArray<AtomicType, K>();

  K max = TICK_TEST_DATA_MIN_VALUE;
  for (ulong i = 0; i < arrA.size(); ++i) max = std::max(arrA[i].load(), max);

  EXPECT_DOUBLE_EQ(arrA.max(), max);
}

TYPED_TEST(AtomicArrayTest, Fill) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  AtomicType arr(TICK_TEST_DATA_SIZE);

  arr.fill(1337.0);

  for (ulong j = 0; j < arr.size(); ++j)
    ASSERT_DOUBLE_EQ(arr.data()[j], 1337.0);
}

TYPED_TEST(AtomicArrayTest, MultOperator) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    AtomicType arrA = ::GenerateRandomArray<AtomicType, VT>();
    TypeParam oldA = tick_array_copy_to_nonatomic<VT>(arrA);

    arrA *= factor;

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_DOUBLE_EQ(arrA[j], static_cast<VT>(oldA[j] * factor));
  }
}

TYPED_TEST(AtomicArrayTest, DivOperator) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  for (VT factor : {1.0, 2.0, 5.0, 10.0}) {
    AtomicType arrA = ::GenerateRandomArray<AtomicType, VT>();
    TypeParam oldA = tick_array_copy_to_nonatomic<VT>(arrA);

    arrA /= factor;

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j], static_cast<VT>(oldA[j] / factor));
  }
}

TYPED_TEST(AtomicArrayTest, DotProduct) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  TypeParam arrA = ::GenerateRandomArray<TypeParam, VT>();
  AtomicType arrB = ::GenerateRandomArray<AtomicType, VT>();

  typename TypeParam::value_type res{0};
  for (ulong j = 0; j < arrA.size(); ++j) res += arrA[j] * arrB[j];

  EXPECT_RELATIVE_ERROR(VT, res, arrA.dot(arrB));
}

TYPED_TEST(AtomicArrayTest, Multiply) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    AtomicType arrA = ::GenerateRandomArray<AtomicType, VT>();
    TypeParam oldA = tick_array_copy_to_nonatomic<VT>(arrA);

    arrA.multiply(factor);

    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_DOUBLE_EQ(arrA[j], static_cast<VT>(oldA[j] * factor));
  }
}

TYPED_TEST(AtomicArrayTest, MultIncr) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    AtomicType atomicArrayA = ::GenerateRandomArray<AtomicType, VT>();
    Array<VT> arrayA = tick_array_copy_to_nonatomic(atomicArrayA);
    Array<VT> oldA = arrayA;
    AtomicType atomicArrayB = ::GenerateRandomArray<AtomicType, VT>();
    Array<VT> arrayB = tick_array_copy_to_nonatomic(atomicArrayB);

    atomicArrayA.mult_incr(atomicArrayB, factor);
    atomicArrayA.mult_incr(arrayB, 2 * factor);
    arrayA.mult_incr(atomicArrayB, factor);
    arrayA.mult_incr(arrayB, 2 * factor);

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < atomicArrayA.size(); ++j) {
      ASSERT_RELATIVE_ERROR(VT, atomicArrayA[j].load(),
                            static_cast<VT>(oldA[j] + arrayB[j] * 3 * factor));
      ASSERT_RELATIVE_ERROR(VT, arrayA[j], atomicArrayA[j].load());
    }
  }
}

TYPED_TEST(AtomicArrayTest, MultAddMultIncr) {
  using VT = typename TypeParam::value_type;
  using AtomicType = Array<std::atomic<VT>>;

  for (VT factor : {1.0, 1.5, 2.0}) {
    AtomicType arrA = ::GenerateRandomArray<AtomicType, VT>();
    AtomicType oldA = arrA;
    AtomicType arrB = ::GenerateRandomArray<AtomicType, VT>();
    AtomicType arrC = ::GenerateRandomArray<AtomicType, VT>();

    VT factor_2 = factor + 3.0;
    arrA.mult_add_mult_incr(arrB, factor, arrC, factor_2);

    for (ulong j = 0; j < arrA.size(); ++j) {
      oldA.set_data_index(
          j, oldA.get_data_index(j) + (arrB.get_data_index(j) * factor));
    }
    for (ulong j = 0; j < arrA.size(); ++j) {
      oldA.set_data_index(
          j, oldA.get_data_index(j) + (arrC.get_data_index(j) * factor_2));
    }

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j], static_cast<VT>(oldA[j]));
  }
}

#ifdef ADD_MAIN
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "fast";
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
