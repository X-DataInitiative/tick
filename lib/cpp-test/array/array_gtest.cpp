// License: BSD 3 clause

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

#include "common.h"

TEST(ArrayTestSetup, RelativeErrors) {
  ASSERT_EQ(GetAcceptedRelativeError<int>(), 0);
  ASSERT_DOUBLE_EQ(GetAcceptedRelativeError<float>(),
                   TICK_TEST_SINGLE_RELATIVE_ERROR);
  ASSERT_DOUBLE_EQ(GetAcceptedRelativeError<double>(),
                   TICK_TEST_DOUBLE_RELATIVE_ERROR);
}

template <typename ArrType>
class ArrayTest : public ::testing::Test {
 public:
  using value_type = typename ArrType::value_type;
};

typedef ::testing::Types<ArrayFloat, ArrayDouble, ArrayShort, ArrayUShort,
                         ArrayInt, ArrayUInt, ArrayLong, ArrayULong>
    MyArrayTypes;
TYPED_TEST_SUITE(ArrayTest, MyArrayTypes);

template <typename ArrType>
class Array2dTest : public ::testing::Test {
 public:
  using value_type = typename ArrType::value_type;
};

typedef ::testing::Types<ArrayFloat2d, ArrayDouble2d, ArrayShort2d,
                         ArrayUShort2d, ArrayInt2d, ArrayUInt2d, ArrayLong2d,
                         ArrayULong2d>
    MyArray2dTypes;
TYPED_TEST_SUITE(Array2dTest, MyArray2dTypes);

TYPED_TEST(ArrayTest, InitToZero) {
  TypeParam arr{TICK_TEST_DATA_SIZE};

  arr.init_to_zero();

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr.data()[j], 0.0);
}

TYPED_TEST(ArrayTest, Copy) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>();
  TypeParam arrCopy = arr;

  EXPECT_EQ(arr.size(), arrCopy.size());
  EXPECT_NE(arr.data(), arrCopy.data());

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr[j], arrCopy[j]);
}

TYPED_TEST(ArrayTest, Move) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>();
  const auto *arrDataPtr = arr.data();

  TypeParam arrCopy = arr;
  TypeParam arrMoved = std::move(arr);

  EXPECT_EQ(0u, arr.size());
  EXPECT_EQ(arrCopy.size(), arrMoved.size());
  EXPECT_NE(arr.data(), arrMoved.data());
  EXPECT_EQ(arrDataPtr, arrMoved.data());

  for (ulong j = 0; j < arrMoved.size(); ++j)
    ASSERT_DOUBLE_EQ(arrCopy[j], arrMoved[j]);
}

TYPED_TEST(ArrayTest, View) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>(100);

  TypeParam arrView = view(arr);

  EXPECT_EQ(100u, arr.size());
  EXPECT_EQ(arrView.size(), arr.size());
  EXPECT_EQ(arr.data(), arrView.data());

  for (ulong j = 0; j < arrView.size(); ++j)
    ASSERT_DOUBLE_EQ(arr[j], arrView[j]);
}

TYPED_TEST(ArrayTest, Index) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>();

  for (ulong j = 0; j < arr.size(); ++j)
    ASSERT_DOUBLE_EQ(arr[j], arr.data()[j]);
  for (ulong j = 0; j < arr.size(); ++j)
    ASSERT_DOUBLE_EQ(arr.value(j), arr.data()[j]);
}

TYPED_TEST(ArrayTest, ConstIndex) {
  const TypeParam arr = ::GenerateRandomArray<TypeParam>();

  for (ulong j = 0; j < arr.size(); ++j)
    ASSERT_DOUBLE_EQ(arr[j], arr.data()[j]);
  for (ulong j = 0; j < arr.size(); ++j)
    ASSERT_DOUBLE_EQ(arr.value(j), arr.data()[j]);
}

TYPED_TEST(ArrayTest, Last) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>();

  ASSERT_DOUBLE_EQ(arr.last(), arr.data()[arr.size() - 1]);
}

TYPED_TEST(ArrayTest, Bounds) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>(100);

  EXPECT_THROW(arr[-1], std::out_of_range);
  EXPECT_THROW(arr[100], std::out_of_range);
  EXPECT_THROW(arr[10000], std::out_of_range);
}

TYPED_TEST(ArrayTest, Size) {
  TypeParam arr = ::GenerateRandomArray<TypeParam>(100);

  EXPECT_EQ(arr.size(), 100u);
  EXPECT_EQ(arr.size_data(), 100u);
}

TYPED_TEST(ArrayTest, InitList) {
  using VT = typename TypeParam::value_type;

  std::array<VT, 6> vals = {{static_cast<VT>(0.0), static_cast<VT>(1.0),
                             static_cast<VT>(2.0), static_cast<VT>(4.0),
                             static_cast<VT>(8.0), static_cast<VT>(16.0)}};

  TypeParam arr{vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]};

  ASSERT_EQ(arr.size(), 6u);

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr[j], vals[j]);
}

TYPED_TEST(ArrayTest, Fill) {
  TypeParam arr{TICK_TEST_DATA_SIZE};

  arr.fill(1337.0);

  for (ulong j = 0; j < arr.size(); ++j)
    ASSERT_DOUBLE_EQ(arr.data()[j], 1337.0);
}

TYPED_TEST(ArrayTest, Fill_Int_to_double_no_cast) {
  TypeParam arr(TICK_TEST_DATA_SIZE);
  uint32_t leet = 1337;
  arr.fill(leet);

  for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr.data()[j], leet);
}

TYPED_TEST(ArrayTest, Sum) {
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();

  const auto sum = std::accumulate(arrA.data(), arrA.data() + arrA.size(), 0.0);

  EXPECT_DOUBLE_EQ(arrA.sum(), sum);
}

TYPED_TEST(ArrayTest, Min) {
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();

  EXPECT_DOUBLE_EQ(arrA.min(),
                   *std::min_element(arrA.data(), arrA.data() + arrA.size()));
}

TYPED_TEST(ArrayTest, Max) {
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();

  EXPECT_DOUBLE_EQ(arrA.max(),
                   *std::max_element(arrA.data(), arrA.data() + arrA.size()));
}

TYPED_TEST(ArrayTest, Contains) {
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();

  typename TypeParam::value_type value_3 = arrA[3] + 0.;
  typename TypeParam::value_type new_value = 7032;

  int retries = 0;
  while(retries++ < 5 && arrA.contains(new_value)) arrA = ::GenerateRandomArray<TypeParam>();

  EXPECT_TRUE(arrA.contains(value_3));
  EXPECT_FALSE(arrA.contains(new_value));
  arrA[3] = new_value;
  EXPECT_FALSE(arrA.contains(value_3));
  EXPECT_TRUE(arrA.contains(new_value));
}

TYPED_TEST(ArrayTest, MultOperator) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray<TypeParam>();
    TypeParam oldA = arrA;

    arrA *= factor;

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_DOUBLE_EQ(arrA[j], static_cast<VT>(oldA[j] * factor));
  }
}

TYPED_TEST(ArrayTest, DivOperator) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {1.0, 2.0, 5.0, 10.0}) {
    TypeParam arrA = ::GenerateRandomArray<TypeParam>();
    TypeParam oldA = arrA;

    arrA /= factor;

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j], static_cast<VT>(oldA[j] / factor));
  }
}

TYPED_TEST(ArrayTest, NormSq) {
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();

  auto norm_sq = arrA.norm_sq();

  decltype(norm_sq) result{0};
  for (ulong j = 0; j < arrA.size(); ++j) result += arrA[j] * arrA[j];

  EXPECT_RELATIVE_ERROR(decltype(norm_sq), arrA.norm_sq(), result);
}

TYPED_TEST(ArrayTest, DotProduct) {
  using VT = typename TypeParam::value_type;
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();
  TypeParam arrB = ::GenerateRandomArray<TypeParam>();

  typename TypeParam::value_type res{0};
  for (ulong j = 0; j < arrA.size(); ++j) res += arrA[j] * arrB[j];

  EXPECT_RELATIVE_ERROR(VT, res, arrA.dot(arrB));
}

TYPED_TEST(ArrayTest, Multiply) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray<TypeParam>();
    TypeParam oldA = arrA;

    arrA.multiply(factor);

    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_DOUBLE_EQ(arrA[j], static_cast<VT>(oldA[j] * factor));
  }
}

TYPED_TEST(ArrayTest, MultIncr) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray<TypeParam>();
    TypeParam oldA = arrA;
    TypeParam arrB = ::GenerateRandomArray<TypeParam>();

    arrA.mult_incr(arrB, factor);

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j],
                            static_cast<VT>(oldA[j] + arrB[j] * factor));
  }
}

TYPED_TEST(Array2dTest, MultIncr) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray2d<TypeParam>();
    TypeParam oldA = arrA;
    TypeParam arrB = ::GenerateRandomArray2d<TypeParam>();

    arrA.mult_incr(arrB, factor);

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j],
                            static_cast<VT>(oldA[j] + arrB[j] * factor));
  }
}

TYPED_TEST(ArrayTest, MultAddMultIncr) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {1.0, 1.5, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray<TypeParam>();
    TypeParam oldA = arrA;
    TypeParam arrB = ::GenerateRandomArray<TypeParam>();
    TypeParam arrC = ::GenerateRandomArray<TypeParam>();

    VT factor_2 = factor + 3.0;
    arrA.mult_add_mult_incr(arrB, factor, arrC, factor_2);

    for (ulong j = 0; j < arrA.size(); ++j) oldA[j] += arrB[j] * factor;
    for (ulong j = 0; j < arrA.size(); ++j) oldA[j] += arrC[j] * factor_2;

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j], static_cast<VT>(oldA[j]));
  }
}

TYPED_TEST(Array2dTest, MultAddMultIncr) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {1.0, 1.5, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray2d<TypeParam>();
    TypeParam oldA = arrA;
    TypeParam arrB = ::GenerateRandomArray2d<TypeParam>();
    TypeParam arrC = ::GenerateRandomArray2d<TypeParam>();

    VT factor_2 = factor + 3.0;
    arrA.mult_add_mult_incr(arrB, factor, arrC, factor_2);

    for (ulong j = 0; j < arrA.size(); ++j) oldA[j] += arrB[j] * factor;
    for (ulong j = 0; j < arrA.size(); ++j) oldA[j] += arrC[j] * factor_2;

    SCOPED_TRACE(factor);
    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_RELATIVE_ERROR(VT, arrA[j], static_cast<VT>(oldA[j]));
  }
}

TYPED_TEST(ArrayTest, MultFill) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray<TypeParam>();
    TypeParam arrB = ::GenerateRandomArray<TypeParam>();

    arrA.mult_fill(arrB, factor);

    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_DOUBLE_EQ(arrA[j], static_cast<VT>(arrB[j] * factor));
  }
}

TYPED_TEST(Array2dTest, MultFill) {
  using VT = typename TypeParam::value_type;
  for (VT factor : {0.0, 0.5, 1.0, 2.0}) {
    TypeParam arrA = ::GenerateRandomArray2d<TypeParam>();
    TypeParam arrB = ::GenerateRandomArray2d<TypeParam>();

    arrA.mult_fill(arrB, factor);

    for (ulong j = 0; j < arrA.size(); ++j)
      ASSERT_DOUBLE_EQ(arrA[j], static_cast<VT>(arrB[j] * factor));
  }
}

namespace {

template <typename ArrType, typename F1, typename F2>
void TestSorting(F1 stlSort, F2 tickSort) {
  ArrType arrA;

  // Repeat until we get a shuffled array (most likely the first one)
  while (std::is_sorted(arrA.data(), arrA.data() + arrA.size(), stlSort) ||
         arrA.size() == 0)
    arrA = ::GenerateRandomArray<ArrType>();

  ASSERT_FALSE(std::is_sorted(arrA.data(), arrA.data() + arrA.size(), stlSort));

  tickSort(arrA);

  EXPECT_TRUE(std::is_sorted(arrA.data(), arrA.data() + arrA.size(), stlSort));
}

}  // namespace

TYPED_TEST(ArrayTest, Sort) {
  SCOPED_TRACE("");
  ::TestSorting<TypeParam>(std::less<typename TypeParam::value_type>(),
                           [](TypeParam &arr) { arr.sort(true); });
}

TYPED_TEST(ArrayTest, SortDecreasing) {
  SCOPED_TRACE("");
  ::TestSorting<TypeParam>(std::greater<typename TypeParam::value_type>(),
                           [](TypeParam &arr) { arr.sort(false); });
}

TYPED_TEST(ArrayTest, SortAbs) {
  using VT = typename TypeParam::value_type;

  SCOPED_TRACE("");
  ::TestSorting<TypeParam>(
      [](const VT &lhs, const VT &rhs) {
        return std::fabs(lhs) < std::fabs(rhs);
      },
      [](TypeParam &arr) {
        Array<ulong> indices(arr.size());
        arr.sort_abs(indices, true);
      });
}

TYPED_TEST(ArrayTest, SortAbsDecreasing) {
  using VT = typename TypeParam::value_type;

  SCOPED_TRACE("");
  ::TestSorting<TypeParam>(
      [](const VT &lhs, const VT &rhs) {
        return std::fabs(lhs) > std::fabs(rhs);
      },
      [](TypeParam &arr) {
        Array<ulong> indices(arr.size());
        arr.sort_abs(indices, false);
      });
}

namespace {

template <typename ArrType, typename F1, typename F2>
void TestSortingAndTracking(F1 stlSort, F2 tickSort) {
  ::TestSorting<ArrType>(stlSort, [&tickSort](ArrType &arr) {
    ArrType arrCopy = arr;
    Array<ulong> indices(arr.size());
    tickSort(arr, indices);

    for (ulong j = 0; j < indices.size(); ++j) {
      ASSERT_DOUBLE_EQ(arr[j], arrCopy[indices[j]]);
    }
  });
}

}  // namespace

TYPED_TEST(ArrayTest, SortTrack) {
  SCOPED_TRACE("");
  ::TestSortingAndTracking<TypeParam>(
      std::less<typename TypeParam::value_type>(),
      [](TypeParam &arr, Array<ulong> &indices) { arr.sort(indices, true); });
}

TYPED_TEST(ArrayTest, SortTrackDecreasing) {
  SCOPED_TRACE("");
  ::TestSortingAndTracking<TypeParam>(
      std::greater<typename TypeParam::value_type>(),
      [](TypeParam &arr, Array<ulong> &indices) { arr.sort(indices, false); });
}

TYPED_TEST(ArrayTest, SortAbsTrack) {
  using VT = typename TypeParam::value_type;

  SCOPED_TRACE("");
  ::TestSortingAndTracking<TypeParam>(
      [](const VT &lhs, const VT &rhs) {
        return std::fabs(lhs) < std::fabs(rhs);
      },
      [](TypeParam &arr, Array<ulong> &indices) {
        arr.sort_abs(indices, true);
      });
}

TYPED_TEST(ArrayTest, SortAbsTrackDecreasing) {
  using VT = typename TypeParam::value_type;

  SCOPED_TRACE("");
  ::TestSortingAndTracking<TypeParam>(
      [](const VT &lhs, const VT &rhs) {
        return std::fabs(lhs) > std::fabs(rhs);
      },
      [](TypeParam &arr, Array<ulong> &indices) {
        arr.sort_abs(indices, false);
      });
}

namespace {

template <typename ArchiveIn, typename ArchiveOut, typename TypeParam>
void TestEmptySerialization() {
  TypeParam arrA;

  std::stringstream os;
  {
    ArchiveOut archive_out(os);

    archive_out(arrA);
  }

  {
    ArchiveIn archive_in(os);

    TypeParam arrB;
    archive_in(arrB);

    ASSERT_EQ(arrA.size(), arrB.size());
    ASSERT_EQ(arrA.size(), 0u);
  }
}

template <typename ArchiveIn, typename ArchiveOut, typename TypeParam>
void TestSerialization() {
  TypeParam arrA = ::GenerateRandomArray<TypeParam>();

  std::stringstream os;
  {
    ArchiveOut archive_out(os);

    archive_out(arrA);
  }

  {
    ArchiveIn archive_in(os);

    TypeParam arrB;
    archive_in(arrB);

    ASSERT_EQ(arrA.size(), arrB.size());
    for (ulong j = 0; j < arrA.size(); ++j) ASSERT_DOUBLE_EQ(arrA[j], arrB[j]);
  }
}

template <typename ArchiveIn, typename ArchiveOut, typename Arr2DType>
void TestEmptySerialization2D() {
  Arr2DType arrA;

  std::stringstream os;
  {
    ArchiveOut archive_out(os);

    archive_out(arrA);
  }

  {
    ArchiveIn archive_in(os);

    Arr2DType arrB;
    archive_in(arrB);

    ASSERT_EQ(arrA.size(), arrB.size());
    ASSERT_EQ(arrA.size(), 0u);
    ASSERT_EQ(arrA.n_rows(), arrB.n_rows());
    ASSERT_EQ(arrA.n_rows(), 0u);
    ASSERT_EQ(arrA.n_cols(), arrB.n_cols());
    ASSERT_EQ(arrA.n_cols(), 0u);
  }
}

template <typename ArchiveIn, typename ArchiveOut, typename Arr2DType>
void TestSerialization2D() {
  Arr2DType arrA = GenerateRandomArray2d<Arr2DType>();

  std::stringstream os;
  {
    ArchiveOut archive_out(os);

    archive_out(arrA);
  }

  {
    ArchiveIn archive_in(os);

    Arr2DType arrB;
    archive_in(arrB);

    ASSERT_EQ(arrA.size(), arrB.size());
    ASSERT_EQ(arrA.n_rows(), arrB.n_rows());
    ASSERT_EQ(arrA.n_cols(), arrB.n_cols());
    for (ulong j = 0; j < arrA.size(); ++j) ASSERT_DOUBLE_EQ(arrA[j], arrB[j]);
  }
}

}  // namespace

TYPED_TEST(ArrayTest, EmptySerializationPortableBinary) {
  SCOPED_TRACE("");
  ::TestEmptySerialization<cereal::PortableBinaryInputArchive, cereal::PortableBinaryOutputArchive,
                           TypeParam>();
}

TYPED_TEST(ArrayTest, EmptySerializationBinary) {
  SCOPED_TRACE("");
  ::TestEmptySerialization<cereal::PortableBinaryInputArchive, cereal::PortableBinaryOutputArchive,
                           TypeParam>();
}

TYPED_TEST(Array2dTest, EmptySerializationPortableBinary) {
  SCOPED_TRACE("");
  ::TestEmptySerialization2D<cereal::PortableBinaryInputArchive,
                             cereal::PortableBinaryOutputArchive, TypeParam>();
}

TYPED_TEST(Array2dTest, EmptySerializationBinary) {
  SCOPED_TRACE("");
  ::TestEmptySerialization2D<cereal::PortableBinaryInputArchive,
                             cereal::PortableBinaryOutputArchive, TypeParam>();
}

TYPED_TEST(ArrayTest, SerializationPortableBinary) {
  SCOPED_TRACE("");
  ::TestSerialization<cereal::PortableBinaryInputArchive, cereal::PortableBinaryOutputArchive,
                      TypeParam>();
}

TYPED_TEST(ArrayTest, SerializationBinary) {
  SCOPED_TRACE("");
  ::TestSerialization<cereal::PortableBinaryInputArchive, cereal::PortableBinaryOutputArchive,
                      TypeParam>();
}

TYPED_TEST(Array2dTest, SerializationPortableBinary) {
  SCOPED_TRACE("");
  ::TestSerialization2D<cereal::PortableBinaryInputArchive, cereal::PortableBinaryOutputArchive,
                        TypeParam>();
}

TYPED_TEST(Array2dTest, SerializationBinary) {
  SCOPED_TRACE("");
  ::TestSerialization2D<cereal::PortableBinaryInputArchive, cereal::PortableBinaryOutputArchive,
                        TypeParam>();
}

#ifdef ADD_MAIN
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::FLAGS_gtest_death_test_style = "fast";
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
