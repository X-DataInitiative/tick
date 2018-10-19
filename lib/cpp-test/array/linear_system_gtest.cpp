// License: BSD 3 clause

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

#include "common.h"

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
