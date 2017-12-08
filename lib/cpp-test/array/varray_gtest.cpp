// License: BSD 3 clause

#include <algorithm>

#define DEBUG_COSTLY_THROW 1
#define XDATA_TEST_DATA_SIZE (100)

#include <gtest/gtest.h>
#include "tick/array/varray.h"

TEST(VArray, Append1) {
    VArrayDouble arr(0);

    EXPECT_NO_THROW(arr.append1(0.0));  // resize to 1
    EXPECT_NO_THROW(arr.append1(1.0));  // resize to (2 * 1.5 = 3)
    EXPECT_NO_THROW(arr.append1(2.0));  // ok
    EXPECT_NO_THROW(arr.append1(3.0));  // resize to (4 * 1.5 = 6)
    EXPECT_NO_THROW(arr.append1(4.0));  // ok
    EXPECT_NO_THROW(arr.append1(5.0));  // ok
    EXPECT_NO_THROW(arr.append1(6.0));  // resize to (7 * 1.5 = 10)

    EXPECT_EQ(arr.size(), 7);
    EXPECT_EQ(arr.get_alloc_size(), static_cast<ulong>(7 * VARRAY_FACTOR_INCRALLOC));

    for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr[j], j);
}

TEST(VArray, Append) {
    VArrayDouble arr(0);

    ArrayDouble x = arange<double>(0, 10);
    ArrayDouble y = arange<double>(10, 20);

    ASSERT_EQ(x.size(), 10);

    arr.append(x.as_sarray_ptr());
    EXPECT_EQ(arr.size(), 10);
    EXPECT_EQ(arr.get_alloc_size(), static_cast<ulong>(10 * VARRAY_FACTOR_INCRALLOC));  // 10 * 1.5 = 15

    arr.append(y.as_sarray_ptr());

    EXPECT_EQ(arr.size(), 20);
    EXPECT_EQ(arr.get_alloc_size(), static_cast<ulong>(20 * VARRAY_FACTOR_INCRALLOC));  // 20 * 1.5 = 30

    for (ulong j = 0; j < arr.size(); ++j) ASSERT_DOUBLE_EQ(arr[j], j);
}

#ifdef ADD_MAIN
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif  // ADD_MAIN
