#ifndef LIB_INCLUDE_TICK_ARRAY_TEST_ARRAY_TEST_H_
#define LIB_INCLUDE_TICK_ARRAY_TEST_ARRAY_TEST_H_

// License: BSD 3 clause

#include "tick/base/base.h"

//
// Simple tests on ArrayDouble
//

/**
 * @brief Testing test_ArrayDouble_creation : Test that we can create ArrayDouble with different
 * constructors,
 * Assign value to specified index
 * Always returns 210
 */
extern double test_constructor_ArrayDouble(ulong size);

/// @brief Testing arange function
extern SArrayDoublePtr test_arange(std::int64_t min, std::int64_t max);

/// @brief  Testing that out of bound index in an array correctly return an error
double test_IndexError_ArrayDouble(ArrayDouble & array);

/// @brief  Testing that out of bound index in an array correctly return an error
extern double test_IndexError_cols_ArrayDouble2d(ArrayDouble2d &array);

//! @brief Test that we can create SparseArrayDouble with different constructor
extern double test_constructor_SparseArrayDouble();

//! @brief Test that we can create SparseArrayDouble2d with different constructor
extern double test_constructor_SparseArrayDouble2d();

//! @brief Test that we can create both sparse and non sparse arrays
extern bool test_BaseArray_empty_constructor(bool flag_dense);

/**
 * @brief Test init_to_zero method
 * The object sent must be modified inplace
 */
#define TEST_INIT_TO_ZERO(ARRAY_TYPE) \
    extern void test_init_to_zero_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_INIT_TO_ZERO(BaseArrayDouble);
TEST_INIT_TO_ZERO(ArrayDouble);
TEST_INIT_TO_ZERO(SparseArrayDouble);
TEST_INIT_TO_ZERO(BaseArrayDouble2d);
TEST_INIT_TO_ZERO(ArrayDouble2d);
TEST_INIT_TO_ZERO(SparseArrayDouble2d);

#define TEST_INIT_TO_ZERO_PTR(ARRAY_PTR_TYPE) \
    extern void test_init_to_zero_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_INIT_TO_ZERO_PTR(SBaseArrayDoublePtr);
TEST_INIT_TO_ZERO_PTR(SArrayDoublePtr);
TEST_INIT_TO_ZERO_PTR(VArrayDoublePtr);
TEST_INIT_TO_ZERO_PTR(SSparseArrayDoublePtr);
TEST_INIT_TO_ZERO_PTR(SBaseArrayDouble2dPtr);
TEST_INIT_TO_ZERO_PTR(SArrayDouble2dPtr);
TEST_INIT_TO_ZERO_PTR(SSparseArrayDouble2dPtr);

/**
 * @brief Test copy assignment method
 * The initial object must not be changed as it has been copied
 */
#define TEST_COPY(ARRAY_TYPE) \
    extern void test_copy_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_COPY(BaseArrayDouble);
TEST_COPY(ArrayDouble);
TEST_COPY(SparseArrayDouble);
TEST_COPY(BaseArrayDouble2d);
TEST_COPY(ArrayDouble2d);
TEST_COPY(SparseArrayDouble2d);

/**
 * @brief Test copy assignment method
 * The object sent must be modified inplace if it was a shared pointer on the same array
 */
#define TEST_COPY_PTR(ARRAY_PTR_TYPE) \
    void test_copy_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_COPY_PTR(SBaseArrayDoublePtr);
TEST_COPY_PTR(SArrayDoublePtr);
TEST_COPY_PTR(VArrayDoublePtr);
TEST_COPY_PTR(SSparseArrayDoublePtr);
TEST_COPY_PTR(SBaseArrayDouble2dPtr);
TEST_COPY_PTR(SArrayDouble2dPtr);
TEST_COPY_PTR(SSparseArrayDouble2dPtr);


/**
 * @brief Test move assignment method
 * We test that data owned by a temporary array is correctly kept and not copied when move
 * constructor is called
 * It must always return True
 * \note : the argument (constructor_array) is only used to construct the temporary array
 */
#define TEST_MOVE(ARRAY_TYPE) \
    extern bool test_move_##ARRAY_TYPE(ARRAY_TYPE & constructor_array);
TEST_MOVE(BaseArrayDouble);
TEST_MOVE(ArrayDouble);
TEST_MOVE(SparseArrayDouble);
TEST_MOVE(BaseArrayDouble2d);
TEST_MOVE(ArrayDouble2d);
TEST_MOVE(SparseArrayDouble2d);

/**
 * @brief Test value method (only available for 1d arrays)
 */
#define TEST_VALUE(ARRAY_TYPE) \
    extern double test_value_##ARRAY_TYPE(ARRAY_TYPE & array, ulong index);
TEST_VALUE(BaseArrayDouble);
TEST_VALUE(ArrayDouble);
TEST_VALUE(SparseArrayDouble);

#define TEST_VALUE_PTR(ARRAY_PTR_TYPE) \
    extern double test_value_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr, ulong index);
TEST_VALUE_PTR(SBaseArrayDoublePtr);
TEST_VALUE_PTR(SArrayDoublePtr);
TEST_VALUE_PTR(VArrayDoublePtr);
TEST_VALUE_PTR(SSparseArrayDoublePtr);

/**
 * @brief Test last method (only available for 1d arrays)
 */
#define TEST_LAST(ARRAY_TYPE) \
    extern double test_last_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_LAST(BaseArrayDouble);
TEST_LAST(ArrayDouble);
TEST_LAST(SparseArrayDouble);

#define TEST_LAST_PTR(ARRAY_PTR_TYPE) \
    extern double test_last_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_LAST_PTR(SBaseArrayDoublePtr);
TEST_LAST_PTR(SArrayDoublePtr);
TEST_LAST_PTR(VArrayDoublePtr);
TEST_LAST_PTR(SSparseArrayDoublePtr);

/**
 * @brief Test dot method (only available for 1d arrays)
 */
#define TEST_DOT(ARRAY_TYPE1, ARRAY_TYPE2) \
    extern double test_dot_##ARRAY_TYPE1##_##ARRAY_TYPE2(ARRAY_TYPE1 & array1, \
                                                         ARRAY_TYPE2 & array2);
TEST_DOT(BaseArrayDouble, BaseArrayDouble);
TEST_DOT(BaseArrayDouble, ArrayDouble);
TEST_DOT(BaseArrayDouble, SparseArrayDouble);
TEST_DOT(ArrayDouble, BaseArrayDouble);
TEST_DOT(ArrayDouble, ArrayDouble);
TEST_DOT(ArrayDouble, SparseArrayDouble);
TEST_DOT(SparseArrayDouble, BaseArrayDouble);
TEST_DOT(SparseArrayDouble, ArrayDouble);
TEST_DOT(SparseArrayDouble, SparseArrayDouble);

#define TEST_DOT_PTR_1(ARRAY_PTR_TYPE1, ARRAY_TYPE2) \
    extern double test_dot_##ARRAY_PTR_TYPE1##_##ARRAY_TYPE2(ARRAY_PTR_TYPE1 array_ptr1, \
                                                             ARRAY_TYPE2 & array2);
TEST_DOT_PTR_1(SBaseArrayDoublePtr, BaseArrayDouble);
TEST_DOT_PTR_1(SBaseArrayDoublePtr, ArrayDouble);
TEST_DOT_PTR_1(SBaseArrayDoublePtr, SparseArrayDouble);
TEST_DOT_PTR_1(SArrayDoublePtr, BaseArrayDouble);
TEST_DOT_PTR_1(SArrayDoublePtr, ArrayDouble);
TEST_DOT_PTR_1(SArrayDoublePtr, SparseArrayDouble);
TEST_DOT_PTR_1(VArrayDoublePtr, BaseArrayDouble);
TEST_DOT_PTR_1(VArrayDoublePtr, ArrayDouble);
TEST_DOT_PTR_1(VArrayDoublePtr, SparseArrayDouble);
TEST_DOT_PTR_1(SSparseArrayDoublePtr, BaseArrayDouble);
TEST_DOT_PTR_1(SSparseArrayDoublePtr, ArrayDouble);
TEST_DOT_PTR_1(SSparseArrayDoublePtr, SparseArrayDouble);

#define TEST_DOT_PTR_2(ARRAY_TYPE1, ARRAY_PTR_TYPE2) \
    extern double test_dot_##ARRAY_TYPE1##_##ARRAY_PTR_TYPE2(ARRAY_TYPE1 & array1, \
                                                             ARRAY_PTR_TYPE2 array_ptr2);
TEST_DOT_PTR_2(BaseArrayDouble, SBaseArrayDoublePtr);
TEST_DOT_PTR_2(BaseArrayDouble, SArrayDoublePtr);
TEST_DOT_PTR_2(BaseArrayDouble, VArrayDoublePtr);
TEST_DOT_PTR_2(BaseArrayDouble, SSparseArrayDoublePtr);
TEST_DOT_PTR_2(ArrayDouble, SBaseArrayDoublePtr);
TEST_DOT_PTR_2(ArrayDouble, SArrayDoublePtr);
TEST_DOT_PTR_2(ArrayDouble, VArrayDoublePtr);
TEST_DOT_PTR_2(ArrayDouble, SSparseArrayDoublePtr);
TEST_DOT_PTR_2(SparseArrayDouble, SBaseArrayDoublePtr);
TEST_DOT_PTR_2(SparseArrayDouble, SArrayDoublePtr);
TEST_DOT_PTR_2(SparseArrayDouble, VArrayDoublePtr);
TEST_DOT_PTR_2(SparseArrayDouble, SSparseArrayDoublePtr);

#define TEST_DOT_PTR_PTR(ARRAY_PTR_TYPE1, ARRAY_PTR_TYPE2) \
    extern double test_dot_##ARRAY_PTR_TYPE1##_##ARRAY_PTR_TYPE2(ARRAY_PTR_TYPE1 array_ptr1, \
                                                                 ARRAY_PTR_TYPE2 array_ptr2);
TEST_DOT_PTR_PTR(SBaseArrayDoublePtr, SBaseArrayDoublePtr);
TEST_DOT_PTR_PTR(SBaseArrayDoublePtr, SArrayDoublePtr);
TEST_DOT_PTR_PTR(SBaseArrayDoublePtr, VArrayDoublePtr);
TEST_DOT_PTR_PTR(SBaseArrayDoublePtr, SSparseArrayDoublePtr);
TEST_DOT_PTR_PTR(SArrayDoublePtr, SBaseArrayDoublePtr);
TEST_DOT_PTR_PTR(SArrayDoublePtr, SArrayDoublePtr);
TEST_DOT_PTR_PTR(SArrayDoublePtr, VArrayDoublePtr);
TEST_DOT_PTR_PTR(SArrayDoublePtr, SSparseArrayDoublePtr);
TEST_DOT_PTR_PTR(VArrayDoublePtr, SBaseArrayDoublePtr);
TEST_DOT_PTR_PTR(VArrayDoublePtr, SArrayDoublePtr);
TEST_DOT_PTR_PTR(VArrayDoublePtr, VArrayDoublePtr);
TEST_DOT_PTR_PTR(VArrayDoublePtr, SSparseArrayDoublePtr);
TEST_DOT_PTR_PTR(SSparseArrayDoublePtr, SBaseArrayDoublePtr);
TEST_DOT_PTR_PTR(SSparseArrayDoublePtr, SArrayDoublePtr);
TEST_DOT_PTR_PTR(SSparseArrayDoublePtr, VArrayDoublePtr);
TEST_DOT_PTR_PTR(SSparseArrayDoublePtr, SSparseArrayDoublePtr);


/** @brief Testing if as_array works
 * It should return the sum of the original array and set its data to 0 if and only if the original
 * array was dense
 */
extern double test_as_array(BaseArrayDouble &a);

/** @brief Testing if as_array2d works
 * It should return the sum of the original array and set its data to 0 if and only if the original
 * array was dense
 */
extern double test_as_array2d(BaseArrayDouble2d &a);

/** @brief Test new_ptr method
 * Data should not be modified as it is copied
 * TODO: when all typemap outs will have been encoded, this function should return the pointer
 * instead of its sum
 */
#define TEST_NEW_PTR(ARRAY_TYPE, ARRAY_PTR_RAW_TYPE, ARRAY_PTR_TYPE) \
    extern double test_new_ptr_##ARRAY_PTR_TYPE(ARRAY_TYPE & array);
TEST_NEW_PTR(ArrayDouble, SarrayDouble, SArrayDoublePtr);
TEST_NEW_PTR(ArrayDouble, VArrayDouble, VArrayDoublePtr);
TEST_NEW_PTR(SparseArrayDouble, SSparseArrayDouble, SSparseArrayDoublePtr);
TEST_NEW_PTR(ArrayDouble2d, SArrayDouble2d, SArrayDouble2dPtr);
TEST_NEW_PTR(SparseArrayDouble2d, SSparseArrayDouble2d, SSparseArrayDouble2dPtr);

/**
 * @brief Test view function
 * \returns an array with the sum of all original arrays
 * It also sets data of the last two to 0
 */
#define TEST_VIEW(ARRAY_TYPE) \
    extern SArrayDoublePtr test_view_##ARRAY_TYPE(ARRAY_TYPE & array1, ARRAY_TYPE & array2, ARRAY_TYPE & array3);

TEST_VIEW(BaseArrayDouble);
TEST_VIEW(ArrayDouble);
TEST_VIEW(SparseArrayDouble);
TEST_VIEW(BaseArrayDouble2d);
TEST_VIEW(ArrayDouble2d);
TEST_VIEW(SparseArrayDouble2d);

extern SArrayDoublePtrList1D test_slice_view1d(ArrayDouble & a, ulong start,
                                         ulong end);

#define TEST_ROW_VIEW(ARRAY_TYPE, ARRAY_2D_TYPE) \
    extern SArrayDoublePtrList1D test_row_view_##ARRAY_2D_TYPE(ARRAY_2D_TYPE & a, \
                                                               ulong row);
TEST_ROW_VIEW(BaseArrayDouble, BaseArrayDouble2d);
TEST_ROW_VIEW(ArrayDouble, ArrayDouble2d);
TEST_ROW_VIEW(SparseArrayDouble, SparseArrayDouble2d);


#define TEST_AS_ARRAY_PTR(ARRAY_TYPE, ARRAY_PTR_TYPE, AS_PTR_FUNC) \
extern ARRAY_PTR_TYPE test_as_array_ptr_##ARRAY_TYPE(ARRAY_TYPE & array);

TEST_AS_ARRAY_PTR(ArrayDouble, SArrayDoublePtr, as_sarray_ptr);
TEST_AS_ARRAY_PTR(SparseArrayDouble, SSparseArrayDoublePtr, as_ssparsearray_ptr);
TEST_AS_ARRAY_PTR(ArrayDouble2d, SArrayDouble2dPtr, as_sarray2d_ptr);
TEST_AS_ARRAY_PTR(SparseArrayDouble2d, SSparseArrayDouble2dPtr, as_ssparsearray2d_ptr);

/**
 * @brief Test sum method
 */
#define TEST_SUM(ARRAY_TYPE) \
    extern double test_sum_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_SUM(BaseArrayDouble);
TEST_SUM(ArrayDouble);
TEST_SUM(SparseArrayDouble);
TEST_SUM(BaseArrayDouble2d);
TEST_SUM(ArrayDouble2d);
TEST_SUM(SparseArrayDouble2d);

#define TEST_SUM_PTR(ARRAY_PTR_TYPE) \
    extern double test_sum_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_SUM_PTR(SBaseArrayDoublePtr);
TEST_SUM_PTR(SArrayDoublePtr);
TEST_SUM_PTR(VArrayDoublePtr);
TEST_SUM_PTR(SSparseArrayDoublePtr);
TEST_SUM_PTR(SBaseArrayDouble2dPtr);
TEST_SUM_PTR(SArrayDouble2dPtr);
TEST_SUM_PTR(SSparseArrayDouble2dPtr);

/**
 * @brief Test min method
 */
#define TEST_MIN(ARRAY_TYPE) \
    extern double test_min_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_MIN(BaseArrayDouble);
TEST_MIN(ArrayDouble);
TEST_MIN(SparseArrayDouble);
TEST_MIN(BaseArrayDouble2d);
TEST_MIN(ArrayDouble2d);
TEST_MIN(SparseArrayDouble2d);

#define TEST_MIN_PTR(ARRAY_PTR_TYPE) \
    extern double test_min_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_MIN_PTR(SBaseArrayDoublePtr);
TEST_MIN_PTR(SArrayDoublePtr);
TEST_MIN_PTR(VArrayDoublePtr);
TEST_MIN_PTR(SSparseArrayDoublePtr);
TEST_MIN_PTR(SBaseArrayDouble2dPtr);
TEST_MIN_PTR(SArrayDouble2dPtr);
TEST_MIN_PTR(SSparseArrayDouble2dPtr);

/**
 * @brief Test max method
 */
#define TEST_MAX(ARRAY_TYPE) \
    extern double test_max_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_MAX(BaseArrayDouble);
TEST_MAX(ArrayDouble);
TEST_MAX(SparseArrayDouble);
TEST_MAX(BaseArrayDouble2d);
TEST_MAX(ArrayDouble2d);
TEST_MAX(SparseArrayDouble2d);

#define TEST_MAX_PTR(ARRAY_PTR_TYPE) \
    extern double test_max_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_MAX_PTR(SBaseArrayDoublePtr);
TEST_MAX_PTR(SArrayDoublePtr);
TEST_MAX_PTR(VArrayDoublePtr);
TEST_MAX_PTR(SSparseArrayDoublePtr);
TEST_MAX_PTR(SBaseArrayDouble2dPtr);
TEST_MAX_PTR(SArrayDouble2dPtr);
TEST_MAX_PTR(SSparseArrayDouble2dPtr);


/**
 * @brief Test norm_sq method
 */
#define TEST_NORM_SQ(ARRAY_TYPE) \
    extern double test_norm_sq_##ARRAY_TYPE(ARRAY_TYPE & array);
TEST_NORM_SQ(BaseArrayDouble);
TEST_NORM_SQ(ArrayDouble);
TEST_NORM_SQ(SparseArrayDouble);
TEST_NORM_SQ(BaseArrayDouble2d);
TEST_NORM_SQ(ArrayDouble2d);
TEST_NORM_SQ(SparseArrayDouble2d);

#define TEST_NORM_SQ_PTR(ARRAY_PTR_TYPE) \
    extern double test_norm_sq_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr);
TEST_NORM_SQ_PTR(SBaseArrayDoublePtr);
TEST_NORM_SQ_PTR(SArrayDoublePtr);
TEST_NORM_SQ_PTR(VArrayDoublePtr);
TEST_NORM_SQ_PTR(SSparseArrayDoublePtr);
TEST_NORM_SQ_PTR(SBaseArrayDouble2dPtr);
TEST_NORM_SQ_PTR(SArrayDouble2dPtr);
TEST_NORM_SQ_PTR(SSparseArrayDouble2dPtr);

/**
 * @brief Test in place multiply method (*=)
 * This will modify given array
 */
#define TEST_MULTIPLY(ARRAY_TYPE) \
    extern void test_multiply_##ARRAY_TYPE(ARRAY_TYPE & array, double scalar);
TEST_MULTIPLY(BaseArrayDouble);
TEST_MULTIPLY(ArrayDouble);
TEST_MULTIPLY(SparseArrayDouble);
TEST_MULTIPLY(BaseArrayDouble2d);
TEST_MULTIPLY(ArrayDouble2d);
TEST_MULTIPLY(SparseArrayDouble2d);

#define TEST_MULTIPLY_PTR(ARRAY_PTR_TYPE) \
    extern void test_multiply_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr, double scalar);
TEST_MULTIPLY_PTR(SBaseArrayDoublePtr);
TEST_MULTIPLY_PTR(SArrayDoublePtr);
TEST_MULTIPLY_PTR(VArrayDoublePtr);
TEST_MULTIPLY_PTR(SSparseArrayDoublePtr);
TEST_MULTIPLY_PTR(SBaseArrayDouble2dPtr);
TEST_MULTIPLY_PTR(SArrayDouble2dPtr);
TEST_MULTIPLY_PTR(SSparseArrayDouble2dPtr);

/**
 * @brief Test in place divide method (/=)
 * This will modify given array
 */
#define TEST_DIVIDE(ARRAY_TYPE) \
    extern void test_divide_##ARRAY_TYPE(ARRAY_TYPE & array, double scalar);
TEST_DIVIDE(BaseArrayDouble);
TEST_DIVIDE(ArrayDouble);
TEST_DIVIDE(SparseArrayDouble);
TEST_DIVIDE(BaseArrayDouble2d);
TEST_DIVIDE(ArrayDouble2d);
TEST_DIVIDE(SparseArrayDouble2d);

#define TEST_DIVIDE_PTR(ARRAY_PTR_TYPE) \
    extern void test_divide_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr, double scalar);
TEST_DIVIDE_PTR(SBaseArrayDoublePtr);
TEST_DIVIDE_PTR(SArrayDoublePtr);
TEST_DIVIDE_PTR(VArrayDoublePtr);
TEST_DIVIDE_PTR(SSparseArrayDoublePtr);
TEST_DIVIDE_PTR(SBaseArrayDouble2dPtr);
TEST_DIVIDE_PTR(SArrayDouble2dPtr);
TEST_DIVIDE_PTR(SSparseArrayDouble2dPtr);


/// @brief Testing VArrayDouble_arange_by_append : Init an empty VArray and append integers in order to create a range
extern VArrayDoublePtr test_VArrayDouble_append1(int size);

/// @brief Testing VArrayDouble_append : Creates two ranges and join each other
extern VArrayDoublePtr test_VArrayDouble_append(VArrayDoublePtr va, SArrayDoublePtr sa);

/// @brief Test sort method of array
extern void test_sort_inplace_ArrayDouble(ArrayDouble & array, bool increasing);

/// @brief Test sort function on array
extern SArrayDoublePtr test_sort_ArrayDouble(ArrayDouble & array, bool increasing);

/// @brief Test sort method of array keeping track of index
extern void test_sort_index_inplace_ArrayDouble(ArrayDouble &array, ArrayULong &index,
                                                bool increasing);

/// @brief Test sort function on array keeping track of index
extern SArrayDoublePtr test_sort_index_ArrayDouble(ArrayDouble &array, ArrayULong &index,
                                                   bool increasing);

/// @brief Test sort method on absolute value keeping track of index
extern void test_sort_abs_index_inplace_ArrayDouble(ArrayDouble &array, ArrayULong &index,
                                                    bool increasing);

extern void test_mult_incr_ArrayDouble(ArrayDouble &array, BaseArrayDouble& x, double a);
extern void test_mult_fill_ArrayDouble(ArrayDouble &array, BaseArrayDouble& x, double a);
extern void test_mult_add_mult_incr_ArrayDouble(ArrayDouble &array, BaseArrayDouble& x, double a,
                                                BaseArrayDouble& y, double b);

#endif  // LIB_INCLUDE_TICK_ARRAY_TEST_ARRAY_TEST_H_
