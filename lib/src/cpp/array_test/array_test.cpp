// License: BSD 3 clause

#include "tick/array_test/array_test.h"

using std::cout;
using std::endl;

double test_constructor_ArrayDouble(ulong size) {
    ArrayDouble a = ArrayDouble(size);
    ulong index = size / 2;

    a[index] = 2;
    double a_i = a[index];

    ArrayDouble b(size);
    b[index] = 3;
    double b_i = b[index];

    double data[5];
    data[0] = 1;
    data[1] = 2;
    data[2] = 7;
    data[3] = 5;
    data[4] = 1;

    ArrayDouble c(size, data);
    ArrayDouble d = ArrayDouble(size, data);

    return a_i * b_i * c[2] * d[3];
}

SArrayDoublePtr test_arange(std::int64_t min, std::int64_t max) {
    ArrayDouble array = arange<double>(min, max);
    SArrayDoublePtr result = array.as_sarray_ptr();
    return result;
}

double test_constructor_SparseArrayDouble() {
    double data[5];
    INDICE_TYPE indices[5];

    for (ulong i = 0; i < 5; i++) data[i] = i;
    indices[0] = 2;
    indices[1] = 4;
    indices[2] = 6;
    indices[3] = 8;
    indices[4] = 10;

    SparseArrayDouble a(12, 5, indices, data);
    SparseArrayDouble b = SparseArrayDouble(12, 5, indices, data);

    return a.sum() + b.sum();
}

double test_constructor_SparseArrayDouble2d() {
    // 0,0,0,1,0
    // 2,0,3,0,1
    // 0,0,0,0,0
    // 0,1,0,0,0
    ulong ncols = 5;
    ulong nrows = 4;
    double data[5];
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    data[3] = 1;
    data[4] = 1;
    INDICE_TYPE indices[5];
    indices[0] = 3;
    indices[1] = 0;
    indices[2] = 2;
    indices[3] = 4;
    indices[4] = 1;
    INDICE_TYPE row_indices[5];
    row_indices[0] = 0;
    row_indices[1] = 1;
    row_indices[2] = 4;
    row_indices[3] = 4;
    row_indices[4] = 5;

    SparseArrayDouble2d a(3, 4);
    a.init_to_zero();
    SparseArrayDouble2d b(nrows, ncols, row_indices, indices, data);
    SparseArrayDouble2d c = SparseArrayDouble2d(nrows, ncols, row_indices, indices, data);

    return a.sum() + b.sum() + c.sum();
}


bool test_BaseArray_empty_constructor(bool flag_dense) {
    BaseArrayDouble ba = BaseArrayDouble(flag_dense);
    return ba.is_dense();
}


double test_IndexError_ArrayDouble(ArrayDouble &array) {
    return array[array.size()];
}

double test_IndexError_rows_ArrayDouble2d(ArrayDouble2d &array) {
    return array(array.n_rows() + 1, 0);
}

double test_IndexError_cols_ArrayDouble2d(ArrayDouble2d &array) {
    return array(0, array.n_cols());
}


#define TEST_INIT_TO_ZERO_CPP(ARRAY_TYPE) \
void test_init_to_zero_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    array.init_to_zero(); \
}

TEST_INIT_TO_ZERO_CPP(BaseArrayDouble);

TEST_INIT_TO_ZERO_CPP(ArrayDouble);

TEST_INIT_TO_ZERO_CPP(SparseArrayDouble);

TEST_INIT_TO_ZERO_CPP(BaseArrayDouble2d);

TEST_INIT_TO_ZERO_CPP(ArrayDouble2d);

TEST_INIT_TO_ZERO_CPP(SparseArrayDouble2d);

#define TEST_INIT_TO_ZERO_PTR_CPP(ARRAY_PTR_TYPE) \
void test_init_to_zero_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    array_ptr->init_to_zero(); \
}

TEST_INIT_TO_ZERO_PTR_CPP(SBaseArrayDoublePtr);

TEST_INIT_TO_ZERO_PTR_CPP(SArrayDoublePtr);

TEST_INIT_TO_ZERO_PTR_CPP(VArrayDoublePtr);

TEST_INIT_TO_ZERO_PTR_CPP(SSparseArrayDoublePtr);

TEST_INIT_TO_ZERO_PTR_CPP(SBaseArrayDouble2dPtr);

TEST_INIT_TO_ZERO_PTR_CPP(SArrayDouble2dPtr);

TEST_INIT_TO_ZERO_PTR_CPP(SSparseArrayDouble2dPtr);

#define TEST_COPY_CPP(ARRAY_TYPE) \
void test_copy_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    /* Copy constructor */\
    ARRAY_TYPE array1 = array; \
    array1.init_to_zero(); \
    /* Copy assignment*/\
    ARRAY_TYPE array2; \
    array2 = array; \
    array2.init_to_zero(); \
}

TEST_COPY_CPP(BaseArrayDouble);

TEST_COPY_CPP(ArrayDouble);

TEST_COPY_CPP(SparseArrayDouble);

TEST_COPY_CPP(BaseArrayDouble2d);

TEST_COPY_CPP(ArrayDouble2d);

TEST_COPY_CPP(SparseArrayDouble2d);

#define TEST_COPY_PTR_CPP(ARRAY_PTR_TYPE) \
void test_copy_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    ARRAY_PTR_TYPE array_ptr1 = array_ptr; \
    array_ptr1->init_to_zero(); \
}

TEST_COPY_PTR_CPP(SBaseArrayDoublePtr);

TEST_COPY_PTR_CPP(SArrayDoublePtr);

TEST_COPY_PTR_CPP(VArrayDoublePtr);

TEST_COPY_PTR_CPP(SSparseArrayDoublePtr);

TEST_COPY_PTR_CPP(SBaseArrayDouble2dPtr);

TEST_COPY_PTR_CPP(SArrayDouble2dPtr);

TEST_COPY_PTR_CPP(SSparseArrayDouble2dPtr);

#define TEST_MOVE_CPP(ARRAY_TYPE) \
/* This function creates a local array that will be returned using move constructor */ \
/* It stores a pointer to the array data of the temporal array */\
ARRAY_TYPE move_##ARRAY_TYPE(ARRAY_TYPE & constructor_array, double** first_ptr) {\
    ARRAY_TYPE array = ARRAY_TYPE(constructor_array); \
    *first_ptr = array.data(); \
    return array;\
} \
\
/* This call the first one and check if the data array is still the same */ \
bool test_move_##ARRAY_TYPE(ARRAY_TYPE & constructor_array) { \
    double** first_ptr = nullptr; \
    /* first_ptr initialization */ \
    double* tmp = 0; \
    first_ptr = &tmp; \
    /* We use move constructor to create this array */ \
    ARRAY_TYPE array = move_##ARRAY_TYPE(constructor_array, first_ptr); \
    /* We check that the newly created array has kept the data of the original one */ \
    bool eq_constructor = *first_ptr == array.data(); \
    /* We use move assignment to assign this array */ \
    array = move_##ARRAY_TYPE(constructor_array, first_ptr); \
    /* We check that the newly assigned array has kept the data of the original one */ \
    bool eq_assignment = *first_ptr == array.data(); \
    return eq_constructor * eq_assignment; \
}

TEST_MOVE_CPP(BaseArrayDouble);

TEST_MOVE_CPP(ArrayDouble);

TEST_MOVE_CPP(SparseArrayDouble);

TEST_MOVE_CPP(BaseArrayDouble2d);

TEST_MOVE_CPP(ArrayDouble2d);

TEST_MOVE_CPP(SparseArrayDouble2d);

#define TEST_VALUE_CPP(ARRAY_TYPE) \
double test_value_##ARRAY_TYPE(ARRAY_TYPE & array, ulong index) {\
    return array.value(index); \
}

TEST_VALUE_CPP(BaseArrayDouble);

TEST_VALUE_CPP(ArrayDouble);

TEST_VALUE_CPP(SparseArrayDouble);

#define TEST_VALUE_PTR_CPP(ARRAY_PTR_TYPE) \
double test_value_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr, ulong index) { \
    return array_ptr->value(index); \
}

TEST_VALUE_PTR_CPP(SBaseArrayDoublePtr);

TEST_VALUE_PTR_CPP(SArrayDoublePtr);

TEST_VALUE_PTR_CPP(VArrayDoublePtr);

TEST_VALUE_PTR_CPP(SSparseArrayDoublePtr);

#define TEST_LAST_CPP(ARRAY_TYPE) \
double test_last_##ARRAY_TYPE(ARRAY_TYPE & array) {\
    return array.last(); \
}

TEST_LAST_CPP(BaseArrayDouble);

TEST_LAST_CPP(ArrayDouble);

TEST_LAST_CPP(SparseArrayDouble);

#define TEST_LAST_PTR_CPP(ARRAY_PTR_TYPE) \
double test_last_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    return array_ptr->last(); \
}

TEST_LAST_PTR_CPP(SBaseArrayDoublePtr);

TEST_LAST_PTR_CPP(SArrayDoublePtr);

TEST_LAST_PTR_CPP(VArrayDoublePtr);

TEST_LAST_PTR_CPP(SSparseArrayDoublePtr);

#define TEST_DOT_CPP(ARRAY_TYPE1, ARRAY_TYPE2) \
double test_dot_##ARRAY_TYPE1##_##ARRAY_TYPE2(ARRAY_TYPE1 & array1, ARRAY_TYPE2 & array2) {\
    return array1.dot(array2); \
}

TEST_DOT_CPP(BaseArrayDouble, BaseArrayDouble);

TEST_DOT_CPP(BaseArrayDouble, ArrayDouble);

TEST_DOT_CPP(BaseArrayDouble, SparseArrayDouble);

TEST_DOT_CPP(ArrayDouble, BaseArrayDouble);

TEST_DOT_CPP(ArrayDouble, ArrayDouble);

TEST_DOT_CPP(ArrayDouble, SparseArrayDouble);

TEST_DOT_CPP(SparseArrayDouble, BaseArrayDouble);

TEST_DOT_CPP(SparseArrayDouble, ArrayDouble);

TEST_DOT_CPP(SparseArrayDouble, SparseArrayDouble);

#define TEST_DOT_PTR_1_CPP(ARRAY_PTR_TYPE1, ARRAY_TYPE2) \
double test_dot_##ARRAY_PTR_TYPE1##_##ARRAY_TYPE2(ARRAY_PTR_TYPE1 array_ptr1, \
                                                  ARRAY_TYPE2 & array2) {\
    return array_ptr1->dot(array2); \
}

TEST_DOT_PTR_1_CPP(SBaseArrayDoublePtr, BaseArrayDouble);

TEST_DOT_PTR_1_CPP(SBaseArrayDoublePtr, ArrayDouble);

TEST_DOT_PTR_1_CPP(SBaseArrayDoublePtr, SparseArrayDouble);

TEST_DOT_PTR_1_CPP(SArrayDoublePtr, BaseArrayDouble);

TEST_DOT_PTR_1_CPP(SArrayDoublePtr, ArrayDouble);

TEST_DOT_PTR_1_CPP(SArrayDoublePtr, SparseArrayDouble);

TEST_DOT_PTR_1_CPP(VArrayDoublePtr, BaseArrayDouble);

TEST_DOT_PTR_1_CPP(VArrayDoublePtr, ArrayDouble);

TEST_DOT_PTR_1_CPP(VArrayDoublePtr, SparseArrayDouble);

TEST_DOT_PTR_1_CPP(SSparseArrayDoublePtr, BaseArrayDouble);

TEST_DOT_PTR_1_CPP(SSparseArrayDoublePtr, ArrayDouble);

TEST_DOT_PTR_1_CPP(SSparseArrayDoublePtr, SparseArrayDouble);


#define TEST_DOT_PTR_2_CPP(ARRAY_TYPE1, ARRAY_PTR_TYPE2) \
double test_dot_##ARRAY_TYPE1##_##ARRAY_PTR_TYPE2(ARRAY_TYPE1 & array1, \
                                                  ARRAY_PTR_TYPE2 array_ptr2) {\
    return array1.dot(*array_ptr2); \
}

TEST_DOT_PTR_2_CPP(BaseArrayDouble, SBaseArrayDoublePtr);

TEST_DOT_PTR_2_CPP(BaseArrayDouble, SArrayDoublePtr);

TEST_DOT_PTR_2_CPP(BaseArrayDouble, VArrayDoublePtr);

TEST_DOT_PTR_2_CPP(BaseArrayDouble, SSparseArrayDoublePtr);

TEST_DOT_PTR_2_CPP(ArrayDouble, SBaseArrayDoublePtr);

TEST_DOT_PTR_2_CPP(ArrayDouble, SArrayDoublePtr);

TEST_DOT_PTR_2_CPP(ArrayDouble, VArrayDoublePtr);

TEST_DOT_PTR_2_CPP(ArrayDouble, SSparseArrayDoublePtr);

TEST_DOT_PTR_2_CPP(SparseArrayDouble, SBaseArrayDoublePtr);

TEST_DOT_PTR_2_CPP(SparseArrayDouble, SArrayDoublePtr);

TEST_DOT_PTR_2_CPP(SparseArrayDouble, VArrayDoublePtr);

TEST_DOT_PTR_2_CPP(SparseArrayDouble, SSparseArrayDoublePtr);

#define TEST_DOT_PTR_PTR_CPP(ARRAY_PTR_TYPE1, ARRAY_PTR_TYPE2) \
double test_dot_##ARRAY_PTR_TYPE1##_##ARRAY_PTR_TYPE2(ARRAY_PTR_TYPE1 array_ptr1, \
                                                      ARRAY_PTR_TYPE2 array_ptr2) {\
    return array_ptr1->dot(*array_ptr2); \
}

TEST_DOT_PTR_PTR_CPP(SBaseArrayDoublePtr, SBaseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SBaseArrayDoublePtr, SArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SBaseArrayDoublePtr, VArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SBaseArrayDoublePtr, SSparseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SArrayDoublePtr, SBaseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SArrayDoublePtr, SArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SArrayDoublePtr, VArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SArrayDoublePtr, SSparseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(VArrayDoublePtr, SBaseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(VArrayDoublePtr, SArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(VArrayDoublePtr, VArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(VArrayDoublePtr, SSparseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SSparseArrayDoublePtr, SBaseArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SSparseArrayDoublePtr, SArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SSparseArrayDoublePtr, VArrayDoublePtr);

TEST_DOT_PTR_PTR_CPP(SSparseArrayDoublePtr, SSparseArrayDoublePtr);


double test_as_array(BaseArrayDouble &a) {
    ArrayDouble array = a.as_array();
    double sum = array.sum();
    array.init_to_zero();
    return sum;
}

double test_as_array2d(BaseArrayDouble2d &a) {
    ArrayDouble2d array_2d = a.as_array2d();
    double sum = array_2d.sum();
    array_2d.init_to_zero();
    return sum;
}

#define TEST_NEW_PTR_CPP(ARRAY_TYPE, ARRAY_PTR_RAW_TYPE, ARRAY_PTR_TYPE) \
double test_new_ptr_##ARRAY_PTR_TYPE(ARRAY_TYPE & array) {\
    ARRAY_PTR_TYPE array_ptr = ARRAY_PTR_RAW_TYPE::new_ptr(array); \
    double sum = array_ptr->sum(); \
    array_ptr->init_to_zero(); \
    return sum; \
}

TEST_NEW_PTR_CPP(ArrayDouble, SArrayDouble, SArrayDoublePtr);

TEST_NEW_PTR_CPP(ArrayDouble, VArrayDouble, VArrayDoublePtr);

TEST_NEW_PTR_CPP(SparseArrayDouble, SSparseArrayDouble, SSparseArrayDoublePtr);

TEST_NEW_PTR_CPP(ArrayDouble2d, SArrayDouble2d, SArrayDouble2dPtr);

TEST_NEW_PTR_CPP(SparseArrayDouble2d, SSparseArrayDouble2d, SSparseArrayDouble2dPtr);


#define TEST_VIEW_CPP(ARRAY_TYPE) \
SArrayDoublePtr test_view_##ARRAY_TYPE(ARRAY_TYPE & array1, ARRAY_TYPE & array2, ARRAY_TYPE & array3) { \
    SArrayDoublePtr results = SArrayDouble::new_ptr(3); \
    /* view1 is a basic view */ \
    ARRAY_TYPE view1 = view(array1); \
    (*results)[0] = view1.sum(); \
    /* view2 is a basic view that which data will be set to 0 */ \
    ARRAY_TYPE view2 = view(array2); \
    (*results)[1] = view2.sum(); \
    view2.init_to_zero(); \
    /* view3b is a view of view which data will be set to 0 */ \
    ARRAY_TYPE view3 = view(array3); \
    ARRAY_TYPE view3b = view(view3); \
    (*results)[2] = view3b.sum(); \
    view3b.init_to_zero(); \
    return results; \
}

TEST_VIEW_CPP(BaseArrayDouble);

TEST_VIEW_CPP(ArrayDouble);

TEST_VIEW_CPP(SparseArrayDouble);

TEST_VIEW_CPP(BaseArrayDouble2d);

TEST_VIEW_CPP(ArrayDouble2d);

TEST_VIEW_CPP(SparseArrayDouble2d);

SArrayDoublePtrList1D test_slice_view1d(ArrayDouble &a,
                                        ulong start,
                                        ulong end) {
    ArrayDouble full_view = view(a);
    ArrayDouble start_slice_view = view(a, start);
    ArrayDouble end_slice_view = view(a, 0, a.size() - end);
    ArrayDouble middle_slice_view = view(a, start, a.size() - end);
    ArrayDouble start_start_slice_view = view(start_slice_view, start);
    ArrayDouble start_end_slice_view = view(end_slice_view, start);
    ArrayDouble end_start_slice_view = view(start_slice_view, 0, start_slice_view.size() - end);
    ArrayDouble end_end_slice_view = view(end_slice_view, 0, end_slice_view.size() - end);

    SArrayDoublePtrList1D results = SArrayDoublePtrList1D(8);
    results[0] = SArrayDouble::new_ptr(full_view);
    results[1] = SArrayDouble::new_ptr(start_slice_view);
    results[2] = SArrayDouble::new_ptr(end_slice_view);
    results[3] = SArrayDouble::new_ptr(middle_slice_view);
    results[4] = SArrayDouble::new_ptr(start_start_slice_view);
    results[5] = SArrayDouble::new_ptr(start_end_slice_view);
    results[6] = SArrayDouble::new_ptr(end_start_slice_view);
    results[7] = SArrayDouble::new_ptr(end_end_slice_view);

    end_start_slice_view.init_to_zero();

    return results;
}

#define TEST_ROW_VIEW_CPP(ARRAY_TYPE, ARRAY_2D_TYPE) \
SArrayDoublePtrList1D test_row_view_##ARRAY_2D_TYPE(ARRAY_2D_TYPE & a, \
                                                    ulong row) \
{ \
    ARRAY_2D_TYPE full_view = view(a); \
    ARRAY_TYPE row_0 = view_row(full_view, 0); \
    ARRAY_TYPE row_1 = view_row(full_view, 1); \
    ARRAY_TYPE row_view = view_row(a, row); \
    /*convert all to dense arrays*/ \
    ArrayDouble row_0_dense = row_0.as_array(); \
    ArrayDouble row_1_dense = row_1.as_array(); \
    ArrayDouble row_view_dense = row_view.as_array(); \
\
    SArrayDoublePtrList1D results = SArrayDoublePtrList1D(3); \
    results[0] = SArrayDouble::new_ptr(row_0_dense); \
    results[1] = SArrayDouble::new_ptr(row_1_dense); \
    results[2] = SArrayDouble::new_ptr(row_view_dense); \
\
    row_view.init_to_zero(); \
    return results; \
}

TEST_ROW_VIEW_CPP(BaseArrayDouble, BaseArrayDouble2d);

TEST_ROW_VIEW_CPP(ArrayDouble, ArrayDouble2d);

TEST_ROW_VIEW_CPP(SparseArrayDouble, SparseArrayDouble2d);

#define TEST_AS_ARRAY_PTR_CPP(ARRAY_TYPE, ARRAY_PTR_TYPE, AS_PTR_FUNC) \
ARRAY_PTR_TYPE test_as_array_ptr_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    /* We first make a copy of it as the first one is a view from Python */ \
    ARRAY_TYPE array2 = array; \
    ARRAY_PTR_TYPE array_ptr = array2.AS_PTR_FUNC(); \
    return array_ptr; \
}

TEST_AS_ARRAY_PTR_CPP(ArrayDouble, SArrayDoublePtr, as_sarray_ptr);

TEST_AS_ARRAY_PTR_CPP(SparseArrayDouble, SSparseArrayDoublePtr, as_ssparsearray_ptr);

TEST_AS_ARRAY_PTR_CPP(ArrayDouble2d, SArrayDouble2dPtr, as_sarray2d_ptr);

TEST_AS_ARRAY_PTR_CPP(SparseArrayDouble2d, SSparseArrayDouble2dPtr, as_ssparsearray2d_ptr);


#define TEST_SUM_CPP(ARRAY_TYPE) \
double test_sum_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    return array.sum(); \
}

TEST_SUM_CPP(BaseArrayDouble);

TEST_SUM_CPP(ArrayDouble);

TEST_SUM_CPP(SparseArrayDouble);

TEST_SUM_CPP(BaseArrayDouble2d);

TEST_SUM_CPP(ArrayDouble2d);

TEST_SUM_CPP(SparseArrayDouble2d);

#define TEST_SUM_PTR_CPP(ARRAY_PTR_TYPE) \
double test_sum_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    return array_ptr->sum(); \
}

TEST_SUM_PTR_CPP(SBaseArrayDoublePtr);

TEST_SUM_PTR_CPP(SArrayDoublePtr);

TEST_SUM_PTR_CPP(VArrayDoublePtr);

TEST_SUM_PTR_CPP(SSparseArrayDoublePtr);

TEST_SUM_PTR_CPP(SBaseArrayDouble2dPtr);

TEST_SUM_PTR_CPP(SArrayDouble2dPtr);

TEST_SUM_PTR_CPP(SSparseArrayDouble2dPtr);


#define TEST_MIN_CPP(ARRAY_TYPE) \
double test_min_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    return array.min(); \
}

TEST_MIN_CPP(BaseArrayDouble);

TEST_MIN_CPP(ArrayDouble);

TEST_MIN_CPP(SparseArrayDouble);

TEST_MIN_CPP(BaseArrayDouble2d);

TEST_MIN_CPP(ArrayDouble2d);

TEST_MIN_CPP(SparseArrayDouble2d);

#define TEST_MIN_PTR_CPP(ARRAY_PTR_TYPE) \
double test_min_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    return array_ptr->min(); \
}

TEST_MIN_PTR_CPP(SBaseArrayDoublePtr);

TEST_MIN_PTR_CPP(SArrayDoublePtr);

TEST_MIN_PTR_CPP(VArrayDoublePtr);

TEST_MIN_PTR_CPP(SSparseArrayDoublePtr);

TEST_MIN_PTR_CPP(SBaseArrayDouble2dPtr);

TEST_MIN_PTR_CPP(SArrayDouble2dPtr);

TEST_MIN_PTR_CPP(SSparseArrayDouble2dPtr);


#define TEST_MAX_CPP(ARRAY_TYPE) \
double test_max_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    return array.max(); \
}

TEST_MAX_CPP(BaseArrayDouble);

TEST_MAX_CPP(ArrayDouble);

TEST_MAX_CPP(SparseArrayDouble);

TEST_MAX_CPP(BaseArrayDouble2d);

TEST_MAX_CPP(ArrayDouble2d);

TEST_MAX_CPP(SparseArrayDouble2d);

#define TEST_MAX_PTR_CPP(ARRAY_PTR_TYPE) \
double test_max_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    return array_ptr->max(); \
}

TEST_MAX_PTR_CPP(SBaseArrayDoublePtr);

TEST_MAX_PTR_CPP(SArrayDoublePtr);

TEST_MAX_PTR_CPP(VArrayDoublePtr);

TEST_MAX_PTR_CPP(SSparseArrayDoublePtr);

TEST_MAX_PTR_CPP(SBaseArrayDouble2dPtr);

TEST_MAX_PTR_CPP(SArrayDouble2dPtr);

TEST_MAX_PTR_CPP(SSparseArrayDouble2dPtr);


#define TEST_NORM_SQ_CPP(ARRAY_TYPE) \
double test_norm_sq_##ARRAY_TYPE(ARRAY_TYPE & array) { \
    return array.norm_sq(); \
}

TEST_NORM_SQ_CPP(BaseArrayDouble);

TEST_NORM_SQ_CPP(ArrayDouble);

TEST_NORM_SQ_CPP(SparseArrayDouble);

TEST_NORM_SQ_CPP(BaseArrayDouble2d);

TEST_NORM_SQ_CPP(ArrayDouble2d);

TEST_NORM_SQ_CPP(SparseArrayDouble2d);

#define TEST_NORM_SQ_PTR_CPP(ARRAY_PTR_TYPE) \
double test_norm_sq_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr) { \
    return array_ptr->norm_sq(); \
}

TEST_NORM_SQ_PTR_CPP(SBaseArrayDoublePtr);

TEST_NORM_SQ_PTR_CPP(SArrayDoublePtr);

TEST_NORM_SQ_PTR_CPP(VArrayDoublePtr);

TEST_NORM_SQ_PTR_CPP(SSparseArrayDoublePtr);

TEST_NORM_SQ_PTR_CPP(SBaseArrayDouble2dPtr);

TEST_NORM_SQ_PTR_CPP(SArrayDouble2dPtr);

TEST_NORM_SQ_PTR_CPP(SSparseArrayDouble2dPtr);


#define TEST_MULTIPLY_CPP(ARRAY_TYPE) \
void test_multiply_##ARRAY_TYPE(ARRAY_TYPE & array, double scalar) { \
    array *= scalar; \
}

TEST_MULTIPLY_CPP(BaseArrayDouble);

TEST_MULTIPLY_CPP(ArrayDouble);

TEST_MULTIPLY_CPP(SparseArrayDouble);

TEST_MULTIPLY_CPP(BaseArrayDouble2d);

TEST_MULTIPLY_CPP(ArrayDouble2d);

TEST_MULTIPLY_CPP(SparseArrayDouble2d);

#define TEST_MULTIPLY_CPP_PTR(ARRAY_PTR_TYPE) \
void test_multiply_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr, double scalar) { \
    *array_ptr *= scalar; \
}

TEST_MULTIPLY_CPP_PTR(SBaseArrayDoublePtr);

TEST_MULTIPLY_CPP_PTR(SArrayDoublePtr);

TEST_MULTIPLY_CPP_PTR(VArrayDoublePtr);

TEST_MULTIPLY_CPP_PTR(SSparseArrayDoublePtr);

TEST_MULTIPLY_CPP_PTR(SBaseArrayDouble2dPtr);

TEST_MULTIPLY_CPP_PTR(SArrayDouble2dPtr);

TEST_MULTIPLY_CPP_PTR(SSparseArrayDouble2dPtr);


#define TEST_DIVIDE_CPP(ARRAY_TYPE) \
void test_divide_##ARRAY_TYPE(ARRAY_TYPE & array, double scalar) { \
    array /= scalar; \
}

TEST_DIVIDE_CPP(BaseArrayDouble);

TEST_DIVIDE_CPP(ArrayDouble);

TEST_DIVIDE_CPP(SparseArrayDouble);

TEST_DIVIDE_CPP(BaseArrayDouble2d);

TEST_DIVIDE_CPP(ArrayDouble2d);

TEST_DIVIDE_CPP(SparseArrayDouble2d);

#define TEST_DIVIDE_CPP_PTR(ARRAY_PTR_TYPE) \
void test_divide_##ARRAY_PTR_TYPE(ARRAY_PTR_TYPE array_ptr, double scalar) { \
    *array_ptr /= scalar; \
}

TEST_DIVIDE_CPP_PTR(SBaseArrayDoublePtr);

TEST_DIVIDE_CPP_PTR(SArrayDoublePtr);

TEST_DIVIDE_CPP_PTR(VArrayDoublePtr);

TEST_DIVIDE_CPP_PTR(SSparseArrayDoublePtr);

TEST_DIVIDE_CPP_PTR(SBaseArrayDouble2dPtr);

TEST_DIVIDE_CPP_PTR(SArrayDouble2dPtr);

TEST_DIVIDE_CPP_PTR(SSparseArrayDouble2dPtr);


// Init an empty VArray and append integers in order to create a range
VArrayDoublePtr test_VArrayDouble_append1(int size) {
    ulong init_size = 0;
    VArrayDoublePtr va1 = VArrayDouble::new_ptr(init_size);
    for (double i = 0; i < size; i++) {
        va1->append1(i);
    }
    return va1;
}

// Recieves two ranges and join each other
VArrayDoublePtr test_VArrayDouble_append(VArrayDoublePtr va, SArrayDoublePtr sa) {
    va->append(sa);
    return va;
}

void test_sort_inplace_ArrayDouble(ArrayDouble &array, bool increasing) {
    array.sort(increasing);
}

SArrayDoublePtr test_sort_ArrayDouble(ArrayDouble &array, bool increasing) {
    return sort<double>(array, increasing).as_sarray_ptr();
}

void test_sort_index_inplace_ArrayDouble(ArrayDouble &array, ArrayULong &index,
                                         bool increasing) {
    array.sort(index, increasing);
}

SArrayDoublePtr test_sort_index_ArrayDouble(ArrayDouble &array, ArrayULong &index,
                                            bool increasing) {
    return sort<double>(array, index, increasing).as_sarray_ptr();
}

void test_sort_abs_index_inplace_ArrayDouble(ArrayDouble &array, ArrayULong &index,
                                             bool increasing) {
    array.sort_abs(index, increasing);
}

void test_mult_incr_ArrayDouble(ArrayDouble &array, BaseArrayDouble &x, double a) {
    array.mult_incr(x, a);
}

void test_mult_fill_ArrayDouble(ArrayDouble &array, BaseArrayDouble &x, double a) {
    array.mult_fill(x, a);
}

void test_mult_add_mult_incr_ArrayDouble(ArrayDouble &array, BaseArrayDouble &x, double a,
                                         BaseArrayDouble &y, double b) {
    array.mult_add_mult_incr(x, a, y, b);
}
