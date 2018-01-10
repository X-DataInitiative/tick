//
// Created by Martin Bompaire on 13/12/15.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_TEST_TYPEMAP_TEST_H_
#define LIB_INCLUDE_TICK_ARRAY_TEST_TYPEMAP_TEST_H_

// License: BSD 3 clause


#include "tick/base/base.h"

/// @brief Test ArrayDouble Typemap

#define ARRAY_TYPEMAP(TYPE, \
                      ARRAY_TYPE, ARRAY2D_TYPE, ARRAYLIST1D_TYPE, ARRAYLIST2D_TYPE, \
                      SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, \
                      SARRAY_TYPE, SARRAY2D_TYPE, \
                      SARRAYPTR_TYPE, SARRAY2DPTR_TYPE, SARRAYPTRLIST1D_TYPE, SARRAYPTRLIST2D_TYPE, \
                      SARRAY2DPTR_LIST1D_TYPE, SARRAY2DPTR_LIST2D_TYPE, \
                      VARRAY_TYPE, VARRAYPTR_TYPE, VARRAYPTRLIST1D_TYPE, VARRAYPTRLIST2D_TYPE, \
                      BASEARRAY_TYPE, BASEARRAY2D_TYPE, \
                      SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, \
                      SSPARSEARRAY2D_TYPE, SSPARSEARRAY2DPTR_TYPE, \
                      SBASEARRAY2DPTR_TYPE, \
                      SBASEARRAYPTR_TYPE, \
                      BASEARRAY_LIST1D_TYPE, BASEARRAY_LIST2D_TYPE, \
                      BASEARRAY2D_LIST1D_TYPE, BASEARRAY2D_LIST2D_TYPE, \
                      SBASEARRAYPTR_LIST1D_TYPE, SBASEARRAYPTR_LIST2D_TYPE, \
                      SBASEARRAY2DPTR_LIST1D_TYPE, SBASEARRAY2DPTR_LIST2D_TYPE) \
    \
    /******************************** TYPEMAP IN ********************************/ \
    extern TYPE test_typemap_in_##ARRAY_TYPE(ARRAY_TYPE & array); \
    extern TYPE test_typemap_in_##ARRAY2D_TYPE(ARRAY2D_TYPE & array2d); \
    extern TYPE test_typemap_in_##ARRAYLIST1D_TYPE(ARRAYLIST1D_TYPE & array_list); \
    extern TYPE test_typemap_in_##ARRAYLIST2D_TYPE(ARRAYLIST2D_TYPE & array_list_list); \
    \
    extern TYPE test_typemap_in_##SPARSEARRAY_TYPE(SPARSEARRAY_TYPE & sparse_array); \
    extern TYPE test_typemap_in_##SPARSEARRAY2D_TYPE(SPARSEARRAY2D_TYPE & sparse_array2d); \
    \
    extern TYPE test_typemap_in_##SARRAYPTR_TYPE(SARRAYPTR_TYPE sarray); \
    extern TYPE test_typemap_in_##SARRAY2DPTR_TYPE(SARRAY2DPTR_TYPE sarray2d); \
    extern TYPE test_typemap_in_##SARRAYPTRLIST1D_TYPE(SARRAYPTRLIST1D_TYPE & sarray_list); \
    extern TYPE test_typemap_in_##SARRAYPTRLIST2D_TYPE(SARRAYPTRLIST2D_TYPE & sarray_list_list); \
    extern TYPE test_typemap_in_##SARRAY2DPTR_LIST1D_TYPE(SARRAY2DPTR_LIST1D_TYPE & sarray2d_list); \
    extern TYPE test_typemap_in_##SARRAY2DPTR_LIST2D_TYPE(SARRAY2DPTR_LIST2D_TYPE & sarray2d_list_list); \
    \
    extern TYPE test_typemap_in_##VARRAYPTR_TYPE(VARRAYPTR_TYPE varray); \
    extern TYPE test_typemap_in_##VARRAYPTRLIST1D_TYPE(VARRAYPTRLIST1D_TYPE & varray_list); \
    extern TYPE test_typemap_in_##VARRAYPTRLIST2D_TYPE(VARRAYPTRLIST2D_TYPE & varray_list_list); \
    \
    extern TYPE test_typemap_in_##BASEARRAY_TYPE(BASEARRAY_TYPE & basearray); \
    extern TYPE test_typemap_in_##BASEARRAY2D_TYPE(BASEARRAY2D_TYPE & basearray2d); \
    \
    extern TYPE test_typemap_in_##SSPARSEARRAYPTR_TYPE(SSPARSEARRAYPTR_TYPE ssparsearray); \
    extern TYPE test_typemap_in_##SSPARSEARRAY2DPTR_TYPE(SSPARSEARRAY2DPTR_TYPE ssparsearray2d); \
    \
    extern TYPE test_typemap_in_##SBASEARRAYPTR_TYPE(SBASEARRAYPTR_TYPE sbasearray); \
    extern TYPE test_typemap_in_##SBASEARRAY2DPTR_TYPE(SBASEARRAY2DPTR_TYPE sbasearray2d); \
    \
    extern TYPE test_typemap_in_##BASEARRAY_LIST1D_TYPE(BASEARRAY_LIST1D_TYPE & basearray_list); \
    extern TYPE test_typemap_in_##BASEARRAY_LIST2D_TYPE(BASEARRAY_LIST2D_TYPE & basearray_list_list); \
    extern TYPE test_typemap_in_##BASEARRAY2D_LIST1D_TYPE(BASEARRAY2D_LIST1D_TYPE & basearray2d_list); \
    extern TYPE test_typemap_in_##BASEARRAY2D_LIST2D_TYPE(BASEARRAY2D_LIST2D_TYPE & basearray2s_list_list); \
    \
    extern TYPE test_typemap_in_##SBASEARRAYPTR_LIST1D_TYPE(SBASEARRAYPTR_LIST1D_TYPE & sbasearray_list); \
    extern TYPE test_typemap_in_##SBASEARRAYPTR_LIST2D_TYPE(SBASEARRAYPTR_LIST2D_TYPE & sbasearray_list_list); \
    extern TYPE test_typemap_in_##SBASEARRAY2DPTR_LIST1D_TYPE(SBASEARRAY2DPTR_LIST1D_TYPE & sbasearray2d_list); \
    extern TYPE test_typemap_in_##SBASEARRAY2DPTR_LIST2D_TYPE(SBASEARRAY2DPTR_LIST2D_TYPE & sbasearray2d_list_list); \
    \
    /******************************** TYPE CHECK ********************************/ \
    /******************************** check function overloading ****************/ \
    extern TYPE test_typemap_in_##ARRAY_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##ARRAY2D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##ARRAYLIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##ARRAYLIST2D_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##SPARSEARRAY_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SPARSEARRAY2D_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##SARRAYPTR_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SARRAY2DPTR_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SARRAYPTRLIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SARRAYPTRLIST2D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SARRAY2DPTR_LIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SARRAY2DPTR_LIST2D_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##VARRAYPTR_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##VARRAYPTRLIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##VARRAYPTRLIST2D_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##BASEARRAY_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##BASEARRAY2D_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##SSPARSEARRAYPTR_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SSPARSEARRAY2DPTR_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##SBASEARRAYPTR_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SBASEARRAY2DPTR_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##BASEARRAY_LIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##BASEARRAY_LIST2D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##BASEARRAY2D_LIST1D_TYPE(TYPE value);\
    extern TYPE test_typemap_in_##BASEARRAY2D_LIST2D_TYPE(TYPE value); \
    \
    extern TYPE test_typemap_in_##SBASEARRAYPTR_LIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SBASEARRAYPTR_LIST2D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SBASEARRAY2DPTR_LIST1D_TYPE(TYPE value); \
    extern TYPE test_typemap_in_##SBASEARRAY2DPTR_LIST2D_TYPE(TYPE value); \
    \
    /* Add functions that are not overloaded to test error messages*/ \
    extern void test_typemap_in_not_ol_##ARRAY_TYPE(ARRAY_TYPE & array); \
    extern void test_typemap_in_not_ol_##ARRAY2D_TYPE(ARRAY2D_TYPE & array); \
    extern void test_typemap_in_not_ol_##SPARSEARRAY_TYPE(SPARSEARRAY_TYPE & spare_array); \
    extern void test_typemap_in_not_ol_##BASEARRAY_TYPE(BASEARRAY_TYPE & base_array); \
    extern void test_typemap_in_not_ol_##ARRAYLIST1D_TYPE(ARRAYLIST1D_TYPE & array_list); \
    extern void test_typemap_in_not_ol_##ARRAYLIST2D_TYPE(ARRAYLIST2D_TYPE & array_list_list); \
    \
    /******************************** TYPEMAP OUT ********************************/ \
    extern SARRAYPTR_TYPE test_typemap_out_##SARRAYPTR_TYPE(ulong size); \
    extern SARRAYPTRLIST1D_TYPE test_typemap_out_##SARRAYPTRLIST1D_TYPE(int size); \
    extern SARRAYPTRLIST2D_TYPE test_typemap_out_##SARRAYPTRLIST2D_TYPE(int size1, \
                                                                        int size2); \
    extern SARRAY2DPTR_TYPE test_typemap_out_##SARRAY2DPTR_TYPE(ulong row_size, \
                                                                ulong col_size);


ARRAY_TYPEMAP(double,
              ArrayDouble, ArrayDouble2d, ArrayDoubleList1D, ArrayDoubleList2D,
              SparseArrayDouble, SparseArrayDouble2d,
              SArrayDouble, SArrayDouble2d,
              SArrayDoublePtr, SArrayDouble2dPtr, SArrayDoublePtrList1D, SArrayDoublePtrList2D,
              SArrayDouble2dPtrList1D, SArrayDouble2dPtrList2D,
              VArrayDouble, VArrayDoublePtr, VArrayDoublePtrList1D, VArrayDoublePtrList2D,
              BaseArrayDouble, BaseArrayDouble2d,
              SSparseArrayDouble, SSparseArrayDoublePtr,
              SSparseArrayDouble2d, SSparseArrayDouble2dPtr,
              SBaseArrayDouble2dPtr,
              SBaseArrayDoublePtr,
              BaseArrayDoubleList1D, BaseArrayDoubleList2D,
              BaseArrayDouble2dList1D, BaseArrayDouble2dList2D,
              SBaseArrayDoublePtrList1D, SBaseArrayDoublePtrList2D,
              SBaseArrayDouble2dPtrList1D, SBaseArrayDouble2dPtrList2D);

ARRAY_TYPEMAP(std::int32_t,
              ArrayInt, ArrayInt2d, ArrayIntList1D, ArrayIntList2D,
              SparseArrayInt, SparseArrayInt2d,
              SArrayInt, SArrayInt2d,
              SArrayIntPtr, SArrayInt2dPtr, SArrayIntPtrList1D, SArrayIntPtrList2D,
              SArrayInt2dPtrList1D, SArrayInt2dPtrList2D,
              VArrayInt, VArrayIntPtr, VArrayIntPtrList1D, VArrayIntPtrList2D,
              BaseArrayInt, BaseArrayInt2d,
              SSparseArrayInt, SSparseArrayIntPtr,
              SSparseArrayInt2d, SSparseArrayInt2dPtr,
              SBaseArrayInt2dPtr,
              SBaseArrayIntPtr,
              BaseArrayIntList1D, BaseArrayIntList2D,
              BaseArrayInt2dList1D, BaseArrayInt2dList2D,
              SBaseArrayIntPtrList1D, SBaseArrayIntPtrList2D,
              SBaseArrayInt2dPtrList1D, SBaseArrayInt2dPtrList2D);

ARRAY_TYPEMAP(std::int16_t,
              ArrayShort, ArrayShort2d, ArrayShortList1D, ArrayShortList2D,
              SparseArrayShort, SparseArrayShort2d,
              SArrayShort, SArrayShort2d,
              SArrayShortPtr, SArrayShort2dPtr, SArrayShortPtrList1D, SArrayShortPtrList2D,
              SArrayShort2dPtrList1D, SArrayShort2dPtrList2D,
              VArrayShort, VArrayShortPtr, VArrayShortPtrList1D, VArrayShortPtrList2D,
              BaseArrayShort, BaseArrayShort2d,
              SSparseArrayShort, SSparseArrayShortPtr,
              SSparseArrayShort2d, SSparseArrayShort2dPtr,
              SBaseArrayShort2dPtr,
              SBaseArrayShortPtr,
              BaseArrayShortList1D, BaseArrayShortList2D,
              BaseArrayShort2dList1D, BaseArrayShort2dList2D,
              SBaseArrayShortPtrList1D, SBaseArrayShortPtrList2D,
              SBaseArrayShort2dPtrList1D, SBaseArrayShort2dPtrList2D);


ARRAY_TYPEMAP(std::uint16_t,
              ArrayUShort, ArrayUShort2d, ArrayUShortList1D, ArrayUShortList2D,
              SparseArrayUShort, SparseArrayUShort2d,
              SArrayUShort, SArrayUShort2d,
              SArrayUShortPtr, SArrayUShort2dPtr, SArrayUShortPtrList1D, SArrayUShortPtrList2D,
              SArrayUShort2dPtrList1D, SArrayUShort2dPtrList2D,
              VArrayUShort, VArrayUShortPtr, VArrayUShortPtrList1D, VArrayUShortPtrList2D,
              BaseArrayUShort, BaseArrayUShort2d,
              SSparseArrayUShort, SSparseArrayUShortPtr,
              SSparseArrayUShort2d, SSparseArrayUShort2dPtr,
              SBaseArrayUShort2dPtr,
              SBaseArrayUShortPtr,
              BaseArrayUShortList1D, BaseArrayUShortList2D,
              BaseArrayUShort2dList1D, BaseArrayUShort2dList2D,
              SBaseArrayUShortPtrList1D, SBaseArrayUShortPtrList2D,
              SBaseArrayUShort2dPtrList1D, SBaseArrayUShort2dPtrList2D);


ARRAY_TYPEMAP(std::int64_t,
              ArrayLong, ArrayLong2d, ArrayLongList1D, ArrayLongList2D,
              SparseArrayLong, SparseArrayLong2d,
              SArrayLong, SArrayLong2d,
              SArrayLongPtr, SArrayLong2dPtr, SArrayLongPtrList1D, SArrayLongPtrList2D,
              SArrayLong2dPtrList1D, SArrayLong2dPtrList2D,
              VArrayLong, VArrayLongPtr, VArrayLongPtrList1D, VArrayLongPtrList2D,
              BaseArrayLong, BaseArrayLong2d,
              SSparseArrayLong, SSparseArrayLongPtr,
              SSparseArrayLong2d, SSparseArrayLong2dPtr,
              SBaseArrayLong2dPtr,
              SBaseArrayLongPtr,
              BaseArrayLongList1D, BaseArrayLongList2D,
              BaseArrayLong2dList1D, BaseArrayLong2dList2D,
              SBaseArrayLongPtrList1D, SBaseArrayLongPtrList2D,
              SBaseArrayLong2dPtrList1D, SBaseArrayLong2dPtrList2D);


ARRAY_TYPEMAP(std::uint64_t,
              ArrayULong, ArrayULong2d, ArrayULongList1D, ArrayULongList2D,
              SparseArrayULong, SparseArrayULong2d,
              SArrayULong, SArrayULong2d,
              SArrayULongPtr, SArrayULong2dPtr, SArrayULongPtrList1D, SArrayULongPtrList2D,
              SArrayULong2dPtrList1D, SArrayULong2dPtrList2D,
              VArrayULong, VArrayULongPtr, VArrayULongPtrList1D, VArrayULongPtrList2D,
              BaseArrayULong, BaseArrayULong2d,
              SSparseArrayULong, SSparseArrayULongPtr,
              SSparseArrayULong2d, SSparseArrayULong2dPtr,
              SBaseArrayULong2dPtr,
              SBaseArrayULongPtr,
              BaseArrayULongList1D, BaseArrayULongList2D,
              BaseArrayULong2dList1D, BaseArrayULong2dList2D,
              SBaseArrayULongPtrList1D, SBaseArrayULongPtrList2D,
              SBaseArrayULong2dPtrList1D, SBaseArrayULong2dPtrList2D);


ARRAY_TYPEMAP(std::uint32_t,
              ArrayUInt, ArrayUInt2d, ArrayUIntList1D, ArrayUIntList2D,
              SparseArrayUInt, SparseArrayUInt2d,
              SArrayUInt, SArrayUInt2d,
              SArrayUIntPtr, SArrayUInt2dPtr, SArrayUIntPtrList1D, SArrayUIntPtrList2D,
              SArrayUInt2dPtrList1D, SArrayUInt2dPtrList2D,
              VArrayUInt, VArrayUIntPtr, VArrayUIntPtrList1D, VArrayUIntPtrList2D,
              BaseArrayUInt, BaseArrayUInt2d,
              SSparseArrayUInt, SSparseArrayUIntPtr,
              SSparseArrayUInt2d, SSparseArrayUInt2dPtr,
              SBaseArrayUInt2dPtr,
              SBaseArrayUIntPtr,
              BaseArrayUIntList1D, BaseArrayUIntList2D,
              BaseArrayUInt2dList1D, BaseArrayUInt2dList2D,
              SBaseArrayUIntPtrList1D, SBaseArrayUIntPtrList2D,
              SBaseArrayUInt2dPtrList1D, SBaseArrayUInt2dPtrList2D);


ARRAY_TYPEMAP(float,
              ArrayFloat, ArrayFloat2d, ArrayFloatList1D, ArrayFloatList2D,
              SparseArrayFloat, SparseArrayFloat2d,
              SArrayFloat, SArrayFloat2d,
              SArrayFloatPtr, SArrayFloat2dPtr, SArrayFloatPtrList1D, SArrayFloatPtrList2D,
              SArrayFloat2dPtrList1D, SArrayFloat2dPtrList2D,
              VArrayFloat, VArrayFloatPtr, VArrayFloatPtrList1D, VArrayFloatPtrList2D,
              BaseArrayFloat, BaseArrayFloat2d,
              SSparseArrayFloat, SSparseArrayFloatPtr,
              SSparseArrayFloat2d, SSparseArrayFloat2dPtr,
              SBaseArrayFloat2dPtr,
              SBaseArrayFloatPtr,
              BaseArrayFloatList1D, BaseArrayFloatList2D,
              BaseArrayFloat2dList1D, BaseArrayFloat2dList2D,
              SBaseArrayFloatPtrList1D, SBaseArrayFloatPtrList2D,
              SBaseArrayFloat2dPtrList1D, SBaseArrayFloat2dPtrList2D);

#endif  // LIB_INCLUDE_TICK_ARRAY_TEST_TYPEMAP_TEST_H_
