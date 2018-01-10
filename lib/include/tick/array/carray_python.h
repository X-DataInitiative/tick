#ifndef LIB_INCLUDE_TICK_ARRAY_CARRAY_PYTHON_H_
#define LIB_INCLUDE_TICK_ARRAY_CARRAY_PYTHON_H_

// License: BSD 3 clause

//
// This is the file to be included in .i files of other modules that want to use arrays.
// It includes the declaration for all the needed typemaps in/out and typemap check
//

#include "array.h"
#include "array2d.h"
#include "sarray.h"
#include "sarray2d.h"
#include "sparsearray.h"
#include "sparsearray2d.h"
#include "ssparsearray.h"
#include "ssparsearray2d.h"
#include "sbasearray.h"
#include "sbasearray2d.h"
#include "varray.h"

// For arrays
#define EXTERN_ARRAY(ARRAY_TYPE, ARRAY2D_TYPE, ARRAY_LIST1D_TYPE, ARRAY_LIST2D_TYPE, C_TYPE) \
    extern DLL_PUBLIC bool BuildFromPyObj_##ARRAY_TYPE(PyObject *obj, ARRAY_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_List1d_##ARRAY_TYPE(PyObject *obj, ARRAY_LIST1D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List2d_##ARRAY_TYPE(PyObject *obj, ARRAY_LIST2D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_##ARRAY2D_TYPE(PyObject *obj, ARRAY2D_TYPE *result); \
    extern DLL_PUBLIC int TypeCheckPyObj_##ARRAY_TYPE(PyObject *obj); \
    extern DLL_PUBLIC int TypeCheckPyObj_##ARRAY2D_TYPE(PyObject *obj); \
    extern DLL_PUBLIC int TestPyObj_List1d_##ARRAY_TYPE(PyObject *obj); \
    extern DLL_PUBLIC int TestPyObj_List2d_##ARRAY_TYPE(PyObject *obj); \
    extern DLL_PUBLIC int TestPyObj_List1d_##ARRAY2D_TYPE(PyObject *obj); \
    extern DLL_PUBLIC int TestPyObj_List2d_##ARRAY2D_TYPE(PyObject *obj);

// For sparsearrays
#define EXTERN_SPARSEARRAY(SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE) \
    extern DLL_PUBLIC bool BuildFromPyObj_##SPARSEARRAY2D_TYPE(PyObject *obj, SPARSEARRAY2D_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_##SPARSEARRAY_TYPE(PyObject *obj, SPARSEARRAY_TYPE *result); \
    extern DLL_PUBLIC int TypeCheckPyObj_##SPARSEARRAY_TYPE(PyObject *obj); \
    extern DLL_PUBLIC int TypeCheckPyObj_##SPARSEARRAY2D_TYPE(PyObject *obj); \
    extern DLL_PUBLIC bool TestPyObj_##SPARSEARRAY2D_TYPE(PyObject * obj, ulong *n_rows, ulong *n_cols, INDICE_TYPE **row_indices, INDICE_TYPE **indices, C_TYPE ** data, ulong *size_sparse, PyObject **obj_indptr, PyObject **obj_indices, PyObject **obj_data)

// For BaseArrays 1d and 2d
#define EXTERN_BASE(BASEARRAY_TYPE, BASEARRAY2D_TYPE, BASEARRAY_LIST1D_TYPE, BASEARRAY_LIST2D_TYPE, BASEARRAY2D_LIST1D_TYPE, BASEARRAY2D_LIST2D_TYPE) \
    extern DLL_PUBLIC bool BuildFromPyObj_##BASEARRAY2D_TYPE(PyObject *obj, BASEARRAY2D_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_##BASEARRAY_TYPE(PyObject *obj, BASEARRAY_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_List1d_##BASEARRAY_TYPE(PyObject *obj, BASEARRAY_LIST1D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List2d_##BASEARRAY_TYPE(PyObject *obj, BASEARRAY_LIST2D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List1d_##BASEARRAY2D_TYPE(PyObject *obj, BASEARRAY2D_LIST1D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List2d_##BASEARRAY2D_TYPE(PyObject *obj, BASEARRAY2D_LIST2D_TYPE &list)



// For SBaseArrays
#define EXTERN_SBASE(SBASEARRAYPTR_TYPE, SBASEARRAY2DPTR_TYPE, SBASEARRAYPTR_LIST1D_TYPE, SBASEARRAYPTR_LIST2D_TYPE, SBASEARRAY2DPTR_LIST1D_TYPE, SBASEARRAY2DPTR_LIST2D_TYPE) \
extern DLL_PUBLIC bool BuildFromPyObj_##SBASEARRAYPTR_TYPE(PyObject *obj, SBASEARRAYPTR_TYPE *result); \
extern DLL_PUBLIC bool BuildFromPyObj_##SBASEARRAY2DPTR_TYPE(PyObject *obj, SBASEARRAY2DPTR_TYPE *result); \
extern DLL_PUBLIC bool BuildFromPyObj_##SBASEARRAYPTR_TYPE(PyObject *obj, SBASEARRAYPTR_TYPE *result); \
extern DLL_PUBLIC bool BuildFromPyObj_##SBASEARRAY2DPTR_TYPE(PyObject *obj, SBASEARRAY2DPTR_TYPE *result); \
extern DLL_PUBLIC bool BuildFromPyObj_List1d_##SBASEARRAYPTR_TYPE(PyObject *obj, SBASEARRAYPTR_LIST1D_TYPE &list); \
extern DLL_PUBLIC bool BuildFromPyObj_List2d_##SBASEARRAYPTR_TYPE(PyObject *obj, SBASEARRAYPTR_LIST2D_TYPE &list); \
extern DLL_PUBLIC bool BuildFromPyObj_List1d_##SBASEARRAY2DPTR_TYPE(PyObject *obj, SBASEARRAY2DPTR_LIST1D_TYPE &list); \
extern DLL_PUBLIC bool BuildFromPyObj_List2d_##SBASEARRAY2DPTR_TYPE(PyObject *obj, SBASEARRAY2DPTR_LIST2D_TYPE &list)


// For SArrays
#define EXTERN_SARRAY(SARRAY_TYPE, SARRAYPTR_TYPE, SARRAY2D_TYPE, SARRAY2DPTR_TYPE, SARRAYPTR_LIST1D_TYPE, SARRAYPTR_LIST2D_TYPE, SARRAY2DPTR_LIST1D_TYPE, SARRAY2DPTR_LIST2D_TYPE, ARRAY_TYPE) \
    extern DLL_PUBLIC PyObject *BuildPyListFromXArrayPtrList1D_##SARRAY_TYPE(SARRAYPTR_LIST1D_TYPE &list); \
    extern DLL_PUBLIC PyObject *BuildPyListFromXArrayPtrList2D_##SARRAY_TYPE(SARRAYPTR_LIST2D_TYPE &list); \
    extern DLL_PUBLIC PyObject *_XArray2NumpyArray(SARRAY_TYPE *sig); \
    extern DLL_PUBLIC PyObject *_XArray2d2NumpyArray(SARRAY2D_TYPE *sig); \
    extern DLL_PUBLIC bool BuildFromPyObj_##SARRAY_TYPE(PyObject *obj, SARRAYPTR_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_List1d_##SARRAY_TYPE(PyObject *obj, SARRAYPTR_LIST1D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List2d_##SARRAY_TYPE(PyObject *obj, SARRAYPTR_LIST2D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_##SARRAY2D_TYPE(PyObject *obj, SARRAY2DPTR_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_List1d_##SARRAY2D_TYPE(PyObject *obj, SARRAY2DPTR_LIST1D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List2d_##SARRAY2D_TYPE(PyObject *obj, SARRAY2DPTR_LIST2D_TYPE &list)

// For SSparseArrays
#define EXTERN_SSPARSEARRAY(SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2D_TYPE, SSPARSEARRAY2DPTR_TYPE) \
    extern DLL_PUBLIC bool BuildFromPyObj_##SSPARSEARRAY_TYPE(PyObject *obj, SSPARSEARRAYPTR_TYPE *result); \
    extern DLL_PUBLIC int TypeCheckPyObj_##SSPARSEARRAY_TYPE(PyObject *obj); \
    extern DLL_PUBLIC PyObject *_XSparseArray2NumpyArray(SSPARSEARRAY_TYPE *sig); \
    extern DLL_PUBLIC PyObject *_XSparseArray2d2NumpyArray(SSPARSEARRAY2D_TYPE *sig); \
    extern DLL_PUBLIC bool BuildFromPyObj_##SSPARSEARRAY_TYPE(PyObject *obj, SSPARSEARRAYPTR_TYPE *result); \
    extern DLL_PUBLIC int TypeCheckPyObj_##SSPARSEARRAY_TYPE(PyObject *obj); \
    extern DLL_PUBLIC bool BuildFromPyObj_##SSPARSEARRAY2D_TYPE(PyObject *obj, SSPARSEARRAY2DPTR_TYPE *result); \
    extern DLL_PUBLIC int TypeCheckPyObj_##SSPARSEARRAY2D_TYPE(PyObject *obj)


// For VArrays
#define EXTERN_VARRAY(VARRAY_TYPE, VARRAYPTR_TYPE, VARRAYPTR_LIST1D_TYPE, VARRAYPTR_LIST2D_TYPE, ARRAY_TYPE) \
    extern DLL_PUBLIC PyObject *BuildPyListFromXArrayPtrList1D_##VARRAY_TYPE(VARRAYPTR_LIST1D_TYPE &list); \
    extern DLL_PUBLIC PyObject *BuildPyListFromXArrayPtrList2D_##VARRAY_TYPE(VARRAYPTR_LIST2D_TYPE &list); \
    extern DLL_PUBLIC PyObject *_XArray2NumpyArray(VARRAY_TYPE *sig); \
    extern DLL_PUBLIC bool BuildFromPyObj_##VARRAY_TYPE(PyObject *obj, VARRAYPTR_TYPE *result); \
    extern DLL_PUBLIC bool BuildFromPyObj_List1d_##VARRAY_TYPE(PyObject *obj, VARRAYPTR_LIST1D_TYPE &list); \
    extern DLL_PUBLIC bool BuildFromPyObj_List2d_##VARRAY_TYPE(PyObject *obj, VARRAYPTR_LIST2D_TYPE &list);




//
// Then the instantiation macro
//

#define INSTANTIATE(C_TYPE, NP_TYPE, \
                    ARRAY_TYPE, ARRAY2D_TYPE, ARRAYLIST1D_TYPE, ARRAYLIST2D_TYPE, \
                    SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, \
                    SARRAY_TYPE, SARRAY2D_TYPE, SARRAYLIST1D_TYPE, SARRAYLIST2D_TYPE, \
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
EXTERN_ARRAY(ARRAY_TYPE, ARRAY2D_TYPE, ARRAYLIST1D_TYPE, ARRAYLIST2D_TYPE, C_TYPE); \
EXTERN_SPARSEARRAY(SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE); \
EXTERN_BASE(BASEARRAY_TYPE, BASEARRAY2D_TYPE, BASEARRAY_LIST1D_TYPE, BASEARRAY_LIST2D_TYPE, BASEARRAY2D_LIST1D_TYPE, BASEARRAY2D_LIST2D_TYPE); \
EXTERN_SARRAY(SARRAY_TYPE, SARRAYPTR_TYPE, SARRAY2D_TYPE, SARRAY2DPTR_TYPE, SARRAYPTRLIST1D_TYPE, SARRAYPTRLIST2D_TYPE, SARRAY2DPTR_LIST1D_TYPE, SARRAY2DPTR_LIST2D_TYPE, ARRAY_TYPE); \
EXTERN_SSPARSEARRAY(SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2D_TYPE, SSPARSEARRAY2DPTR_TYPE); \
EXTERN_SBASE(SBASEARRAYPTR_TYPE, SBASEARRAY2DPTR_TYPE, SBASEARRAYPTR_LIST1D_TYPE, SBASEARRAYPTR_LIST2D_TYPE, SBASEARRAY2DPTR_LIST1D_TYPE, SBASEARRAY2DPTR_LIST2D_TYPE); \
EXTERN_VARRAY(VARRAY_TYPE, VARRAYPTR_TYPE, VARRAYPTRLIST1D_TYPE, VARRAYPTRLIST2D_TYPE, ARRAY_TYPE)



//
// Then the various instantations
//


INSTANTIATE(double, NPY_DOUBLE,
            ArrayDouble, ArrayDouble2d, ArrayDoubleList1D, ArrayDoubleList2D,
            SparseArrayDouble, SparseArrayDouble2d,
            SArrayDouble, SArrayDouble2d, SArrayDoubleList1D, SArrayDoubleList2D,
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

INSTANTIATE(std::int32_t, NPY_INT32,
            ArrayInt, ArrayInt2d, ArrayIntList1D, ArrayIntList2D,
            SparseArrayInt, SparseArrayInt2d,
            SArrayInt, SArrayInt2d, SArrayIntList1D, SArrayIntList2D,
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

INSTANTIATE(std::int16_t, NPY_INT16,
            ArrayShort, ArrayShort2d, ArrayShortList1D, ArrayShortList2D,
            SparseArrayShort, SparseArrayShort2d,
            SArrayShort, SArrayShort2d, SArrayShortList1D, SArrayShortList2D,
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

INSTANTIATE(std::uint16_t, NPY_UINT16,
            ArrayUShort, ArrayUShort2d, ArrayUShortList1D, ArrayUShortList2D,
            SparseArrayUShort, SparseArrayUShort2d,
            SArrayUShort, SArrayUShort2d, SArrayUShortList1D, SArrayUShortList2D,
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

INSTANTIATE(std::int64_t, NPY_INT64,
            ArrayLong, ArrayLong2d, ArrayLongList1D, ArrayLongList2D,
            SparseArrayLong, SparseArrayLong2d,
            SArrayLong, SArrayLong2d, SArrayLongList1D, SArrayLongList2D,
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

INSTANTIATE(std::uint64_t, NPY_UINT64,
            ArrayULong, ArrayULong2d, ArrayULongList1D, ArrayULongList2D,
            SparseArrayULong, SparseArrayULong2d,
            SArrayULong, SArrayULong2d, SArrayULongList1D, SArrayULongList2D,
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

INSTANTIATE(std::uint32_t, NPY_UINT32,
            ArrayUInt, ArrayUInt2d, ArrayUIntList1D, ArrayUIntList2D,
            SparseArrayUInt, SparseArrayUInt2d,
            SArrayUInt, SArrayUInt2d, SArrayUIntList1D, SArrayUIntList2D,
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

INSTANTIATE(float, NPY_FLOAT,
            ArrayFloat, ArrayFloat2d, ArrayFloatList1D, ArrayFloatList2D,
            SparseArrayFloat, SparseArrayFloat2d,
            SArrayFloat, SArrayFloat2d, SArrayFloatList1D, SArrayFloatList2D,
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

#endif  // LIB_INCLUDE_TICK_ARRAY_CARRAY_PYTHON_H_
