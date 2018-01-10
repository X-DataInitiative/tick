// License: BSD 3 clause


%module array

%import defs.i

%{
#include <system_error>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "tick/base/debug.h"
#include "tick/array/carray_python.h"
%}


// C-Initialization of numpy
%init %{
	import_array();
%}


//
// The following files define some macros
//

// Include miscellaneous test C functions templates used in typemaps
%include array_test_cfunc.i

// Dealing with typemap in templates of Array
%include array_typemap_in.i

// Dealing with typemap in templates of SparseArray
%include sparsearray_typemap_in.i

// Dealing with typemap in templates of BaseArray
%include basearray_typemap_in.i

// Dealing with typemap in templates of SArray
%include sarray_typemap_in.i

// Dealing with typemap in templates of SSparseArray
%include ssparsearray_typemap_in.i

// Dealing with typemap out in templates of SSparseArray
%include ssparsearray_typemap_out.i

// Dealing with SBaseArrayPtr
%include sbasearray_typemap_in.i

// Dealing with typemap in templates of VArray
%include varray_typemap_in.i

// Dealing with typemap out in templates of SArray
%include sarray_typemap_out.i

// Dealing with typemap out in templates of VArray
%include varray_typemap_out.i


//
// Then we define the macro that will call all the above macros
//

%define INSTANTIATE(C_TYPE,NP_TYPE,
                    ARRAY_TYPE,ARRAY2D_TYPE,ARRAYLIST1D_TYPE,ARRAYLIST2D_TYPE,
                    SPARSEARRAY_TYPE,SPARSEARRAY2D_TYPE,
                    SARRAY_TYPE,SARRAY2D_TYPE,SARRAYLIST1D_TYPE,SARRAYLIST2D_TYPE,
                    SARRAYPTR_TYPE,SARRAY2DPTR_TYPE,SARRAYPTRLIST1D_TYPE,SARRAYPTRLIST2D_TYPE,
                    SARRAY2DPTR_LIST1D_TYPE, SARRAY2DPTR_LIST2D_TYPE,
                    VARRAY_TYPE,VARRAYPTR_TYPE,VARRAYPTRLIST1D_TYPE,VARRAYPTRLIST2D_TYPE,
                    BASEARRAY_TYPE,BASEARRAY2D_TYPE,
                    SSPARSEARRAY_TYPE,SSPARSEARRAYPTR_TYPE,
                    SSPARSEARRAY2D_TYPE,SSPARSEARRAY2DPTR_TYPE,
                    SBASEARRAY2DPTR_TYPE,
                    SBASEARRAYPTR_TYPE,
                    BASEARRAY_LIST1D_TYPE,BASEARRAY_LIST2D_TYPE,
                    BASEARRAY2D_LIST1D_TYPE,BASEARRAY2D_LIST2D_TYPE,
                    SBASEARRAYPTR_LIST1D_TYPE,SBASEARRAYPTR_LIST2D_TYPE,
                    SBASEARRAY2DPTR_LIST1D_TYPE,SBASEARRAY2DPTR_LIST2D_TYPE)

TEST_MACROS(ARRAY_TYPE,SPARSEARRAY2D_TYPE,ARRAY2D_TYPE,ARRAYLIST1D_TYPE,ARRAYLIST2D_TYPE,
            C_TYPE,NP_TYPE)

ARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE,ARRAY2D_TYPE,ARRAYLIST1D_TYPE,ARRAYLIST2D_TYPE,
                       C_TYPE,NP_TYPE)

SARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE,ARRAY2D_TYPE,SARRAYPTR_TYPE,SARRAY_TYPE,SARRAY2DPTR_TYPE,
                        SARRAY2D_TYPE,SARRAYPTRLIST1D_TYPE,SARRAYPTRLIST2D_TYPE,
                        SARRAY2DPTR_LIST1D_TYPE, SARRAY2DPTR_LIST2D_TYPE,
                        C_TYPE,NP_TYPE)

SPARSEARRAY_TYPEMAPIN_MACROS(SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE,C_TYPE, NP_TYPE);


SSPARSEARRAY_TYPEMAPIN_MACROS(SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2D_TYPE,
                              SSPARSEARRAY2DPTR_TYPE, SPARSEARRAY_TYPE,
                              SPARSEARRAY2D_TYPE, C_TYPE)

SBASEARRAYPTR_TYPEMAPIN_MACROS(ARRAY_TYPE, ARRAY2D_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, SARRAY_TYPE, SARRAY2D_TYPE,
                                SSPARSEARRAY_TYPE,SSPARSEARRAY2D_TYPE, SARRAYPTR_TYPE, SARRAY2DPTR_TYPE,
                                SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2DPTR_TYPE,
                                SBASEARRAYPTR_TYPE,SBASEARRAY2DPTR_TYPE,
                                SBASEARRAYPTR_LIST1D_TYPE,SBASEARRAYPTR_LIST2D_TYPE,
                                SBASEARRAY2DPTR_LIST1D_TYPE,SBASEARRAY2DPTR_LIST2D_TYPE,C_TYPE)

BASEARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE,ARRAY2D_TYPE,SPARSEARRAY_TYPE,
                               SPARSEARRAY2D_TYPE,BASEARRAY_TYPE,BASEARRAY2D_TYPE,
                               BASEARRAY_LIST1D_TYPE,BASEARRAY_LIST2D_TYPE,
                               BASEARRAY2D_LIST1D_TYPE,BASEARRAY2D_LIST2D_TYPE,
                               C_TYPE,NP_TYPE)

VARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE,VARRAYPTR_TYPE,VARRAY_TYPE,VARRAYPTRLIST1D_TYPE,
                        VARRAYPTRLIST2D_TYPE,C_TYPE,NP_TYPE)

XARRAY_FINAL_MACROS(SARRAYPTR_TYPE, SARRAY_TYPE,
                    SARRAY2DPTR_TYPE, SARRAY2D_TYPE,
                    SARRAYPTRLIST1D_TYPE, SARRAYPTRLIST2D_TYPE,
                    C_TYPE, NP_TYPE)

XSPARSEARRAY_FINAL_MACROS(SSPARSEARRAYPTR_TYPE, SSPARSEARRAY_TYPE,
                    SSPARSEARRAY2DPTR_TYPE, SSPARSEARRAY2D_TYPE,
                    C_TYPE, NP_TYPE)                    

VARRAY_TYPEMAPOUT_MACROS(VARRAYPTR_TYPE, VARRAY_TYPE,
                         VARRAYPTRLIST1D_TYPE, VARRAYPTRLIST2D_TYPE,
                         C_TYPE, NP_TYPE)
%enddef


//
// Then we perform the instantiation of all the types
//



INSTANTIATE(double, NPY_DOUBLE,
            ArrayDouble,ArrayDouble2d,ArrayDoubleList1D,ArrayDoubleList2D,
            SparseArrayDouble,SparseArrayDouble2d,
            SArrayDouble,SArrayDouble2d,SArrayDoubleList1D,SArrayDoubleList2D,
            SArrayDoublePtr,SArrayDouble2dPtr,SArrayDoublePtrList1D,SArrayDoublePtrList2D,
            SArrayDouble2dPtrList1D, SArrayDouble2dPtrList2D,
            VArrayDouble,VArrayDoublePtr,VArrayDoublePtrList1D,VArrayDoublePtrList2D,
            BaseArrayDouble,BaseArrayDouble2d,
            SSparseArrayDouble, SSparseArrayDoublePtr,
            SSparseArrayDouble2d, SSparseArrayDouble2dPtr,
            SBaseArrayDouble2dPtr,
            SBaseArrayDoublePtr,
            BaseArrayDoubleList1D,BaseArrayDoubleList2D,
            BaseArrayDouble2dList1D,BaseArrayDouble2dList2D,
            SBaseArrayDoublePtrList1D,SBaseArrayDoublePtrList2D,
            SBaseArrayDouble2dPtrList1D,SBaseArrayDouble2dPtrList2D);

INSTANTIATE(std::int32_t, NPY_INT32,
            ArrayInt,ArrayInt2d,ArrayIntList1D,ArrayIntList2D,
            SparseArrayInt,SparseArrayInt2d,
            SArrayInt,SArrayInt2d,SArrayIntList1D,SArrayIntList2D,
            SArrayIntPtr,SArrayInt2dPtr,SArrayIntPtrList1D,SArrayIntPtrList2D,
            SArrayInt2dPtrList1D, SArrayInt2dPtrList2D,
            VArrayInt,VArrayIntPtr,VArrayIntPtrList1D,VArrayIntPtrList2D,
            BaseArrayInt,BaseArrayInt2d,
            SSparseArrayInt, SSparseArrayIntPtr,
            SSparseArrayInt2d, SSparseArrayInt2dPtr,
            SBaseArrayInt2dPtr,
            SBaseArrayIntPtr,
            BaseArrayIntList1D,BaseArrayIntList2D,
            BaseArrayInt2dList1D,BaseArrayInt2dList2D,
            SBaseArrayIntPtrList1D,SBaseArrayIntPtrList2D,
            SBaseArrayInt2dPtrList1D,SBaseArrayInt2dPtrList2D);


INSTANTIATE(std::int16_t, NPY_INT16,
            ArrayShort,ArrayShort2d,ArrayShortList1D,ArrayShortList2D,
            SparseArrayShort,SparseArrayShort2d,
            SArrayShort,SArrayShort2d,SArrayShortList1D,SArrayShortList2D,
            SArrayShortPtr,SArrayShort2dPtr,SArrayShortPtrList1D,SArrayShortPtrList2D,
            SArrayShort2dPtrList1D, SArrayShort2dPtrList2D,
            VArrayShort,VArrayShortPtr,VArrayShortPtrList1D,VArrayShortPtrList2D,
            BaseArrayShort,BaseArrayShort2d,
            SSparseArrayShort, SSparseArrayShortPtr,
            SSparseArrayShort2d, SSparseArrayShort2dPtr,
            SBaseArrayShort2dPtr,
            SBaseArrayShortPtr,
            BaseArrayShortList1D,BaseArrayShortList2D,
            BaseArrayShort2dList1D,BaseArrayShort2dList2D,
            SBaseArrayShortPtrList1D,SBaseArrayShortPtrList2D,
            SBaseArrayShort2dPtrList1D,SBaseArrayShort2dPtrList2D);


INSTANTIATE(std::uint16_t, NPY_UINT16,
            ArrayUShort,ArrayUShort2d,ArrayUShortList1D,ArrayUShortList2D,
            SparseArrayUShort,SparseArrayUShort2d,
            SArrayUShort,SArrayUShort2d,SArrayUShortList1D,SArrayUShortList2D,
            SArrayUShortPtr,SArrayUShort2dPtr,SArrayUShortPtrList1D,SArrayUShortPtrList2D,
            SArrayUShort2dPtrList1D, SArrayUShort2dPtrList2D,
            VArrayUShort,VArrayUShortPtr,VArrayUShortPtrList1D,VArrayUShortPtrList2D,
            BaseArrayUShort,BaseArrayUShort2d,
            SSparseArrayUShort, SSparseArrayUShortPtr,
            SSparseArrayUShort2d, SSparseArrayUShort2dPtr,
            SBaseArrayUShort2dPtr,
            SBaseArrayUShortPtr,
            BaseArrayUShortList1D,BaseArrayUShortList2D,
            BaseArrayUShort2dList1D,BaseArrayUShort2dList2D,
            SBaseArrayUShortPtrList1D,SBaseArrayUShortPtrList2D,
            SBaseArrayUShort2dPtrList1D,SBaseArrayUShort2dPtrList2D);


INSTANTIATE(std::int64_t, NPY_INT64,
            ArrayLong,ArrayLong2d,ArrayLongList1D,ArrayLongList2D,
            SparseArrayLong,SparseArrayLong2d,
            SArrayLong,SArrayLong2d,SArrayLongList1D,SArrayLongList2D,
            SArrayLongPtr,SArrayLong2dPtr,SArrayLongPtrList1D,SArrayLongPtrList2D,
            SArrayLong2dPtrList1D, SArrayLong2dPtrList2D,
            VArrayLong,VArrayLongPtr,VArrayLongPtrList1D,VArrayLongPtrList2D,
            BaseArrayLong,BaseArrayLong2d,
            SSparseArrayLong, SSparseArrayLongPtr,
            SSparseArrayLong2d, SSparseArrayLong2dPtr,
            SBaseArrayLong2dPtr,
            SBaseArrayLongPtr,
            BaseArrayLongList1D,BaseArrayLongList2D,
            BaseArrayLong2dList1D,BaseArrayLong2dList2D,
            SBaseArrayLongPtrList1D,SBaseArrayLongPtrList2D,
            SBaseArrayLong2dPtrList1D,SBaseArrayLong2dPtrList2D);


INSTANTIATE(std::uint64_t, NPY_UINT64,
            ArrayULong,ArrayULong2d,ArrayULongList1D,ArrayULongList2D,
            SparseArrayULong,SparseArrayULong2d,
            SArrayULong,SArrayULong2d,SArrayULongList1D,SArrayULongList2D,
            SArrayULongPtr,SArrayULong2dPtr,SArrayULongPtrList1D,SArrayULongPtrList2D,
            SArrayULong2dPtrList1D, SArrayULong2dPtrList2D,
            VArrayULong,VArrayULongPtr,VArrayULongPtrList1D,VArrayULongPtrList2D,
            BaseArrayULong,BaseArrayULong2d,
            SSparseArrayULong, SSparseArrayULongPtr,
            SSparseArrayULong2d, SSparseArrayULong2dPtr,
            SBaseArrayULong2dPtr,
            SBaseArrayULongPtr,
            BaseArrayULongList1D,BaseArrayULongList2D,
            BaseArrayULong2dList1D,BaseArrayULong2dList2D,
            SBaseArrayULongPtrList1D,SBaseArrayULongPtrList2D,
            SBaseArrayULong2dPtrList1D,SBaseArrayULong2dPtrList2D);


INSTANTIATE(std::uint32_t, NPY_UINT32,
            ArrayUInt,ArrayUInt2d,ArrayUIntList1D,ArrayUIntList2D,
            SparseArrayUInt,SparseArrayUInt2d,
            SArrayUInt,SArrayUInt2d,SArrayUIntList1D,SArrayUIntList2D,
            SArrayUIntPtr,SArrayUInt2dPtr,SArrayUIntPtrList1D,SArrayUIntPtrList2D,
            SArrayUInt2dPtrList1D, SArrayUInt2dPtrList2D,
            VArrayUInt,VArrayUIntPtr,VArrayUIntPtrList1D,VArrayUIntPtrList2D,
            BaseArrayUInt,BaseArrayUInt2d,
            SSparseArrayUInt, SSparseArrayUIntPtr,
            SSparseArrayUInt2d, SSparseArrayUInt2dPtr,
            SBaseArrayUInt2dPtr,
            SBaseArrayUIntPtr,
            BaseArrayUIntList1D,BaseArrayUIntList2D,
            BaseArrayUInt2dList1D,BaseArrayUInt2dList2D,
            SBaseArrayUIntPtrList1D,SBaseArrayUIntPtrList2D,
            SBaseArrayUInt2dPtrList1D,SBaseArrayUInt2dPtrList2D);


INSTANTIATE(float, NPY_FLOAT,
            ArrayFloat,ArrayFloat2d,ArrayFloatList1D,ArrayFloatList2D,
            SparseArrayFloat,SparseArrayFloat2d,
            SArrayFloat,SArrayFloat2d,SArrayFloatList1D,SArrayFloatList2D,
            SArrayFloatPtr,SArrayFloat2dPtr,SArrayFloatPtrList1D,SArrayFloatPtrList2D,
            SArrayFloat2dPtrList1D, SArrayFloat2dPtrList2D,
            VArrayFloat,VArrayFloatPtr,VArrayFloatPtrList1D,VArrayFloatPtrList2D,
            BaseArrayFloat,BaseArrayFloat2d,
            SSparseArrayFloat, SSparseArrayFloatPtr,
            SSparseArrayFloat2d, SSparseArrayFloat2dPtr,
            SBaseArrayFloat2dPtr,
            SBaseArrayFloatPtr,
            BaseArrayFloatList1D,BaseArrayFloatList2D,
            BaseArrayFloat2dList1D,BaseArrayFloat2dList2D,
            SBaseArrayFloatPtrList1D,SBaseArrayFloatPtrList2D,
            SBaseArrayFloat2dPtrList1D,SBaseArrayFloat2dPtrList2D);


%include serializer.i
