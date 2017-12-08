// License: BSD 3 clause


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

%define TYPEMAPIN_SARRAY(SARRAY_TYPE, SARRAYPTR_TYPE, ARRAY_TYPE, C_TYPE, NP_TYPE)
%{
    // A macro for creating the function that builds an Array<T> from a numpy array (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SARRAY_TYPE(PyObject *obj, SARRAYPTR_TYPE *result) {
        
        if (!TestPyObj_##ARRAY_TYPE((PyObject *) obj)) return false;
        *result = SARRAY_TYPE::new_ptr(0);
        (*result)->set_data((C_TYPE *) PyArray_DATA((PyArrayObject *) (obj)), PyArray_DIM((PyArrayObject *) (obj),0),obj);
        return true;
    }
    %}
%typemap(in) SARRAYPTR_TYPE {if (!BuildFromPyObj_##SARRAY_TYPE($input, &$1)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SARRAYPTR_TYPE {$1=TypeCheckPyObj_##ARRAY_TYPE($input);}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SARRAY2D<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

%define TYPEMAPIN_SARRAY2D(SARRAY2D_TYPE, SARRAY2DPTR_TYPE, ARRAY2D_TYPE, C_TYPE, NP_TYPE)
%{
    // A macro for creating the function that builds an Array<T> from a numpy array (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SARRAY2D_TYPE(PyObject *obj, SARRAY2DPTR_TYPE *result) {
        
        if (!TestPyObj_##ARRAY2D_TYPE((PyObject *) obj)) return false;
        *result = SARRAY2D_TYPE::new_ptr();
        (*result)->set_data((C_TYPE *) PyArray_DATA((PyArrayObject *) (obj)),
                               PyArray_DIM((PyArrayObject *) (obj),0),
                               PyArray_DIM((PyArrayObject *) (obj),1),obj);
        return true;
    }
    %}
%typemap(in) (SARRAY2DPTR_TYPE) {if (!BuildFromPyObj_##SARRAY2D_TYPE($input, &$1)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SARRAY2DPTR_TYPE {$1=TypeCheckPyObj_##ARRAY2D_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of SARRAY<T>Ptr TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of SArrayList1D
%define TYPEMAPIN_SARRAY_LIST1D(ARRAY_TYPE,SARRAY_TYPE,SARRAYPTR_LIST1D_TYPE)
%{
    // Now a function that fills an existing ArrayList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##SARRAY_TYPE(PyObject *obj,SARRAYPTR_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a list of SArrays)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!BuildFromPyObj_##SARRAY_TYPE(obj_i,&(list[i]))) return false;
        }
        return true;
    }
    %}
%typemap(in) (SARRAYPTR_LIST1D_TYPE &) (SARRAYPTR_LIST1D_TYPE res) {$1 = &res; if (!BuildFromPyObj_List1d_##SARRAY_TYPE($input,res)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SARRAYPTR_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of SARRAY<T>Ptr TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of ArrayList2D
%define TYPEMAPIN_SARRAY_LIST2D(ARRAY_TYPE,SARRAY_TYPE,SARRAYPTR_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##SARRAY_TYPE(PyObject *obj,SARRAYPTR_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a 2d-list of SArrays)");
            return false;
        }
        long nCols=0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of SArrays)");
                return false;
            }
            nCols = PyList_Size(obj_i);
//            else if (nCols != PyList_Size(obj_i)) {
//                PyErr_SetString(PyExc_ValueError,"Failed converting argument to 2d-list of SArrays");
//                return false;
//            }
            list[i].resize(nCols);
            for (long j=0;j<nCols;j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (!BuildFromPyObj_##SARRAY_TYPE(obj_ij,&(list[i][j]))) return false;
            }
        }
        return true;
    }
    %}
%typemap(in) (SARRAYPTR_LIST2D_TYPE &) (SARRAYPTR_LIST2D_TYPE res) {$1 = &res; if (!BuildFromPyObj_List2d_##SARRAY_TYPE($input,res)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SARRAYPTR_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY_TYPE($input);}

%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of SARRAY2D<T>Ptr TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of SArray2dList1D
%define TYPEMAPIN_SARRAY2D_LIST1D(ARRAY2D_TYPE, SARRAY2D_TYPE, SARRAY2DPTR_LIST1D_TYPE)
%{
    // Now a function that fills an existing ArrayList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##SARRAY2D_TYPE(PyObject *obj,SARRAY2DPTR_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a list of SArray2d)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i = 0; i < nRows; i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!BuildFromPyObj_##SARRAY2D_TYPE(obj_i, &(list[i]))) return false;
        }
        return true;
    }
    %}
%typemap(in) (SARRAY2DPTR_LIST1D_TYPE &) (SARRAY2DPTR_LIST1D_TYPE res) {$1 = &res; if (!BuildFromPyObj_List1d_##SARRAY2D_TYPE($input,res)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SARRAY2DPTR_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY2D_TYPE($input);}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of SARRAY2d<T>Ptr TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of ArrayList2D
%define TYPEMAPIN_SARRAY2D_LIST2D(ARRAY2D_TYPE, SARRAY2D_TYPE, SARRAY2DPTR_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##SARRAY2D_TYPE(PyObject *obj,SARRAY2DPTR_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a 2d-list of SArray2D)");
            return false;
        }
        long nCols = 0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);

        for (long i = 0; i < nRows; i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of SArray2D)");
                return false;
            }
            nCols = PyList_Size(obj_i);

            list[i].resize(nCols);
            for (long j = 0; j < nCols; j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (!BuildFromPyObj_##SARRAY2D_TYPE(obj_ij, &(list[i][j]))) return false;
            }
        }
        return true;
    }
    %}
%typemap(in) (SARRAY2DPTR_LIST2D_TYPE &) (SARRAY2DPTR_LIST2D_TYPE res) {$1 = &res; if (!BuildFromPyObj_List2d_##SARRAY2D_TYPE($input,res)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SARRAY2DPTR_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY2D_TYPE($input);}

%enddef

////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////


// The final macro for dealing with sarrays
%define SARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE, ARRAY2D_TYPE,
                                SARRAYPTR_TYPE, SARRAY_TYPE,
                                SARRAY2DPTR_TYPE, SARRAY2D_TYPE,
                                SARRAYPTR_LIST1D_TYPE, SARRAYPTR_LIST2D_TYPE,
                                SARRAY2DPTR_LIST1D_TYPE, SARRAY2DPTR_LIST2D_TYPE,
                                C_TYPE, NP_TYPE)
TYPEMAPIN_SARRAY(SARRAY_TYPE, SARRAYPTR_TYPE, ARRAY_TYPE, C_TYPE, NP_TYPE);
TYPEMAPIN_SARRAY2D(SARRAY2D_TYPE, SARRAY2DPTR_TYPE, ARRAY2D_TYPE, C_TYPE, NP_TYPE)
TYPEMAPIN_SARRAY_LIST1D(ARRAY_TYPE,SARRAY_TYPE,SARRAYPTR_LIST1D_TYPE)
TYPEMAPIN_SARRAY_LIST2D(ARRAY_TYPE,SARRAY_TYPE,SARRAYPTR_LIST2D_TYPE)
TYPEMAPIN_SARRAY2D_LIST1D(ARRAY2D_TYPE,SARRAY2D_TYPE,SARRAY2DPTR_LIST1D_TYPE)
TYPEMAPIN_SARRAY2D_LIST2D(ARRAY2D_TYPE,SARRAY2D_TYPE,SARRAY2DPTR_LIST2D_TYPE)

%enddef
