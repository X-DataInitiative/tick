// License: BSD 3 clause



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH ARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

%define TYPEMAPIN_ARRAY(ARRAY_TYPE, C_TYPE, NP_TYPE)
%{
    // A macro for creating the function that builds an Array<T> from a numpy array
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##ARRAY_TYPE(PyObject *obj, ARRAY_TYPE *result) {

        if (!TestPyObj_##ARRAY_TYPE((PyObject *) obj)) return false;
        *result = ARRAY_TYPE(PyArray_DIM((PyArrayObject *) (obj),0),
                            (C_TYPE *) PyArray_DATA((PyArrayObject *) (obj)));
        return true;
    }
%}
%{
    DLL_PUBLIC int TypeCheckPyObj_##ARRAY_TYPE(PyObject *obj) {
        if (PyArray_CheckExact(obj)) return 1; else return 0;
    }
%}
%typemap(in) (ARRAY_TYPE &) (ARRAY_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_##ARRAY_TYPE($input, &res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) ARRAY_TYPE & {$1=TypeCheckPyObj_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH ARRAY2D<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro for dealing with the typemaps[in] of Array 2d
%define TYPEMAPIN_ARRAY2D(ARRAY2D_TYPE, C_TYPE, NP_TYPE)
%{
    // A macro for creating the function that builds an Array<T> from a numpy array
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##ARRAY2D_TYPE(PyObject *obj, ARRAY2D_TYPE *result) {

        if (!TestPyObj_##ARRAY2D_TYPE((PyObject *) obj)) return false;
        *result = ARRAY2D_TYPE(PyArray_DIM((PyArrayObject *) (obj), 0),
                             PyArray_DIM((PyArrayObject *) (obj), 1),
                             (C_TYPE *) PyArray_DATA((PyArrayObject *) (obj)));
        return true;
    }
%}
%{
    DLL_PUBLIC int TypeCheckPyObj_##ARRAY2D_TYPE(PyObject *obj) {
        if (PyArray_CheckExact(obj)) return 1; else return 0;
    }
%}
%typemap(in) (ARRAY2D_TYPE &) (ARRAY2D_TYPE res) {
    $1=&res;
    if (!BuildFromPyObj_##ARRAY2D_TYPE($input, &res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) ARRAY2D_TYPE & {$1=TypeCheckPyObj_##ARRAY2D_TYPE($input);}
%enddef

////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of ARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of ArrayList1D
%define TYPEMAPIN_ARRAY_LIST1D(ARRAY_TYPE,ARRAY_LIST1D_TYPE)
%{
    // Now a function that fills an existing ArrayList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##ARRAY_TYPE(PyObject *obj,ARRAY_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a list of Arrays)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!BuildFromPyObj_##ARRAY_TYPE(obj_i,&(list[i]))) return false;
        }
        return true;
    }
%}
%typemap(in) (ARRAY_LIST1D_TYPE &) (ARRAY_LIST1D_TYPE res) {
    $1 = &res;
    if (!BuildFromPyObj_List1d_##ARRAY_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) ARRAY_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of ARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////


// A macro that deals with typemap[in] of ArrayList2D
%define TYPEMAPIN_ARRAY_LIST2D(ARRAY_TYPE,ARRAY_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##ARRAY_TYPE(PyObject *obj,ARRAY_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a 2d-list of Arrays)");
            return false;
        }
        long nCols=0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);

        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of Arrays)");
                return false;
            }
            nCols = PyList_Size(obj_i);
//            else if (nCols != PyList_Size(obj_i)) {
//                PyErr_SetString(PyExc_ValueError,"Failed converting argument to 2d-list of Arrays");
//                return false;
//            }
            list[i].resize(nCols);
            for (long j=0;j<nCols;j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (!BuildFromPyObj_##ARRAY_TYPE(obj_ij,&(list[i][j]))) return false;
            }
        }
        return true;
    }
%}
%typemap(in) (ARRAY_LIST2D_TYPE &) (ARRAY_LIST2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_List2d_##ARRAY_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) ARRAY_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with arrays
%define ARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE, ARRAY2D_TYPE, ARRAY_LIST1D_TYPE, ARRAY_LIST2D_TYPE,
                               C_TYPE, NP_TYPE)
TYPEMAPIN_ARRAY(ARRAY_TYPE,C_TYPE,NP_TYPE);
TYPEMAPIN_ARRAY2D(ARRAY2D_TYPE,C_TYPE,NP_TYPE);
TYPEMAPIN_ARRAY_LIST1D(ARRAY_TYPE,ARRAY_LIST1D_TYPE);
TYPEMAPIN_ARRAY_LIST2D(ARRAY_TYPE,ARRAY_LIST2D_TYPE);
%enddef




