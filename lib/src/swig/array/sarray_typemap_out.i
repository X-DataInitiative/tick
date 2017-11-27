// License: BSD 3 clause



////////////////////////////////////////////////////////////////////////////////////////
//
//  Some useful macros
//
////////////////////////////////////////////////////////////////////////////////////////

%define XARRAY_MISC(XARRAY_TYPE,NP_TYPE)

// Convert a shared array to an array
// We give the ownership of the allocation to the newly allocated array.
%{
DLL_PUBLIC PyObject *_XArray2NumpyArray(XARRAY_TYPE *sig)
{
    npy_intp dims[1];
    dims[0] = sig->size();

    PyArrayObject *array;

    // We must build an array
    array = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NP_TYPE, sig->data());

    // If data is already owned by somebody else we should inform the newly created array
    if (sig->data_owner()) {
        #ifdef DEBUG_SHAREDARRAY
        std::cout << "Shared Array -> NumpyArray " << sig << " owner is already Python " << sig->data_owner() << std::endl;
        #endif
        #if (NPY_API_VERSION >= 7)
        PyArray_SetBaseObject(array, (PyObject *) sig->data_owner());
        Py_INCREF(sig->data_owner());
        #else
        array->base = (PyObject *) sig->data_owner();
        Py_INCREF(array->base);
        #endif
    }
    // Otherwise the new array should own the data
    else {
        #ifdef DEBUG_SHAREDARRAY
        std::cout << "Shared Array -> NumpyArray " << sig << " owner set to NumpyArray " <<  std::endl;
        #endif
        #if (NPY_API_VERSION >= 7)
        PyArray_ENABLEFLAGS(array,NPY_ARRAY_OWNDATA);
        #else
        PyArray_FLAGS(array) |= NPY_OWNDATA ;
        #endif
        sig->give_data_ownership(array);
    }
    return (PyObject *) array;
}
%}
%enddef

%define SARRAY2D_MISC(XARRAY2D_TYPE, NP_TYPE)
%{
DLL_PUBLIC PyObject *_XArray2d2NumpyArray(XARRAY2D_TYPE *sig)
{
    npy_intp dims[2];
    dims[0] = sig->n_rows();
    dims[1] = sig->n_cols();

    PyArrayObject *array;

    // We must build an array
    array = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NP_TYPE, sig->data());

    // If data is already owned by somebody else we should inform the newly created array
    if (sig->data_owner()) {
        #ifdef DEBUG_SHAREDARRAY
        std::cout << "Shared Array -> NumpyArray " << sig << " owner is already Python " << sig->data_owner() << std::endl;
        #endif
        #if (NPY_API_VERSION >= 7)
        PyArray_SetBaseObject(array, (PyObject *) sig->data_owner());
        Py_INCREF(sig->data_owner());
        #else
        array->base = (PyObject *) sig->data_owner();
        Py_INCREF(array->base);
        #endif
    }
    // Otherwise the new array should own the data
    else {
        #ifdef DEBUG_SHAREDARRAY
        std::cout << "Shared Array -> NumpyArray " << sig << " owner set to NumpyArray " <<  std::endl;
        #endif
        #if (NPY_API_VERSION >= 7)
        PyArray_ENABLEFLAGS(array,NPY_ARRAY_OWNDATA);
        #else
        PyArray_FLAGS(array) |= NPY_OWNDATA ;
        #endif
        sig->give_data_ownership(array);
    }
    return (PyObject *) array;
}
%}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SARRAY<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////


%define TYPEMAPOUT_XARRAYPTR(XARRAY_TYPE,XARRAYPTR_TYPE)
%typemap(out) (XARRAYPTR_TYPE) {
    if (!($1)) {
      $result=Py_None;
      Py_INCREF(Py_None);
    }
    else $result = _XArray2NumpyArray(($1).get());
}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SARRAY2d<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////


%define TYPEMAPOUT_XARRAY2DPTR(XARRAY2D_TYPE, XARRAY2DPTR_TYPE)
%typemap(out) (XARRAY2DPTR_TYPE) {
    if (!($1)) {
      $result=Py_None;
      Py_INCREF(Py_None);
    }
    else $result = _XArray2d2NumpyArray(($1).get());
}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST1D OF SARRAY<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////

%define TYPEMAP_XARRAYPTR_LIST1D(XARRAY_TYPE,XARRAYPTR_TYPE,XARRAYPTR_LIST1D_TYPE)

%{
    DLL_PUBLIC PyObject *BuildPyListFromXArrayPtrList1D_##XARRAY_TYPE(XARRAYPTR_LIST1D_TYPE &list)
    {
        unsigned int i;
        PyObject *result = PyList_New(list.size());
        PyObject *o;
        for (i = 0; i < list.size(); i++) {
            if (!(list[i])) {
                o=Py_None;
                Py_INCREF(Py_None);
            }
            else {
                o = _XArray2NumpyArray((list[i]).get());
            }
            PyList_SetItem(result,i,o);
        }
        return result;
    }
%}
%typemap(out) (XARRAYPTR_LIST1D_TYPE) {return BuildPyListFromXArrayPtrList1D_ ##XARRAY_TYPE($1);}

%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST2D OF SARRAY<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////


// A macro that deals with typemap[in,out] of ArrayList2D
%define TYPEMAP_XARRAYPTR_LIST2D(XARRAY_TYPE,XARRAYPTR_TYPE,XARRAYPTR_LIST2D_TYPE)
%{
// The typemap out
DLL_PUBLIC PyObject *BuildPyListFromXArrayPtrList2D_ ##XARRAY_TYPE(XARRAYPTR_LIST2D_TYPE &list)
{
    PyObject *result = PyList_New(list.size());

    for (unsigned int i = 0;i<list.size();i++) {
        PyObject *o = BuildPyListFromXArrayPtrList1D_##XARRAY_TYPE(list[i]);
        PyList_SetItem(result,i,o);
    }
    return result;
}
%}
%typemap(out) (XARRAYPTR_LIST2D_TYPE) {return BuildPyListFromXArrayPtrList2D_ ##XARRAY_TYPE($1);}

%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with arrays
%define XARRAY_FINAL_MACROS(XARRAYPTR_TYPE, XARRAY_TYPE,
                            XARRAY2DPTR_TYPE, XARRAY2D_TYPE,
                            XARRAYPTR_LIST1D_TYPE, XARRAYPTR_LIST2D_TYPE,
                            C_TYPE,NP_TYPE)

// The check procedure
XARRAY_MISC(XARRAY_TYPE, NP_TYPE);
SARRAY2D_MISC(XARRAY2D_TYPE, NP_TYPE);

// Typemaps
TYPEMAPOUT_XARRAYPTR(XARRAY_TYPE,XARRAYPTR_TYPE)
TYPEMAPOUT_XARRAY2DPTR(XARRAY2D_TYPE, XARRAY2DPTR_TYPE)
TYPEMAP_XARRAYPTR_LIST1D(XARRAY_TYPE,XARRAYPTR_TYPE,XARRAYPTR_LIST1D_TYPE);
TYPEMAP_XARRAYPTR_LIST2D(XARRAY_TYPE,XARRAYPTR_TYPE,XARRAYPTR_LIST2D_TYPE);

%enddef
