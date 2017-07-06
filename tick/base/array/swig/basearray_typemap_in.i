// License: BSD 3 clause



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH BASE<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////


%define TYPEMAPIN_BASEARRAY2D(ARRAY2D_TYPE,SPARSEARRAY2D_TYPE,BASEARRAY2D_TYPE, C_TYPE)
%{
    // A macro for creating the function that builds a BaseArray2d<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##BASEARRAY2D_TYPE(PyObject *obj, BASEARRAY2D_TYPE *result) {
        
        // We first try to see if it is a dense array
        if (PyArray_CheckExact(obj)) {
            if (!TestPyObj_##ARRAY2D_TYPE((PyObject *) obj)) return false;
            *result = static_cast<BASEARRAY2D_TYPE>(ARRAY2D_TYPE(PyArray_DIM((PyArrayObject *) (obj), 0),
                                    PyArray_DIM((PyArrayObject *) (obj), 1),
                                    (C_TYPE *) PyArray_DATA((PyArrayObject *) (obj))));
            return true;
        }
        
        // Then we test a sparse one
        unsigned long n_rows, n_cols,size_sparse;
        C_TYPE *data;
        INDICE_TYPE *row_indices, *indices;
        PyObject *obj_indptr, *obj_indices, *obj_data;

        if (!TestPyObj_##SPARSEARRAY2D_TYPE((PyObject *) obj,&n_rows,&n_cols,&row_indices,&indices,&data,&size_sparse,&obj_indptr,&obj_indices,&obj_data)) {
            PyErr_Clear();
            PyErr_SetString(PyExc_ValueError,"Expecting a 2d " #C_TYPE " numpy array or a sparse array");
            return false;
        }
        *result = static_cast<BASEARRAY2D_TYPE>(SPARSEARRAY2D_TYPE(n_rows,n_cols,row_indices,indices,data));
        return true;
    }
%}
%typemap(in) (BASEARRAY2D_TYPE &) (BASEARRAY2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_##BASEARRAY2D_TYPE($input, &res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) BASEARRAY2D_TYPE & {
    $1 = TypeCheckPyObj_##ARRAY2D_TYPE($input) || TypeCheckPyObj_##SPARSEARRAY2D_TYPE($input);
}
%enddef




%define TYPEMAPIN_BASEARRAY(ARRAY_TYPE,SPARSEARRAY_TYPE,SPARSEARRAY2D_TYPE,BASEARRAY_TYPE, C_TYPE)
%{
    // A macro for creating the function that builds a SparseArray<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##BASEARRAY_TYPE(PyObject *obj, BASEARRAY_TYPE *result) {
        
        // We first try to see if it is a dense array
        if (PyArray_CheckExact(obj)) {
            if (!TestPyObj_##ARRAY_TYPE((PyObject *) obj)) return false;
            // This generates a move assignement
            *result = static_cast<BASEARRAY_TYPE>(ARRAY_TYPE(PyArray_DIM((PyArrayObject *) (obj),0),
                                                      (C_TYPE *) PyArray_DATA((PyArrayObject *) (obj))));
            return true;
        }
    
        // Then we test a sparse one
        unsigned long n_rows, n_cols,size_sparse;
        C_TYPE *data;
        INDICE_TYPE *row_indices, *indices;
        PyObject *obj_indptr, *obj_indices, *obj_data;

        if (TestPyObj_##SPARSEARRAY2D_TYPE((PyObject *) obj,&n_rows,&n_cols,&row_indices,&indices,&data,&size_sparse,&obj_indptr,&obj_indices,&obj_data)) {
            if (n_rows > 1) {
                PyErr_SetString(PyExc_ValueError,"Expecting a dimension 1 SparseArray");
                return(false);
            }
            // This generates a move assignement
            *result = static_cast<BASEARRAY_TYPE>(SPARSEARRAY_TYPE(n_cols,size_sparse,indices,data));
            return true;
        }
    
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError,"Expecting a 1d " #C_TYPE " numpy array or a sparse array");
        return false;
    }
%}

%typemap(in) (BASEARRAY_TYPE &) (BASEARRAY_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_##BASEARRAY_TYPE($input, &res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) BASEARRAY_TYPE & {
    $1 = TypeCheckPyObj_##ARRAY_TYPE($input) || TypeCheckPyObj_##SPARSEARRAY_TYPE($input);
}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of BASEARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of BaseArrayList1D
%define TYPEMAPIN_BASEARRAY_LIST1D(ARRAY_TYPE, SPARSEARRAY_TYPE, BASEARRAY_TYPE, BASEARRAY_LIST1D_TYPE)
%{
    // Now a function that fills an existing ArrayList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##BASEARRAY_TYPE(PyObject *obj,BASEARRAY_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a list of BaseArrays)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (TypeCheckPyObj_##ARRAY_TYPE((PyObject *)obj_i)) {
                if (!BuildFromPyObj_##ARRAY_TYPE(obj_i,(ARRAY_TYPE *) &(list[i])))
                    return false;
            }
            else {
                PyErr_Clear();
                if (!BuildFromPyObj_##SPARSEARRAY_TYPE(obj_i, (SPARSEARRAY_TYPE *) &(list[i]))) {
                    return false;
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (BASEARRAY_LIST1D_TYPE &) (BASEARRAY_LIST1D_TYPE res) {
    $1 = &res;
    if (!BuildFromPyObj_List1d_##BASEARRAY_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) BASEARRAY_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of BASEARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////


// A macro that deals with typemap[in] of BaseArrayList2D
%define TYPEMAPIN_BASEARRAY_LIST2D(ARRAY_TYPE, SPARSEARRAY_TYPE, BASEARRAY_TYPE, BASEARRAY_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##BASEARRAY_TYPE(PyObject *obj,BASEARRAY_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a 2d-list of BaseArrays)");
            return false;
        }
        long nCols=0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        
        for (long i=0;i<nRows;i++) {

            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of BaseArrays)");
                return false;
            }
            nCols = PyList_Size(obj_i);
//            else if (nCols != PyList_Size(obj_i)) {
//                PyErr_SetString(PyExc_ValueError,"Failed converting argument to 2d-list of BaseArrays");
//                return false;
//            }
            list[i].resize(nCols);
            for (long j=0;j<nCols;j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (TypeCheckPyObj_##ARRAY_TYPE((PyObject *)obj_ij)) {
                    if (!BuildFromPyObj_##ARRAY_TYPE(obj_ij,(ARRAY_TYPE *) &(list[i][j])))
                        return false;
                }
                else {
                    PyErr_Clear();
                    if (!BuildFromPyObj_##SPARSEARRAY_TYPE(obj_ij, (SPARSEARRAY_TYPE *) &(list[i][j]))) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (BASEARRAY_LIST2D_TYPE &) (BASEARRAY_LIST2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_List2d_##BASEARRAY_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) BASEARRAY_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY_TYPE($input);}
%enddef




////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of BASEARRAY2D<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of BaseArray2dList1D
%define TYPEMAPIN_BASEARRAY2D_LIST1D(ARRAY_TYPE, ARRAY2D_TYPE, SPARSEARRAY2D_TYPE, BASEARRAY2D_TYPE, BASEARRAY2D_LIST1D_TYPE)
%{
    // Now a function that fills an existing ArrayList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##BASEARRAY2D_TYPE(PyObject *obj,BASEARRAY2D_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a list of BaseArray2d)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (TypeCheckPyObj_##ARRAY2D_TYPE((PyObject *)obj_i)) {
                if (!BuildFromPyObj_##ARRAY2D_TYPE(obj_i,(ARRAY2D_TYPE *) &(list[i])))
                    return false;
            }
            else {
                PyErr_Clear();
                if (!BuildFromPyObj_##SPARSEARRAY2D_TYPE(obj_i, (SPARSEARRAY2D_TYPE *) &(list[i]))) {
                    return false;
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (BASEARRAY2D_LIST1D_TYPE &) (BASEARRAY2D_LIST1D_TYPE res) {
    $1 = &res;
    if (!BuildFromPyObj_List1d_##BASEARRAY2D_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) BASEARRAY2D_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of BASEARRAY2D<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////


// A macro that deals with typemap[in] of BaseArrayList2D
%define TYPEMAPIN_BASEARRAY2D_LIST2D(ARRAY_TYPE, ARRAY2D_TYPE, SPARSEARRAY2D_TYPE, BASEARRAY2D_TYPE, BASEARRAY2D_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##BASEARRAY2D_TYPE(PyObject *obj,BASEARRAY2D_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a 2d-list of BaseArray2d)");
            return false;
        }
        long nCols=0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of BaseArray2d)");
                return false;
            }
            nCols = PyList_Size(obj_i);
//            else if (nCols != PyList_Size(obj_i)) {
//                PyErr_SetString(PyExc_ValueError,"Failed converting argument to 2d-list of BaseArray2d");
//                return false;
//            }
            list[i].resize(nCols);
            for (long j=0;j<nCols;j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (TypeCheckPyObj_##ARRAY2D_TYPE((PyObject *)obj_ij)) {
                    if (!BuildFromPyObj_##ARRAY2D_TYPE(obj_ij,(ARRAY2D_TYPE *) &(list[i][j])))
                        return false;
                }
                else {
                    PyErr_Clear();
                    if (!BuildFromPyObj_##SPARSEARRAY2D_TYPE(obj_ij, (SPARSEARRAY2D_TYPE *) &(list[i][j]))) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (BASEARRAY2D_LIST2D_TYPE &) (BASEARRAY2D_LIST2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_List2d_##BASEARRAY2D_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) BASEARRAY2D_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY_TYPE($input);}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with sparse arrays
%define BASEARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE,ARRAY2D_TYPE,SPARSEARRAY_TYPE,
                                       SPARSEARRAY2D_TYPE,BASEARRAY_TYPE,
                                       BASEARRAY2D_TYPE,BASEARRAY_LIST1D_TYPE,
                                       BASEARRAY_LIST2D_TYPE,BASEARRAY2D_LIST1D_TYPE,
                                       BASEARRAY2D_LIST2D_TYPE,
                                       C_TYPE, NP_TYPE)
TYPEMAPIN_BASEARRAY2D(ARRAY2D_TYPE,SPARSEARRAY2D_TYPE,BASEARRAY2D_TYPE, C_TYPE)
TYPEMAPIN_BASEARRAY(ARRAY_TYPE,SPARSEARRAY_TYPE,SPARSEARRAY2D_TYPE,BASEARRAY_TYPE, C_TYPE)
TYPEMAPIN_BASEARRAY_LIST1D(ARRAY_TYPE, SPARSEARRAY_TYPE, BASEARRAY_TYPE, BASEARRAY_LIST1D_TYPE)
TYPEMAPIN_BASEARRAY_LIST2D(ARRAY_TYPE, SPARSEARRAY_TYPE, BASEARRAY_TYPE, BASEARRAY_LIST2D_TYPE)
TYPEMAPIN_BASEARRAY2D_LIST1D(ARRAY_TYPE, ARRAY2D_TYPE, SPARSEARRAY2D_TYPE, BASEARRAY2D_TYPE, BASEARRAY2D_LIST1D_TYPE)
TYPEMAPIN_BASEARRAY2D_LIST2D(ARRAY_TYPE, ARRAY2D_TYPE, SPARSEARRAY2D_TYPE, BASEARRAY2D_TYPE, BASEARRAY2D_LIST2D_TYPE)
%enddef
