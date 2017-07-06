// License: BSD 3 clause



%define TYPEMAPIN_SBASEARRAYPTR(ARRAY_TYPE,SARRAY_TYPE,SPARSEARRAY_TYPE,SPARSEARRAY2D_TYPE,SSPARSEARRAY_TYPE,SARRAYPTR_TYPE,SSPARSEARRAYPTR_TYPE,SBASEARRAYPTR_TYPE,C_TYPE)
%{
    // A macro for creating the function that builds a SAbstractArray<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SBASEARRAYPTR_TYPE(PyObject *obj, SBASEARRAYPTR_TYPE *result) {
    
        // We first try to see if it is a dense array
        if (PyArray_CheckExact(obj)) {
            if (!TestPyObj_##ARRAY_TYPE((PyObject *) obj)) return false;
            SARRAYPTR_TYPE res = SARRAY_TYPE::new_ptr(0);
            res->set_data((C_TYPE *) PyArray_DATA((PyArrayObject *) (obj)), PyArray_DIM((PyArrayObject *) (obj),0),obj);
            *result = res;
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
            SSPARSEARRAYPTR_TYPE res = SSPARSEARRAY_TYPE::new_ptr(0,0);
            res->set_data_indices(data,indices,n_cols,size_sparse,obj_data,obj_indices);
            *result = res;
            return true;
        }
        
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError,"Expecting a 1d " #C_TYPE " numpy array or a sparse array");
        return false;
    }
%}
%typemap(in) SBASEARRAYPTR_TYPE {if (!BuildFromPyObj_##SBASEARRAYPTR_TYPE($input, &$1)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SBASEARRAYPTR_TYPE {
    $1 = TypeCheckPyObj_##ARRAY_TYPE($input) || TypeCheckPyObj_##SPARSEARRAY_TYPE($input);
}
%enddef

%define TYPEMAPIN_SBASEARRAY2DPTR(ARRAY2D_TYPE,SARRAY2D_TYPE,SPARSEARRAY2D_TYPE,SSPARSEARRAY2D_TYPE,SARRAY2DPTR_TYPE,SSPARSEARRAY2DPTR_TYPE,SBASEARRAY2DPTR_TYPE,C_TYPE)
%{
    // A macro for creating the function that builds a SAbstractArray<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SBASEARRAY2DPTR_TYPE(PyObject *obj, SBASEARRAY2DPTR_TYPE *result) {
        
        // We first try to see if it is a dense array
        if (PyArray_CheckExact(obj)) {
            if (!TestPyObj_##ARRAY2D_TYPE((PyObject *) obj)) return false;
            SARRAY2DPTR_TYPE res = SARRAY2D_TYPE::new_ptr(0,0);
            res->set_data((C_TYPE *) PyArray_DATA((PyArrayObject *) (obj)),
                                                  PyArray_DIM((PyArrayObject *) (obj), 0),
                                                  PyArray_DIM((PyArrayObject *) (obj), 1),obj);
            *result = res;
            return true;
        }
        
        // Then we test a sparse one
        unsigned long n_rows, n_cols,size_sparse;
        C_TYPE *data;
        INDICE_TYPE *row_indices, *indices;
        PyObject *obj_indptr, *obj_indices, *obj_data;
        
        if (TestPyObj_##SPARSEARRAY2D_TYPE((PyObject *) obj,&n_rows,&n_cols,&row_indices,&indices,&data,&size_sparse,&obj_indptr,&obj_indices,&obj_data)) {
            SSPARSEARRAY2DPTR_TYPE res = SSPARSEARRAY2D_TYPE::new_ptr(0,0,0);
            res->set_data_indices_rowindices(data,indices,row_indices,n_rows,n_cols,obj_data,obj_indices,obj_indptr);
            *result = res;
            return true;
        }
                          
        PyErr_Clear();
        PyErr_SetString(PyExc_ValueError,"Expecting a 2d " #C_TYPE " numpy array or a sparse array");
        return false;
    }
%}
%typemap(in) SBASEARRAY2DPTR_TYPE {if (!BuildFromPyObj_##SBASEARRAY2DPTR_TYPE($input, &$1)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SBASEARRAY2DPTR_TYPE {
    $1 = TypeCheckPyObj_##ARRAY2D_TYPE($input) || TypeCheckPyObj_##SPARSEARRAY2D_TYPE($input);
}
%enddef
                          



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of SBASEARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of BaseArrayList1D
%define TYPEMAPIN_SBASEARRAYPTR_LIST1D(ARRAY_TYPE,SARRAY_TYPE,SSPARSEARRAY_TYPE,SBASEARRAYPTR_TYPE,SARRAYPTR_TYPE,SSPARSEARRAYPTR_TYPE,SBASEARRAYPTR_LIST1D_TYPE)
%{
    // Now a function that fills an existing ArrayList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##SBASEARRAYPTR_TYPE(PyObject *obj,SBASEARRAYPTR_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a list of SBaseArray)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (TypeCheckPyObj_##ARRAY_TYPE((PyObject *)obj_i)) {
                if (!BuildFromPyObj_##SARRAY_TYPE(obj_i,(SARRAYPTR_TYPE *) &(list[i])))
                    return false;
            }
            else {
                PyErr_Clear();
                if (!BuildFromPyObj_##SSPARSEARRAY_TYPE(obj_i, (SSPARSEARRAYPTR_TYPE *) &(list[i]))) {
                    return false;
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (SBASEARRAYPTR_LIST1D_TYPE &) (SBASEARRAYPTR_LIST1D_TYPE res) {
    $1 = &res;
    if (!BuildFromPyObj_List1d_##SBASEARRAYPTR_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) SBASEARRAYPTR_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of SBASEARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////


// A macro that deals with typemap[in] of SBaseArrayList2D
%define TYPEMAPIN_SBASEARRAYPTR_LIST2D(ARRAY_TYPE,SARRAY_TYPE,SSPARSEARRAY_TYPE,SBASEARRAYPTR_TYPE,SARRAYPTR_TYPE,SSPARSEARRAYPTR_TYPE,SBASEARRAYPTR_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##SBASEARRAYPTR_TYPE(PyObject *obj,SBASEARRAYPTR_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of SBaseArrays)");
            return false;
        }
        long nCols=0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        
        for (long i=0;i<nRows;i++) {
            
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of SBaseArrays)");
                return false;
            }
            nCols = PyList_Size(obj_i);
//            else if (nCols != PyList_Size(obj_i)) {
//                PyErr_SetString(PyExc_ValueError,"Failed converting argument to 2d-list of SAbstractArrays");
//                return false;
//            }
            list[i].resize(nCols);
            for (long j=0;j<nCols;j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (TypeCheckPyObj_##ARRAY_TYPE((PyObject *)obj_ij)) {
                    if (!BuildFromPyObj_##SARRAY_TYPE(obj_ij,(SARRAYPTR_TYPE *) &(list[i][j])))
                        return false;
                }
                else {
                    PyErr_Clear();
                    if (!BuildFromPyObj_##SSPARSEARRAY_TYPE(obj_ij, (SSPARSEARRAYPTR_TYPE *) &(list[i][j]))) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (SBASEARRAYPTR_LIST2D_TYPE &) (SBASEARRAYPTR_LIST2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_List2d_##SBASEARRAYPTR_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) SBASEARRAYPTR_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY_TYPE($input);}
%enddef




////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 1D of SBASEARRAY2D<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro that deals with typemap[in] of SAbstractArray2dList1D
%define TYPEMAPIN_SBASEARRAY2DPTR_LIST1D(ARRAY_TYPE,ARRAY2D_TYPE,SARRAY2D_TYPE,SSPARSEARRAY2D_TYPE,SBASEARRAY2DPTR_TYPE,SARRAY2DPTR_TYPE,SSPARSEARRAY2DPTR_TYPE,SBASEARRAY2DPTR_LIST1D_TYPE)
%{
    // Now a function that fills an existing SAbstractArray2dList1D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List1d_##SBASEARRAY2DPTR_TYPE(PyObject *obj,SBASEARRAY2DPTR_LIST1D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list (it should be a list of SBaseArray2d)");
            return false;
        }
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (TypeCheckPyObj_##ARRAY2D_TYPE((PyObject *)obj_i)) {
                if (!BuildFromPyObj_##SARRAY2D_TYPE(obj_i,(SARRAY2DPTR_TYPE *) &(list[i])))
                    return false;
            }
            else {
                PyErr_Clear();
                if (!BuildFromPyObj_##SSPARSEARRAY2D_TYPE(obj_i, (SSPARSEARRAY2DPTR_TYPE *) &(list[i]))) {
                    return false;
                }
            }
        }
        return true;
    }
%}
%typemap(in) (SBASEARRAY2DPTR_LIST1D_TYPE &) (SBASEARRAY2DPTR_LIST1D_TYPE res) {
    $1 = &res;
    if (!BuildFromPyObj_List1d_##SBASEARRAY2DPTR_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) SBASEARRAY2DPTR_LIST1D_TYPE & {$1=TestPyObj_List1d_##ARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH LIST 2D of SBASEARRAY2D<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////


// A macro that deals with typemap[in] of BaseArrayList2D
%define TYPEMAPIN_SBASEARRAY2DPTR_LIST2D(ARRAY_TYPE,ARRAY2D_TYPE,SARRAY2D_TYPE,SSPARSEARRAY2D_TYPE,SBASEARRAY2DPTR_TYPE,SARRAY2DPTR_TYPE,SSPARSEARRAY2DPTR_TYPE,SBASEARRAY2DPTR_LIST2D_TYPE)
%{
    // Now a function that fills an existing ArrayList2D from a python list of numpy arrays
    DLL_PUBLIC bool BuildFromPyObj_List2d_##SBASEARRAY2DPTR_TYPE(PyObject *obj,SBASEARRAY2DPTR_LIST2D_TYPE &list)
    {
        if (!PyList_Check((PyObject *) obj)) {
            PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of SBaseArray2d)");
            return false;
        }
        long nCols=0;
        long nRows = PyList_Size((PyObject *) obj);
        list.resize(nRows);
        
        for (long i=0;i<nRows;i++) {
            PyObject *obj_i = PyList_GetItem((PyObject *) obj,i);
            if (!PyList_Check(obj_i)) {
                PyErr_SetString(PyExc_ValueError,"Argument is not a list of lists (it should be a 2d-list of SBaseArray2d)");
                return false;
            }
            nCols = PyList_Size(obj_i);
//            else if (nCols != PyList_Size(obj_i)) {
//                PyErr_SetString(PyExc_ValueError,"Failed converting argument to 2d-list of SBaseArray2d");
//                return false;
//            }
            list[i].resize(nCols);
            for (long j=0;j<nCols;j++) {
                PyObject *obj_ij = PyList_GetItem(obj_i,j);
                if (TypeCheckPyObj_##ARRAY2D_TYPE((PyObject *)obj_ij)) {
                    if (!BuildFromPyObj_##SARRAY2D_TYPE(obj_ij,(SARRAY2DPTR_TYPE *) &(list[i][j])))
                        return false;
                }
                else {
                    PyErr_Clear();
                    if (!BuildFromPyObj_##SSPARSEARRAY2D_TYPE(obj_ij, (SSPARSEARRAY2DPTR_TYPE *) &(list[i][j]))) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    %}
%typemap(in) (SBASEARRAY2DPTR_LIST2D_TYPE &) (SBASEARRAY2DPTR_LIST2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_List2d_##SBASEARRAY2DPTR_TYPE($input,res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) SBASEARRAY2DPTR_LIST2D_TYPE & {$1=TestPyObj_List2d_##ARRAY_TYPE($input);}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with sparse arrays
%define SBASEARRAYPTR_TYPEMAPIN_MACROS(ARRAY_TYPE, ARRAY2D_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, SARRAY_TYPE, SARRAY2D_TYPE,
                                        SSPARSEARRAY_TYPE,SSPARSEARRAY2D_TYPE, SARRAYPTR_TYPE, SARRAY2DPTR_TYPE,
                                        SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2DPTR_TYPE,
                                        SBASEARRAYPTR_TYPE,SBASEARRAY2DPTR_TYPE,
                                        SBASEARRAYPTR_LIST1D_TYPE,SBASEARRAYPTR_LIST2D_TYPE,
                                        SBASEARRAY2DPTR_LIST1D_TYPE,SBASEARRAY2DPTR_LIST2D_TYPE,C_TYPE)
TYPEMAPIN_SBASEARRAY2DPTR(ARRAY2D_TYPE,SARRAY2D_TYPE,SPARSEARRAY2D_TYPE,SSPARSEARRAY2D_TYPE,SARRAY2DPTR_TYPE,SSPARSEARRAY2DPTR_TYPE,SBASEARRAY2DPTR_TYPE,C_TYPE)
TYPEMAPIN_SBASEARRAYPTR(ARRAY_TYPE,SARRAY_TYPE,SPARSEARRAY_TYPE,SPARSEARRAY2D_TYPE,SSPARSEARRAY_TYPE,SARRAYPTR_TYPE,SSPARSEARRAYPTR_TYPE,SBASEARRAYPTR_TYPE,C_TYPE)
TYPEMAPIN_SBASEARRAYPTR_LIST1D(ARRAY_TYPE,SARRAY_TYPE,SSPARSEARRAY_TYPE,SBASEARRAYPTR_TYPE,SARRAYPTR_TYPE,SSPARSEARRAYPTR_TYPE,SBASEARRAYPTR_LIST1D_TYPE)
TYPEMAPIN_SBASEARRAYPTR_LIST2D(ARRAY_TYPE,SARRAY_TYPE,SSPARSEARRAY_TYPE,SBASEARRAYPTR_TYPE,SARRAYPTR_TYPE,SSPARSEARRAYPTR_TYPE,SBASEARRAYPTR_LIST2D_TYPE)
TYPEMAPIN_SBASEARRAY2DPTR_LIST1D(ARRAY_TYPE,ARRAY2D_TYPE,SARRAY2D_TYPE,SSPARSEARRAY2D_TYPE,SBASEARRAY2DPTR_TYPE,SARRAY2DPTR_TYPE,SSPARSEARRAY2DPTR_TYPE,SBASEARRAY2DPTR_LIST1D_TYPE)
TYPEMAPIN_SBASEARRAY2DPTR_LIST2D(ARRAY_TYPE,ARRAY2D_TYPE,SARRAY2D_TYPE,SSPARSEARRAY2D_TYPE,SBASEARRAY2DPTR_TYPE,SARRAY2DPTR_TYPE,SSPARSEARRAY2DPTR_TYPE,SBASEARRAY2DPTR_LIST2D_TYPE)
%enddef


