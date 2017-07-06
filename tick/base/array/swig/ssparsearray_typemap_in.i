// License: BSD 3 clause



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SPARSEARRAY<T> TYPEMAP(in)
//
//  A Sparse matrix in python is an objet with 4 fields
//      - indptr
//      - shape
//      - indices
//      - data
//
////////////////////////////////////////////////////////////////////////////////////////

%define TYPEMAPIN_SSPARSEARRAY2D(SSPARSEARRAY2D_TYPE, SSPARSEARRAY2DPTR_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE)
%{
    // A macro for creating the function that builds a SparseArray<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SSPARSEARRAY2D_TYPE(PyObject *obj, SSPARSEARRAY2DPTR_TYPE *result) {
        
        unsigned long n_rows, n_cols,size_sparse;
        C_TYPE *data;
        INDICE_TYPE *row_indices, *indices;
        PyObject *obj_indptr, *obj_indices, *obj_data;
        
        if (!TestPyObj_##SPARSEARRAY2D_TYPE((PyObject *) obj,&n_rows,&n_cols,&row_indices,&indices,&data,&size_sparse,&obj_indptr,&obj_indices,&obj_data))
            return false;
        *result = SSPARSEARRAY2D_TYPE::new_ptr(0,0,0);
        (*result)->set_data_indices_rowindices(data,indices,row_indices,n_rows,n_cols,obj_data,obj_indices,obj_indptr);
        return true;
    }
    %}
%{
    DLL_PUBLIC int TypeCheckPyObj_##SSPARSEARRAY2D_TYPE(PyObject *obj) {
        return 1;
    }
    %}
%typemap(in) SSPARSEARRAY2DPTR_TYPE {if (!BuildFromPyObj_##SSPARSEARRAY2D_TYPE($input, &$1)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SSPARSEARRAY2DPTR_TYPE {$1=TypeCheckPyObj_##SPARSEARRAY2D_TYPE($input);}
%enddef



%define TYPEMAPIN_SSPARSEARRAY(SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE)
%{
    // A macro for creating the function that builds a SparseArray<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SSPARSEARRAY_TYPE(PyObject *obj, SSPARSEARRAYPTR_TYPE *result) {
        
        unsigned long n_rows, n_cols,size_sparse;
        C_TYPE *data;
        INDICE_TYPE *row_indices, *indices;
        PyObject *obj_indptr, *obj_indices, *obj_data;
        
        if (!TestPyObj_##SPARSEARRAY2D_TYPE((PyObject *) obj,&n_rows,&n_cols,&row_indices,&indices,&data,&size_sparse,&obj_indptr,&obj_indices,&obj_data))
            return false;
        if (n_rows > 1) {
            PyErr_SetString(PyExc_ValueError,"Expecting a dimension 1 SparseArray");
            return(false);
        }
        *result = SSPARSEARRAY_TYPE::new_ptr(0,0);
        (*result)->set_data_indices(data,indices,n_cols,size_sparse,obj_data,obj_indices);
        return true;
    }
    %}
%{
    DLL_PUBLIC int TypeCheckPyObj_##SSPARSEARRAY_TYPE(PyObject *obj) {
        return 1;
    }
    %}
%typemap(in) SSPARSEARRAYPTR_TYPE {if (!BuildFromPyObj_##SSPARSEARRAY_TYPE($input, &$1)) SWIG_fail;}
%typecheck(SWIG_TYPECHECK_POINTER) SSPARSEARRAYPTR_TYPE {$1=TypeCheckPyObj_##SPARSEARRAY_TYPE($input);}
%enddef




////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with sparse arrays
%define SSPARSEARRAY_TYPEMAPIN_MACROS(SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, SSPARSEARRAY2D_TYPE, SSPARSEARRAY2DPTR_TYPE,
                                      SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE)
TYPEMAPIN_SSPARSEARRAY(SSPARSEARRAY_TYPE, SSPARSEARRAYPTR_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE);
TYPEMAPIN_SSPARSEARRAY2D(SSPARSEARRAY2D_TYPE, SSPARSEARRAY2DPTR_TYPE, SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE)
%enddef
