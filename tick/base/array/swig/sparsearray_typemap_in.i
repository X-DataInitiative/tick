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


%define TYPEMAPIN_SPARSEARRAY2D(SPARSEARRAY2D_TYPE, SPARSEARRAY_TYPE, C_TYPE)
%{
    // A macro for creating the function that builds a SparseArray2d<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SPARSEARRAY2D_TYPE(PyObject *obj, SPARSEARRAY2D_TYPE *result) {

        unsigned long n_rows, n_cols,size_sparse;
        C_TYPE *data;
        INDICE_TYPE *row_indices, *indices;
        PyObject *obj_indptr, *obj_indices, *obj_data;

        if (!TestPyObj_##SPARSEARRAY2D_TYPE((PyObject *) obj,&n_rows,&n_cols,&row_indices,&indices,&data,&size_sparse,&obj_indptr,&obj_indices,&obj_data))
            return false;
        *result = SPARSEARRAY2D_TYPE(n_rows,n_cols,row_indices,indices,data);
        return true;
    }
%}
%{
    DLL_PUBLIC int TypeCheckPyObj_##SPARSEARRAY2D_TYPE(PyObject *obj) {
        return TypeCheckPyObj_##SPARSEARRAY_TYPE(obj);
    }
%}
%typemap(in) (SPARSEARRAY2D_TYPE &) (SPARSEARRAY2D_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_##SPARSEARRAY2D_TYPE($input, &res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) SPARSEARRAY2D_TYPE & {$1=TypeCheckPyObj_##SPARSEARRAY_TYPE($input);}
%enddef




%define TYPEMAPIN_SPARSEARRAY(SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE)
%{
    // A macro for creating the function that builds a SparseArray<T> from a python object
    // (returns false if an error occurred)
    // 'result' will hold the resulting array
    DLL_PUBLIC bool BuildFromPyObj_##SPARSEARRAY_TYPE(PyObject *obj, SPARSEARRAY_TYPE *result) {

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
        *result = SPARSEARRAY_TYPE(n_cols,size_sparse,indices,data);
        return true;
    }
%}
%{
    DLL_PUBLIC int TypeCheckPyObj_##SPARSEARRAY_TYPE(PyObject *obj) {
        PyArrayObject *obj_indptr = (PyArrayObject *) PyObject_GetAttrString(obj,"indptr");
        PyArrayObject *obj_indices = (PyArrayObject *) PyObject_GetAttrString(obj,"indices");
        PyArrayObject *obj_data = (PyArrayObject *) PyObject_GetAttrString(obj,"data");
        
        int res = !(obj_indptr == NULL || obj_indices == NULL || obj_data == NULL);
        
        if (obj_indptr) Py_DECREF(obj_indptr);
        if (obj_indices) Py_DECREF(obj_indices);
        if (obj_data) Py_DECREF(obj_data);
        PyErr_Clear();
        return res;
    }
%}
%typemap(in) (SPARSEARRAY_TYPE &) (SPARSEARRAY_TYPE res) {
    $1 = &res; if (!BuildFromPyObj_##SPARSEARRAY_TYPE($input, &res)) SWIG_fail;
}
%typecheck(SWIG_TYPECHECK_POINTER) SPARSEARRAY_TYPE & {$1=TypeCheckPyObj_##SPARSEARRAY_TYPE($input);}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with sparse arrays
%define SPARSEARRAY_TYPEMAPIN_MACROS(SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE, NP_TYPE)
TYPEMAPIN_SPARSEARRAY(SPARSEARRAY_TYPE, SPARSEARRAY2D_TYPE, C_TYPE);
TYPEMAPIN_SPARSEARRAY2D(SPARSEARRAY2D_TYPE, SPARSEARRAY_TYPE, C_TYPE);
%enddef




