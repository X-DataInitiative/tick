// License: BSD 3 clause

////////////////////////////////////////////////////////////////////////////////////////
//
//  SOME BASIC MACROS
//
////////////////////////////////////////////////////////////////////////////////////////

// A macro for creating a function testing if a numpy object is a numpy 1D-array of a specified type
%define TEST_PYARRAY(ARRAY_TYPE,C_TYPE,NP_TYPE)
%{
    bool TestPyObj_##ARRAY_TYPE(PyObject * obj) {
        
        if (!PyArray_CheckExact(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expecting a dense numpy array");
            return(false);
        }
        
        PyArrayObject *arrayObject = (PyArrayObject *) (obj);
        
        if (!(PyArray_FLAGS(arrayObject) & NPY_ARRAY_C_CONTIGUOUS)) {
            PyErr_SetString(PyExc_ValueError,
                            "Numpy array data should be contiguous (use numpy.ascontiguousarray)");
            return(false);
        }
        
        // Check dimension is 1
        if (PyArray_NDIM(arrayObject) != 1) {
            PyErr_SetString(PyExc_ValueError,"Numpy array should be 1-dimensional");
            return(false);
        }
        
        // Case it is not the right type
        if (PyArray_TYPE(arrayObject) != NP_TYPE || PyArray_ITEMSIZE(arrayObject) != sizeof(C_TYPE))
        {
            PyErr_SetString(PyExc_ValueError,"Expecting a " #C_TYPE " numpy array");
            return(false);
        }
        
        return true;
    }
    %}
%enddef

// A macro for creating a function testing if a numpy object is a numpy 2D-array of a specified type
%define TEST_PYARRAY2D(ARRAY2D_TYPE, C_TYPE, NP_TYPE)
%{
    bool TestPyObj_##ARRAY2D_TYPE(PyObject * obj) {
        
        if (!PyArray_CheckExact(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expecting a dense numpy array");
            return(false);
        }
        
        PyArrayObject *arrayObject = (PyArrayObject *) (obj);
        
        if (!(PyArray_FLAGS(arrayObject) & NPY_ARRAY_C_CONTIGUOUS)) {
            PyErr_SetString(PyExc_ValueError,
                            "Numpy array data should be contiguous (use numpy.ascontiguousarray)");
            return(false);
        }
        
        // Check dimension is 1
        if (PyArray_NDIM(arrayObject) != 2) {
            PyErr_SetString(PyExc_ValueError,"Numpy array should be 2-dimensional");
            return(false);
        }
        
        // Case it is not the right type
        if (PyArray_TYPE(arrayObject) != NP_TYPE || PyArray_ITEMSIZE(arrayObject) != sizeof(C_TYPE))
        {
            PyErr_SetString(PyExc_ValueError,"Expecting a " #C_TYPE " numpy array");
            return(false);
        }
        
        return true;
    }
    %}
%enddef


// A macro for creating a function testing if a python object is a sparse array of a specified type
%define TEST_PYSPARSEARRAY2D(SPARSEARRAY2D_TYPE,C_TYPE,NP_TYPE)
%{
    bool TestPyObj_##SPARSEARRAY2D_TYPE(PyObject * obj, unsigned long *n_rows, unsigned long *n_cols,
                                          INDICE_TYPE **row_indices, INDICE_TYPE **indices,
                                          C_TYPE ** data, unsigned long *size_sparse,
                                          PyObject **obj_indptr1, PyObject **obj_indices1, PyObject **obj_data1) {

        if (PyArray_CheckExact(obj)) {
            PyErr_SetString(PyExc_ValueError,"Expecting a sparse numpy array not a dense numpy array");
            return(false);
        }
        
        PyObject *obj_shape = PyObject_GetAttrString(obj,"shape");
        PyArrayObject *obj_indptr = (PyArrayObject *) PyObject_GetAttrString(obj,"indptr");
        PyArrayObject *obj_indices = (PyArrayObject *) PyObject_GetAttrString(obj,"indices");
        PyArrayObject *obj_data = (PyArrayObject *) PyObject_GetAttrString(obj,"data");
        *obj_indptr1 = (PyObject *) obj_indptr;
        *obj_indices1 = (PyObject *) obj_indices;
        *obj_data1 = (PyObject *) obj_data;

        if (obj_shape == NULL || obj_indptr == NULL || obj_indices == NULL || obj_data == NULL) {
            PyErr_SetString(PyExc_ValueError,"Expecting a 2d sparse numpy array (i.e., a python object with 3 fields 'indptr', 'indices' and 'data')");
            if (obj_shape) Py_DECREF(obj_shape);
            if (obj_indptr) Py_DECREF(obj_indptr);
            if (obj_indices) Py_DECREF(obj_indices);
            if (obj_data) Py_DECREF(obj_data);
            return(false);
        }

        if (!PyArray_IS_C_CONTIGUOUS(obj_data) || !PyArray_IS_C_CONTIGUOUS(obj_indptr) || !PyArray_IS_C_CONTIGUOUS(obj_indices)) {
            PyErr_SetString(PyExc_ValueError,"The fields indptr, indices and data of sparse matrix must be contiguous numpy arrays.");
            Py_DECREF(obj_indptr);Py_DECREF(obj_indices);Py_DECREF(obj_data);Py_DECREF(obj_shape);
            return(false);
        }

        PyObject *obj_nrows = PyTuple_GET_ITEM(obj_shape,0);
        PyObject *obj_ncols = PyTuple_GET_ITEM(obj_shape,1);
        *n_rows = PyLong_AsLong(obj_nrows);
        *n_cols = PyLong_AsLong(obj_ncols);


        if (PyArray_TYPE(obj_data) != NP_TYPE || PyArray_ITEMSIZE(obj_data) != sizeof(C_TYPE))
        {
            PyErr_SetString(PyExc_ValueError,"Expecting a " #C_TYPE " numpy array for data field of sparse matrix");
            Py_DECREF(obj_indptr);Py_DECREF(obj_indices);Py_DECREF(obj_data);Py_DECREF(obj_shape);
            return(false);
        }

        *data = (C_TYPE *) PyArray_DATA(obj_data);

#ifndef _WIN32
        if (PyArray_TYPE(obj_indices) != NPY_INT32 || PyArray_ITEMSIZE(obj_indices) != sizeof(INDICE_TYPE)) {
            PyErr_SetString(PyExc_ValueError,"Expecting 4 bytes integer array for field indices of sparse matrix");
            Py_DECREF(obj_indptr);Py_DECREF(obj_indices);Py_DECREF(obj_data);Py_DECREF(obj_shape);
            return(false);
        }
#endif
        *indices = (INDICE_TYPE *) PyArray_DATA(obj_indices);

#ifndef _WIN32
       if (PyArray_TYPE(obj_indptr) != NPY_INT32 || PyArray_ITEMSIZE(obj_indptr) != sizeof(INDICE_TYPE)) {
            PyErr_SetString(PyExc_ValueError,"Expecting 4 bytes integer array for field indptr of sparse matrix");
            Py_DECREF(obj_indptr);Py_DECREF(obj_indices);Py_DECREF(obj_data);Py_DECREF(obj_shape);
            return(false);
        }
#endif
        *row_indices = (INDICE_TYPE *) PyArray_DATA(obj_indptr);

        *size_sparse = PyArray_DIM(obj_data,0);

        Py_DECREF(obj_indptr);Py_DECREF(obj_indices);Py_DECREF(obj_data);Py_DECREF(obj_shape);
        return true;
    }
%}
%enddef


// A macro for creating a function testing if a numpy object is a 1D/2D list of 1d arrays
%define TEST_ARRAY_LIST(ARRAY_TYPE, ARRAY2D_TYPE)
%{
    int TestPyObj_List1d_##ARRAY_TYPE(PyObject *obj) {
        if (PyList_Check(obj)) return 1; else return 0;
    }
    int TestPyObj_List1d_##ARRAY2D_TYPE(PyObject *obj) {
        if (PyList_Check(obj)) return 1; else return 0;
    }
    int TestPyObj_List2d_##ARRAY_TYPE(PyObject *obj) {
        if (PyList_Check(obj)) return 1; else return 0;
    }
    int TestPyObj_List2d_##ARRAY2D_TYPE(PyObject *obj) {
        if (PyList_Check(obj)) return 1; else return 0;
    }
%}
%enddef


////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

%define TEST_MACROS(ARRAY_TYPE, SPARSEARRAY2D_TYPE, ARRAY2D_TYPE, ARRAY_LIST1D_TYPE, ARRAY_LIST2D_TYPE,C_TYPE, NP_TYPE)

TEST_PYARRAY(ARRAY_TYPE,C_TYPE,NP_TYPE);
TEST_PYARRAY2D(ARRAY2D_TYPE,C_TYPE,NP_TYPE);
TEST_ARRAY_LIST(ARRAY_TYPE, ARRAY2D_TYPE);
TEST_PYSPARSEARRAY2D(SPARSEARRAY2D_TYPE,C_TYPE,NP_TYPE)

%enddef
