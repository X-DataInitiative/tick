// License: BSD 3 clause



////////////////////////////////////////////////////////////////////////////////////////
//
//  Some useful macros
//
////////////////////////////////////////////////////////////////////////////////////////

%define XSPARSEARRAY_MISC(XSPARSEARRAY_TYPE,NP_TYPE)

// Convert a shared array to an array
// We give the ownership of the allocation to the newly allocated array.
%{
DLL_PUBLIC PyObject *_XSparseArray2NumpyArray(XSPARSEARRAY_TYPE *sig)
{
    npy_intp dims[1];
    dims[0] = sig->size();

    PyArrayObject *array;
    PyArrayObject *indices;

    // We must build an array
    array = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NP_TYPE, sig->data());
    indices = (PyArrayObject *) PyArray_SimpleNewFromData(2, dims, NP_TYPE, sig->indices());

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
        sig->give_data_indices_owners(array, indices);
    }
    return (PyObject *) array;
}
%}
%enddef

%define SSPARSEARRAY2D_MISC(XSPARSEARRAY2D_TYPE, NP_TYPE)
%{
DLL_PUBLIC PyObject *_XSparseArray2d2NumpyArray(XSPARSEARRAY2D_TYPE *sig)
{

#ifdef TICK_SPARSE_INDICES_INT64
    auto indice_type = NPY_UINT64;
#else
#define INDICE_TYPE std::uint32_t
    auto indice_type = NPY_UINT32;
#endif

#ifdef DEBUG_SHAREDARRAY
    std::cout << "Sparse2d Shared Array -> size " << sig->size_data() << std::endl;
    std::cout << "Sparse2d Shared Array -> size_sparse " << sig->size_sparse() << std::endl;
    std::cout << "Sparse2d Shared Array -> n_rows " << sig->n_rows() << std::endl;
    std::cout << "Sparse2d Shared Array -> n_cols " << sig->n_cols() << std::endl;
#endif

    npy_intp dims[1];  dims[0] = sig->size_sparse();
    npy_intp rowDim[1]; rowDim[0] = sig->n_rows() + 1;

    PyArrayObject *array = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NP_TYPE, sig->data());
    if(!PyArray_Check(array)) throw std::runtime_error("Array check failed");

    PyArrayObject *indices = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, indice_type, sig->indices());
    if(!PyArray_Check(indices)) throw std::runtime_error("indices check failed");

    PyArrayObject *row_indices = (PyArrayObject *) PyArray_SimpleNewFromData(1, rowDim, indice_type, sig->row_indices());
    if(!PyArray_Check(row_indices)) throw std::runtime_error("row_indices check failed");

    if(!array) throw std::runtime_error("Array failed");
    if(!indices) throw std::runtime_error("indices failed");
    if(!row_indices) throw std::runtime_error("row_indices failed");

    PyObject* tuple = PyTuple_New(3);
    if(!tuple) throw std::runtime_error("tuple new failed");
    if(!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 1 failed");
    
    if(PyTuple_SetItem(tuple, 0, (PyObject *) array)) throw std::runtime_error("tuple PyTuple_SetItem 0 failed");
    if(PyTuple_SetItem(tuple, 1, (PyObject *) indices)) throw std::runtime_error("tuple PyTuple_SetItem 1 failed");
    if(PyTuple_SetItem(tuple, 2, (PyObject *) row_indices)) throw std::runtime_error("tuple PyTuple_SetItem 2 failed");
    if(!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 2 failed");
    
    PyObject* Otuple = PyTuple_New(1);
    if(!Otuple) throw std::runtime_error("Otuple new failed");
    if(PyTuple_SetItem(Otuple, 0, (PyObject *) tuple)) throw std::runtime_error("Otuple PyTuple_SetItem 0 failed");
    if(!PyTuple_Check(tuple)) throw std::runtime_error("Otuple check failed");    

    PyObject* shape = Py_BuildValue("ii", sig->n_rows(), sig->n_cols());
    if(!shape) throw std::runtime_error("Shape tuple new failed");
    if(!PyTuple_Check(shape)) throw std::runtime_error("shape tuple check failed");

    PyObject *dic = PyDict_New();
    if(!dic) throw std::runtime_error("dict new failed");
    if(PyDict_SetItemString(dic, "shape", shape) == -1)
       throw std::runtime_error("shape set failed on dic");
    if(!PyDict_Check(dic)) throw std::runtime_error("dic is no dic");

    PyObject *scipy_sparse_csr, *csr_matrix, *instance;
    scipy_sparse_csr = PyImport_ImportModule("scipy.sparse.csr"); 

    if(!scipy_sparse_csr) throw std::runtime_error("scipy_sparse_csr failed");
    csr_matrix = PyObject_GetAttrString(scipy_sparse_csr, "csr_matrix");
    if(!csr_matrix) throw std::runtime_error("csr_matrix failed");
    if(!PyCallable_Check(csr_matrix)) throw std::runtime_error("csr_matrix check failed");

    // If data is already owned by somebody else we should inform the newly created array
    #ifdef DEBUG_SHAREDARRAY
    std::cout << "NPY_API_VERSION " << std::to_string(NPY_API_VERSION) << std::endl;
    #endif
    if (sig->data_owner()) {
        #ifdef DEBUG_SHAREDARRAY
        std::cout << "Sparse 2d Shared Array -> NumpyArray " << sig << " owner is already Python " << sig->data_owner() << std::endl;
        #endif
        #if (NPY_API_VERSION >= 7)
        PyArray_SetBaseObject(array, (PyObject *) sig->data_owner());
        Py_INCREF(sig->data_owner());
        PyArray_SetBaseObject(indices, (PyObject *) sig->indices_owner());
        Py_INCREF(sig->indices_owner());
        PyArray_SetBaseObject(row_indices, (PyObject *) sig->row_indices_owner());
        Py_INCREF(sig->row_indices_owner());
        #else
        array->base = (PyObject *) sig->data_owner();
        Py_INCREF(array->base);
        indices->base = (PyObject *) sig->indices_owner();
        Py_INCREF(indices->base);
        row_indices->base = (PyObject *) sig->row_indices_owner();
        Py_INCREF(row_indices->base);
        #endif
    }
    // Otherwise the new array should own the data
    else {
        #ifdef DEBUG_SHAREDARRAY
        std::cout << "Sparse2d Shared Array -> NumpyArray " << sig << " owner set to NumpyArray " <<  std::endl;
        #endif
        #if (NPY_API_VERSION >= 7)
        PyArray_ENABLEFLAGS(array,NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS(indices,NPY_ARRAY_OWNDATA);
        PyArray_ENABLEFLAGS(row_indices,NPY_ARRAY_OWNDATA);
        #else
        PyArray_FLAGS(array) |= NPY_OWNDATA ;
        PyArray_FLAGS(indices) |= NPY_OWNDATA ;
        PyArray_FLAGS(row_indices) |= NPY_OWNDATA ;
        #endif
        sig->give_data_indices_rowindices_owners(array, indices, row_indices);
    }
#ifdef DEBUG_SHAREDARRAY
    std::cout << "Sparse2d Shared Array -> array ref count " << ((PyObject *)array)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> indices ref count " << ((PyObject *)indices)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> row_indices ref count " << ((PyObject *)row_indices)->ob_refcnt << std::endl;
#endif

    instance = PyObject_Call(csr_matrix, Otuple, dic);
    if(!instance) throw std::runtime_error("Instnace failed to call object");

#ifdef DEBUG_SHAREDARRAY
    std::cout << "Sparse2d Shared Array -> array ref count " << ((PyObject *)array)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> indices ref count " << ((PyObject *)indices)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> row_indices ref count " << ((PyObject *)row_indices)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> instance ref count " << ((PyObject *)instance)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> Otuple ref count " << ((PyObject *)Otuple)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> tuple ref count " << ((PyObject *)tuple)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> dic ref count " << ((PyObject *)dic)->ob_refcnt << std::endl;
    std::cout << "Sparse2d Shared Array -> shape ref count " << ((PyObject *)shape)->ob_refcnt << std::endl;
#endif
  
    // Usage check for current working scenario to block unexpected outcomes.
    if(((PyObject *)array)->ob_refcnt != 3 
      || ((PyObject *)indices)->ob_refcnt != 2 
      || ((PyObject *)row_indices)->ob_refcnt != 2){
      throw std::runtime_error("SparseArray2d Reference count unexpected in SWIG layer - recompile with -DDEBUG_SHAREDARRAY and check");
    }

    // these lines are required for the following arrays to be de-allocated properly when owned by python
    if(((PyObject *)array)->ob_refcnt > 2) Py_DECREF(array);
    if(((PyObject *)indices)->ob_refcnt > 1) Py_DECREF(indices);
    if(((PyObject *)row_indices)->ob_refcnt > 1) Py_DECREF(row_indices);

    return (PyObject *) instance;
}
%}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SSPARSEARRAY<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////


%define TYPEMAPOUT_XSPARSEARRAYPTR(XSPARSEARRAY_TYPE,XSPARSEARRAYPTR_TYPE)
%typemap(out) (XSPARSEARRAYPTR_TYPE) {
    if (!($1)) {
      $result=Py_None;
      Py_INCREF(Py_None);
    }
    else $result = _XSparseArray2NumpyArray(($1).get());
}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH SSPARSEARRAY2d<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////


%define TYPEMAPOUT_XSPARSEARRAY2DPTR(XSPARSEARRAY2D_TYPE, XSPARSEARRAY2DPTR_TYPE)
%typemap(out) (XSPARSEARRAY2DPTR_TYPE) {
    if (!($1)) {
      $result=Py_None;
      Py_INCREF(Py_None);
    }
    else $result = _XSparseArray2d2NumpyArray(($1).get());
}
%enddef



////////////////////////////////////////////////////////////////////////////////////////
//
//  THE FINAL MACRO
//
////////////////////////////////////////////////////////////////////////////////////////

// The final macro for dealing with arrays
%define XSPARSEARRAY_FINAL_MACROS(XSPARSEARRAYPTR_TYPE, XSPARSEARRAY_TYPE,
                            XSPARSEARRAY2DPTR_TYPE, XSPARSEARRAY2D_TYPE,
                            C_TYPE,NP_TYPE)

// The check procedure
XSPARSEARRAY_MISC(XSPARSEARRAY_TYPE, NP_TYPE);
SSPARSEARRAY2D_MISC(XSPARSEARRAY2D_TYPE, NP_TYPE);

// Typemaps
TYPEMAPOUT_XSPARSEARRAYPTR(XSPARSEARRAY_TYPE,XSPARSEARRAYPTR_TYPE)
TYPEMAPOUT_XSPARSEARRAY2DPTR(XSPARSEARRAY2D_TYPE, XSPARSEARRAY2DPTR_TYPE)
TYPEMAP_XSPARSEARRAYPTR_LIST1D(XSPARSEARRAY_TYPE,XSPARSEARRAYPTR_TYPE,XSPARSEARRAYPTR_LIST1D_TYPE);
TYPEMAP_XSPARSEARRAYPTR_LIST2D(XSPARSEARRAY_TYPE,XSPARSEARRAYPTR_TYPE,XSPARSEARRAYPTR_LIST2D_TYPE);

%enddef
