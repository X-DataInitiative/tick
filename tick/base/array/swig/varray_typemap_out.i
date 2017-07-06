

////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH VARRAY<T> TYPEMAP(out)
//
////////////////////////////////////////////////////////////////////////////////////////

// Here we can use the SARRAY macros !

%define VARRAY_TYPEMAPOUT_MACROS(VARRAYPTR_TYPE, VARRAY_TYPE,
                                 VARRAYPTR_LIST1D_TYPE, VARRAYPTR_LIST2D_TYPE,
                                 C_TYPE, NP_TYPE)

// The check procedure
XARRAY_MISC(VARRAY_TYPE, NP_TYPE);

// Typemaps
TYPEMAPOUT_XARRAYPTR(VARRAY_TYPE, VARRAYPTR_TYPE)
TYPEMAP_XARRAYPTR_LIST1D(VARRAY_TYPE, VARRAYPTR_TYPE, VARRAYPTR_LIST1D_TYPE);
TYPEMAP_XARRAYPTR_LIST2D(VARRAY_TYPE, VARRAYPTR_TYPE, VARRAYPTR_LIST2D_TYPE);

%enddef

