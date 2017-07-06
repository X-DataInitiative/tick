// License: BSD 3 clause



////////////////////////////////////////////////////////////////////////////////////////
//
//  DEALING WITH VARRAY<T> TYPEMAP(in)
//
////////////////////////////////////////////////////////////////////////////////////////

// Here we can use the SARRAY macros !

%define VARRAY_TYPEMAPIN_MACROS(ARRAY_TYPE,VARRAYPTR_TYPE, VARRAY_TYPE,VARRAYPTR_LIST1D_TYPE,VARRAYPTR_LIST2D_TYPE,C_TYPE, NP_TYPE)
TYPEMAPIN_SARRAY(VARRAY_TYPE, VARRAYPTR_TYPE, ARRAY_TYPE, C_TYPE, NP_TYPE);
TYPEMAPIN_SARRAY_LIST1D(ARRAY_TYPE,VARRAY_TYPE,VARRAYPTR_LIST1D_TYPE)
TYPEMAPIN_SARRAY_LIST2D(ARRAY_TYPE,VARRAY_TYPE,VARRAYPTR_LIST2D_TYPE)
%enddef

