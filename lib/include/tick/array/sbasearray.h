//
//  sbasearray.h
//

/// @file

#ifndef LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_
#define LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_

// License: BSD 3 clause

//
// IMPORTANT : let us note that we do not define a class SBaseArray<T>
// it would make inheritance much more complicated.
// Since we are not allowed to directly instantiate base arrays. It is not really a problem.
//

#include "basearray.h"

// Instanciations

/**
 * \defgroup SArray_typedefs_mod shared array related typedef
 * \brief List of all the instantiations of the shared array mechanism and associated
 *  shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup sabstractarrayptr_sub_mod The shared pointer basearray classes
 *  @ingroup SArray_typedefs_mod
 * @{
 */

typedef std::shared_ptr<BaseArrayFloat> SBaseArrayFloatPtr;
typedef std::shared_ptr<BaseArrayInt> SBaseArrayIntPtr;
typedef std::shared_ptr<BaseArrayUInt> SBaseArrayUIntPtr;
typedef std::shared_ptr<BaseArrayShort> SBaseArrayShortPtr;
typedef std::shared_ptr<BaseArrayUShort> SBaseArrayUShortPtr;
typedef std::shared_ptr<BaseArrayLong> SBaseArrayLongPtr;
typedef std::shared_ptr<BaseArrayULong> SBaseArrayULongPtr;
typedef std::shared_ptr<BaseArrayDouble> SBaseArrayDoublePtr;

/**
 * @}
 */

/** @defgroup sabstractarrayptrlist1d_sub_mod The classes for dealing with 1d-list of shared pointer basearrays
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SArrayList1D classes
typedef std::vector<SBaseArrayFloatPtr> SBaseArrayFloatPtrList1D;
typedef std::vector<SBaseArrayIntPtr> SBaseArrayIntPtrList1D;
typedef std::vector<SBaseArrayUIntPtr> SBaseArrayUIntPtrList1D;
typedef std::vector<SBaseArrayShortPtr> SBaseArrayShortPtrList1D;
typedef std::vector<SBaseArrayUShortPtr> SBaseArrayUShortPtrList1D;
typedef std::vector<SBaseArrayLongPtr> SBaseArrayLongPtrList1D;
typedef std::vector<SBaseArrayULongPtr> SBaseArrayULongPtrList1D;
typedef std::vector<SBaseArrayDoublePtr> SBaseArrayDoublePtrList1D;

/**
 * @}
 */

/** @defgroup sabstractarrayptrlist2d_sub_mod The classes for dealing with 2d-list of shared pointer basearrays
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SArrayList2D classes
typedef std::vector<SBaseArrayFloatPtrList1D> SBaseArrayFloatPtrList2D;
typedef std::vector<SBaseArrayIntPtrList1D> SBaseArrayIntPtrList2D;
typedef std::vector<SBaseArrayUIntPtrList1D> SBaseArrayUIntPtrList2D;
typedef std::vector<SBaseArrayShortPtrList1D> SBaseArrayShortPtrList2D;
typedef std::vector<SBaseArrayUShortPtrList1D> SBaseArrayUShortPtrList2D;
typedef std::vector<SBaseArrayLongPtrList1D> SBaseArrayLongPtrList2D;
typedef std::vector<SBaseArrayULongPtrList1D> SBaseArrayULongPtrList2D;
typedef std::vector<SBaseArrayDoublePtrList1D> SBaseArrayDoublePtrList2D;

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_
