//
//  sabstractarray2d.h
//

/// @file

#ifndef LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_

// License: BSD 3 clause

//
// IMPORTANT : let us note that we do not define a class SBaseArray2d<T>
// it would make inheritance much more complicated.
// Since we are not allowed to directly instantiate Base arrays. It is not really a problem.
//

#include "basearray2d.h"

// Instanciations

/**
 * \defgroup SArray2d_typedefs_mod shared array related typedef
 * \brief List of all the instantiations of the shared array2d mechanism and associated
 *  shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup sabstractarray2dptr_sub_mod The shared pointer basearray classes
 *  @ingroup SArray2d_typedefs_mod
 * @{
 */

typedef std::shared_ptr<BaseArrayFloat2d> SBaseArrayFloat2dPtr;
typedef std::shared_ptr<BaseArrayInt2d> SBaseArrayInt2dPtr;
typedef std::shared_ptr<BaseArrayUInt2d> SBaseArrayUInt2dPtr;
typedef std::shared_ptr<BaseArrayShort2d> SBaseArrayShort2dPtr;
typedef std::shared_ptr<BaseArrayUShort2d> SBaseArrayUShort2dPtr;
typedef std::shared_ptr<BaseArrayLong2d> SBaseArrayLong2dPtr;
typedef std::shared_ptr<BaseArrayULong2d> SBaseArrayULong2dPtr;
typedef std::shared_ptr<BaseArrayDouble2d> SBaseArrayDouble2dPtr;

/**
 * @}
 */

/** @defgroup sabstractarray2dptrlist1d_sub_mod The classes for dealing with 1d-list of shared pointer basearrays2d
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SArrayList1D classes
typedef std::vector<SBaseArrayFloat2dPtr> SBaseArrayFloat2dPtrList1D;
typedef std::vector<SBaseArrayInt2dPtr> SBaseArrayInt2dPtrList1D;
typedef std::vector<SBaseArrayUInt2dPtr> SBaseArrayUInt2dPtrList1D;
typedef std::vector<SBaseArrayShort2dPtr> SBaseArrayShort2dPtrList1D;
typedef std::vector<SBaseArrayUShort2dPtr> SBaseArrayUShort2dPtrList1D;
typedef std::vector<SBaseArrayLong2dPtr> SBaseArrayLong2dPtrList1D;
typedef std::vector<SBaseArrayULong2dPtr> SBaseArrayULong2dPtrList1D;
typedef std::vector<SBaseArrayDouble2dPtr> SBaseArrayDouble2dPtrList1D;

/**
 * @}
 */

/** @defgroup sabstractarray2dptrlist2d_sub_mod The classes for dealing with 2d-list of shared pointer basearrays2d
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SArrayList2D classes
typedef std::vector<SBaseArrayFloat2dPtrList1D> SBaseArrayFloat2dPtrList2D;
typedef std::vector<SBaseArrayInt2dPtrList1D> SBaseArrayInt2dPtrList2D;
typedef std::vector<SBaseArrayUInt2dPtrList1D> SBaseArrayUInt2dPtrList2D;
typedef std::vector<SBaseArrayShort2dPtrList1D> SBaseArrayShort2dPtrList2D;
typedef std::vector<SBaseArrayUShort2dPtrList1D> SBaseArrayUShort2dPtrList2D;
typedef std::vector<SBaseArrayLong2dPtrList1D> SBaseArrayLong2dPtrList2D;
typedef std::vector<SBaseArrayULong2dPtrList1D> SBaseArrayULong2dPtrList2D;
typedef std::vector<SBaseArrayDouble2dPtrList1D> SBaseArrayDouble2dPtrList2D;

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_
