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
// Since we are not allowed to directly instantiate Base arrays. It is not
// really a problem.
//

#include "basearray2d.h"

// Instanciations

/**
 * \defgroup SArray2d_typedefs_mod shared array related typedef
 * \brief List of all the instantiations of the shared array2d mechanism and
 * associated shared pointers and 1d and 2d List of these classes
 * @{
 */
/**
 * @}
 */
/** @defgroup sabstractarray2dptr_sub_mod The shared pointer basearray classes
 *  @ingroup SArray2d_typedefs_mod
 * @{
 */

#define SBASE_ARRAY2D_DEFINE_TYPE(TYPE, NAME)                                 \
  typedef std::shared_ptr<BaseArray##NAME##2d> SBaseArray##NAME##2dPtr;       \
  typedef std::vector<SBaseArray##NAME##2dPtr> SBaseArray##NAME##2dPtrList1D; \
  typedef std::vector<SBaseArray##NAME##2dPtrList1D>                          \
      SBaseArray##NAME##2dPtrList2D

SBASE_ARRAY2D_DEFINE_TYPE(double, Double);
SBASE_ARRAY2D_DEFINE_TYPE(float, Float);
SBASE_ARRAY2D_DEFINE_TYPE(int32_t, Int);
SBASE_ARRAY2D_DEFINE_TYPE(uint32_t, UInt);
SBASE_ARRAY2D_DEFINE_TYPE(int16_t, Short);
SBASE_ARRAY2D_DEFINE_TYPE(uint16_t, UShort);
SBASE_ARRAY2D_DEFINE_TYPE(int64_t, Long);
SBASE_ARRAY2D_DEFINE_TYPE(ulong, ULong);
SBASE_ARRAY2D_DEFINE_TYPE(std::atomic<double>, AtomicDouble);
SBASE_ARRAY2D_DEFINE_TYPE(std::atomic<float>, AtomicFloat);

#undef SBASE_ARRAY2D_DEFINE_TYPE

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_
