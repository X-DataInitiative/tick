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
// Since we are not allowed to directly instantiate base arrays. It is not
// really a problem.
//

#include "basearray.h"

// Instantiations

/**
 * \defgroup SArray_typedefs_mod shared array related typedef
 * \brief List of all the instantiations of the shared array mechanism and
 * associated shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup sabstractarrayptr_sub_mod The shared pointer basearray classes
 *  @ingroup SArray_typedefs_mod
 * @{
 */

#define SBASE_ARRAY_DEFINE_TYPE(TYPE, NAME)                               \
  typedef std::shared_ptr<BaseArray##NAME> SBaseArray##NAME##Ptr;         \
  typedef std::vector<SBaseArray##NAME##Ptr> SBaseArray##NAME##PtrList1D; \
  typedef std::vector<SBaseArray##NAME##PtrList1D> SBaseArray##NAME##PtrList2D

SBASE_ARRAY_DEFINE_TYPE(double, Double);
SBASE_ARRAY_DEFINE_TYPE(float, Float);
SBASE_ARRAY_DEFINE_TYPE(int32_t, Int);
SBASE_ARRAY_DEFINE_TYPE(uint32_t, UInt);
SBASE_ARRAY_DEFINE_TYPE(int16_t, Short);
SBASE_ARRAY_DEFINE_TYPE(uint16_t, UShort);
SBASE_ARRAY_DEFINE_TYPE(int64_t, Long);
SBASE_ARRAY_DEFINE_TYPE(ulong, ULong);
SBASE_ARRAY_DEFINE_TYPE(std::atomic<double>, AtomicDouble);
SBASE_ARRAY_DEFINE_TYPE(std::atomic<float>, AtomicFloat);

#undef SBASE_ARRAY_DEFINE_TYPE

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_
