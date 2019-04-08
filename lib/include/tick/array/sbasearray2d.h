//
//  sabstractarray2d.h
//

/// @file

#ifndef LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_

// License: BSD 3 clause

//
// IMPORTANT : let us note that we do not define a class SBaseArray2d<T, MAJ>
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

#define SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(TYPE, NAME)                         \
  typedef std::shared_ptr<BaseArray##NAME##2d> SBaseArray##NAME##2dPtr;       \
  typedef std::vector<SBaseArray##NAME##2dPtr> SBaseArray##NAME##2dPtrList1D; \
  typedef std::vector<SBaseArray##NAME##2dPtrList1D> SBaseArray##NAME##2dPtrList2D

SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(double, Double);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(float, Float);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(int32_t, Int);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(uint32_t, UInt);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(int16_t, Short);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(uint16_t, UShort);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(int64_t, Long);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(ulong, ULong);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(std::atomic<double>, AtomicDouble);
SBASE_ARRAY_DEFINE_TYPE_SERIALIZE(std::atomic<float>, AtomicFloat);

#undef SBASE_ARRAY2D_DEFINE_TYPE

template <typename T, typename MAJ>
std::shared_ptr<BaseArray2d<T, MAJ>> BaseArray2d<T, MAJ>::as_sarray2d_ptr() {
  if (!is_data_allocation_owned)
    TICK_ERROR(
        "This method cannot be called on an object that does not own its "
        "allocations");

  std::shared_ptr<BaseArray2d<T, MAJ>> arrayptr(std::make_shared<BaseArray2d<T, MAJ>>(*this));
  is_data_allocation_owned = false;
  return arrayptr;
}

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SBASEARRAY2D_H_
