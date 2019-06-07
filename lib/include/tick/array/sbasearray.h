//
//  sbasearray.h
//

#ifndef LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_
#define LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_

#include "basearray.h"

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

#endif  // LIB_INCLUDE_TICK_ARRAY_SBASEARRAY_H_
