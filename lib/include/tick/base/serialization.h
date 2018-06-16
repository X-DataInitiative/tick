
// License: BSD 3 clause

/*
 *  WARNING : Running clang format on this file
 *   can break the WARNING macro lines
 */

#ifndef LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
#define LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_

#include "tick/base/defs.h"

// clang-format off
// Don't touch this!
#ifndef TICK_SWIG_INCLUDE
DISABLE_WARNING(unused, exceptions, 42)
DISABLE_WARNING(unused, unused-private-field, 42)
DISABLE_WARNING(delete-non-virtual-dtor, delete-non-virtual-dtor, 42)
#endif
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#ifndef TICK_SWIG_INCLUDE
ENABLE_WARNING(unused, exceptions, 42)
ENABLE_WARNING(unused, unused-private-field, 42)
ENABLE_WARNING(delete-non-virtual-dtor, delete-non-virtual-dtor, 42)
#endif
// clang-format on
// Carry on formatting

namespace tick {

template <typename T>
inline std::string object_to_string(T* ptr) {
  std::ostringstream ss(std::ios::binary);

  {
    cereal::JSONOutputArchive ar(ss);
    ar(*ptr);
  }

  return ss.str();
}

template <typename T>
inline void object_from_string(T* ptr, const std::string& data) {
  std::istringstream ss(data, std::ios::binary);

  cereal::JSONInputArchive ar(ss);
  ar(*ptr);
}

}  // namespace tick

#endif  // LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
