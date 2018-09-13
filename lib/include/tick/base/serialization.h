
// License: BSD 3 clause

/*
 *  WARNING : Running clang format on this file
 *   can break the WARNING macro lines
 */

#ifndef LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
#define LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_

#include "cereal/archives/json.hpp"
#include "tick/base/base.h"

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
