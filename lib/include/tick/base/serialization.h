#ifndef LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
#define LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_

// License: BSD 3 clause

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

namespace tick {

template <typename T>
std::string object_to_string(T* ptr) {
  std::ostringstream ss(std::ios::binary);

  {
    cereal::JSONOutputArchive ar(ss);
    ar(*ptr);
  }

  return ss.str();
}

template <typename T>
void object_from_string(T* ptr, const std::string& data) {
  std::istringstream ss(data, std::ios::binary);

  cereal::JSONInputArchive ar(ss);
  ar(*ptr);
}

}  // namespace tick

#endif  // LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
