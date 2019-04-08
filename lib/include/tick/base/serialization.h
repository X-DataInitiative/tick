
// License: BSD 3 clause

/*
 *  WARNING : Running clang format on this file
 *   can break the WARNING macro lines
 */

#ifndef LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
#define LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_

#include "tick/base/base.h"

// clang-format off
// Don't touch this! - it messes with the "strings" inside the macros
#ifndef TICK_SWIG_INCLUDE
DISABLE_WARNING(unused, exceptions, 42)
DISABLE_WARNING(unused, unused-private-field, 42)
DISABLE_WARNING(delete-non-virtual-dtor, delete-non-virtual-dtor, 42)
#endif
#include "cereal/archives/portable_binary.hpp"
#include <iomanip>
#ifndef TICK_SWIG_INCLUDE
ENABLE_WARNING(unused, exceptions, 42)
ENABLE_WARNING(unused, unused-private-field, 42)
ENABLE_WARNING(delete-non-virtual-dtor, delete-non-virtual-dtor, 42)
#endif
// clang-format on
// Carry on formatting


// Turn serialzed PortableBinary bytes into hex
namespace tick {

inline std::string to_hex(const std::string &bytes) {
  std::stringstream out;
  for (const char &c : bytes) {
    std::stringstream stream;
    stream << std::setfill('0')<< std::setw(2) << std::hex << static_cast<unsigned int>(c);
    out << stream.str() << " ";  // parsing back hex to binary requires spaces for some reason
  }
  return out.str();
}

template <typename T>
inline std::string object_to_string(T* ptr) {
  std::ostringstream ss(std::ios::out | std::ios::binary);
  {
    cereal::PortableBinaryOutputArchive ar(ss);
    ar(*ptr);
  }
  return to_hex(ss.str());
}

inline std::string to_bytes(const std::string &hex) {
  std::istringstream hex_chars_stream(hex);
  std::vector<unsigned char> bytes;
  unsigned int c;
  while (hex_chars_stream >> std::hex >> std::setw(2) >> c) bytes.push_back(c);
  return std::string(bytes.begin(), bytes.end());
}

// Convert hex into bytes to deserialize
template <typename T>
inline void object_from_string(T* ptr, const std::string& hex) {
  std::istringstream ss(to_bytes(hex));
  cereal::PortableBinaryInputArchive ar(ss);
  ar(*ptr);
}
}  // namespace tick

#endif  // LIB_INCLUDE_TICK_BASE_SERIALIZATION_H_
