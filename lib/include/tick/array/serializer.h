#ifndef LIB_INCLUDE_TICK_ARRAY_SERIALIZER_H_
#define LIB_INCLUDE_TICK_ARRAY_SERIALIZER_H_

#include "array.h"
#include "sarray.h"
#include "array2d.h"
#include "sarray2d.h"

#include "sparsearray2d.h"
#include "ssparsearray2d.h"

#include <fstream>
#include <cereal/archives/portable_binary.hpp>

template<typename A>
void array_to_file(std::string _file, const A &array) {
  std::ofstream ss(_file, std::ios::out | std::ios::binary);
  {
    cereal::PortableBinaryOutputArchive oarchive(ss);
    oarchive(array);
  }
}

template<typename A>
std::shared_ptr<A> array_from_file(const std::string &_file) {
  auto arr = std::make_shared<A>();
  {
    std::ifstream data(_file, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(data);
    iarchive(*arr.get());
  }
  return arr;
}

void tick_double_array_to_file(std::string _file, const ArrayDouble &array) {
  array_to_file<ArrayDouble>(_file, array);
}

SArrayDoublePtr tick_double_array_from_file(std::string _file) {
  return array_from_file<SArrayDouble>(_file);
}

void tick_double_array2d_to_file(std::string _file, const ArrayDouble2d &array) {
  array_to_file<ArrayDouble2d>(_file, array);
}

SArrayDouble2dPtr tick_double_array2d_from_file(std::string _file) {
  return array_from_file<SArrayDouble2d>(_file);
}

void tick_double_sparse2d_to_file(std::string _file, const SparseArrayDouble2d &array) {
  array_to_file<SparseArrayDouble2d>(_file, array);;
}

SSparseArrayDouble2dPtr tick_double_sparse2d_from_file(std::string _file) {
  return array_from_file<SSparseArrayDouble2d>(_file);
}

#endif  // LIB_INCLUDE_TICK_ARRAY_SERIALIZER_H_
