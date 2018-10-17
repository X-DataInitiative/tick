

#ifndef LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_CONSTRUCTOR_H_
#define LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_CONSTRUCTOR_H_

#include "tick/array/array.h"

// Main constructor
template <typename T, typename MAJ>
AbstractArray1d2d<T, MAJ>::AbstractArray1d2d(bool flag_dense) {
  _size_sparse = 0;
  _size = 0;
  _data = nullptr;
  _indices = nullptr;
  if (flag_dense) _size_sparse = 1;
  is_data_allocation_owned = true;
  is_indices_allocation_owned = true;
}

// Destructor
template <typename T, typename MAJ>
AbstractArray1d2d<T, MAJ>::~AbstractArray1d2d() {
#ifdef DEBUG_ARRAY
  std::cout << type() << " Destructor : ~AbstractArray1d2d on " << this
            << std::endl;
#endif
  // Delete owned allocations
  if (is_data_allocation_owned && _data != nullptr) TICK_PYTHON_FREE(_data);
  if (is_indices_allocation_owned && _indices != nullptr)
    TICK_PYTHON_FREE(_indices);

  _data = nullptr;
  _indices = nullptr;
}

// The copy constructor : copies its data
template <typename T, typename MAJ>
AbstractArray1d2d<T, MAJ>::AbstractArray1d2d(const AbstractArray1d2d<T, MAJ> &other) {
#ifdef DEBUG_ARRAY
  std::cout << other.type()
            << " Copy Constructor : AbstractArray1d2d(AbstractArray1d2d & "
            << &other << ") --> " << this << std::endl;
#endif
  _size = other._size;
  _size_sparse = other._size_sparse;
  is_indices_allocation_owned = true;
  is_data_allocation_owned = true;
  _data = nullptr;
  if (other.is_dense()) {
    TICK_PYTHON_MALLOC(_data, T, _size);
    memcpy(_data, other._data, sizeof(T) * _size);
    _indices = nullptr;
  } else {
    TICK_PYTHON_MALLOC(_data, T, _size_sparse);
    memcpy(_data, other._data, sizeof(T) * _size_sparse);
    TICK_PYTHON_MALLOC(_indices, INDICE_TYPE, _size_sparse);
    memcpy(_indices, other._indices, sizeof(INDICE_TYPE) * _size_sparse);
  }
}

// The move constructor : does not copy the data
template <typename T, typename MAJ>
AbstractArray1d2d<T, MAJ>::AbstractArray1d2d(AbstractArray1d2d<T, MAJ> &&other) {
#ifdef DEBUG_ARRAY
  std::cout << other.type()
            << " Move Constructor : AbstractArray1d2d(AbstractArray1d2d && "
            << &other << ") --> " << this << std::endl;
#endif
  _size = other._size;
  _data = other._data;
  _size_sparse = other._size_sparse;
  _indices = other._indices;
  is_indices_allocation_owned = other.is_indices_allocation_owned;
  is_data_allocation_owned = other.is_data_allocation_owned;
  if (other.is_sparse()) other._size_sparse = 0;
  other._data = nullptr;
  other.is_data_allocation_owned = true;
  other._indices = nullptr;
  other.is_indices_allocation_owned = true;
  other._size = 0;
}



#endif  // LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_CONSTRUCTOR_H_
