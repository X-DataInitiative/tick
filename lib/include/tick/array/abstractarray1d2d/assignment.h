
#ifndef LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_ASSIGNMENT_H_
#define LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_ASSIGNMENT_H_

// Assignement operator : copies the data
template <typename T, typename MAJ>
AbstractArray1d2d<T, MAJ>& AbstractArray1d2d<T, MAJ>::operator=(const AbstractArray1d2d<T, MAJ> &other) {
#ifdef DEBUG_ARRAY
  std::cout << type() << " Assignement : operator = (AbstractArray1d2d & "
            << &other << ") --> " << this << std::endl;
#endif
  if (this != &other) {
    // Delete owned allocations
    if (is_data_allocation_owned && _data != nullptr) TICK_PYTHON_FREE(_data);
    if (is_indices_allocation_owned && _indices != nullptr)
      TICK_PYTHON_FREE(_indices);
    is_indices_allocation_owned = true;
    is_data_allocation_owned = true;
    _size = other._size;
    _size_sparse = other._size_sparse;
    if (other.is_dense()) {
      TICK_PYTHON_MALLOC(_data, T, _size);
      tick::Allocator<T, typename AbstractArray1d2d<T, MAJ>::K>
        ::memcopy(_data, other._data, _size);
      _indices = nullptr;
    } else {
      if (_size_sparse > 0) {
        TICK_PYTHON_MALLOC(_data, T, _size_sparse);
        tick::Allocator<T, typename AbstractArray1d2d<T, MAJ>::K>
          ::memcopy(_data, other._data, _size_sparse);
        TICK_PYTHON_MALLOC(_indices, INDICE_TYPE, _size_sparse);
        memcpy(_indices, other._indices, sizeof(INDICE_TYPE) * _size_sparse);
      }
    }
  }
  return *this;
}

// Assignement operator from an array with another Major.
template <typename T, typename MAJ>
template <typename RIGHT_MAJ>
typename std::enable_if<!std::is_same<MAJ, RIGHT_MAJ>::value, AbstractArray1d2d<T, MAJ>>::type&
AbstractArray1d2d<T, MAJ>::operator=(const AbstractArray1d2d<T, RIGHT_MAJ> &other) {
#ifdef DEBUG_ARRAY
  std::cout << type() << " Assignement : operator = (AbstractArray1d2d & "
            << &other << ") --> " << this << std::endl;
#endif
  // Delete owned allocations
  if (is_data_allocation_owned && _data != nullptr) TICK_PYTHON_FREE(_data);
  if (is_indices_allocation_owned && _indices != nullptr)
    TICK_PYTHON_FREE(_indices);
  is_indices_allocation_owned = true;
  is_data_allocation_owned = true;
  _size = other.size();
  _size_sparse = other.size_sparse();
  if (other.is_dense()) {
    TICK_PYTHON_MALLOC(_data, T, _size);
    memcpy(_data, other.data(), sizeof(T) * _size);
    _indices = nullptr;
  } else {
    if (_size_sparse > 0) {
      TICK_PYTHON_MALLOC(_data, T, _size_sparse);
      memcpy(_data, other.data(), sizeof(T) * _size_sparse);
      TICK_PYTHON_MALLOC(_indices, INDICE_TYPE, _size_sparse);
      memcpy(_indices, other.indices(), sizeof(INDICE_TYPE) * _size_sparse);
    }
  }
  return *this;
}

// Move assignement operator : No copy
template <typename T, typename MAJ>
AbstractArray1d2d<T, MAJ> &AbstractArray1d2d<T, MAJ>::operator=(
    AbstractArray1d2d<T, MAJ> &&other) {
#ifdef DEBUG_ARRAY
  std::cout << type() << " Move Assignement : operator = (AbstractArray1d2d && "
            << &other << ") --> " << this << std::endl;
#endif
  if (is_data_allocation_owned && _data != nullptr) TICK_PYTHON_FREE(_data);
  if (is_indices_allocation_owned && _indices != nullptr)
    TICK_PYTHON_FREE(_indices);
  _size = other._size;
  is_indices_allocation_owned = other.is_indices_allocation_owned;
  is_data_allocation_owned = other.is_data_allocation_owned;
  _data = other._data;
  _size_sparse = other._size_sparse;
  _indices = other._indices;
  if (other.is_sparse()) other._size_sparse = 0;
  other._data = nullptr;
  other.is_data_allocation_owned = true;
  other._indices = nullptr;
  other.is_indices_allocation_owned = true;
  other._size = 0;
  return *this;
}

#endif  // LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_ASSIGNMENT_H_
