//
//  AbstractArray1d2d.h
//  tests
//
//  Created by bacry on 07/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

#ifndef LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_H_
#define LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_H_

// License: BSD 3 clause

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <type_traits>
#include <typeinfo>

#include "alloc.h"
#include "promote.h"
#include "vector_operations.h"

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
#include <cereal/archives/portable_binary.hpp>
#ifndef TICK_SWIG_INCLUDE
ENABLE_WARNING(unused, exceptions, 42)
ENABLE_WARNING(unused, unused-private-field, 42)
ENABLE_WARNING(delete-non-virtual-dtor, delete-non-virtual-dtor, 42)
#endif
// clang-format on
// Carry on formatting

// This macro defines the type to be used for indices of sparse arrays.
// In python the scipy csr_matrix uses NPY_INT32 to encode them
#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE ulong
#else
#define INDICE_TYPE std::uint32_t
#endif

template <typename T>
struct InnerType {};

#define AtomicOuterType(TYPE)           \
  template <>                           \
  struct InnerType<std::atomic<TYPE>> { \
    using type = TYPE;                  \
  };

#define NonAtomicOuterType(TYPE) \
  template <>                    \
  struct InnerType<TYPE> {       \
    using type = TYPE;           \
  };

AtomicOuterType(double);
AtomicOuterType(float);

NonAtomicOuterType(double);
NonAtomicOuterType(float);

NonAtomicOuterType(int64_t);
NonAtomicOuterType(uint64_t);

NonAtomicOuterType(int32_t);
NonAtomicOuterType(uint32_t);
NonAtomicOuterType(int16_t);
NonAtomicOuterType(uint16_t);

#if defined(__APPLE__) || defined(_WIN32)
NonAtomicOuterType(unsigned long);
#endif

/*! \class AbstractArray1d2d
 * \brief Base template purely virtual class for
 * all the 2d and 1d-array (sparse and dense) classes of type `T`.
 */
template <typename T>
class AbstractArray1d2d {
 protected:
  //! @brief inner type used for most outputs.
  //! Basic usage: `AbstractArray1d2d<std::atomic<double>>::K` is `double`
  using K = typename InnerType<T>::type;

  //! @brief The size of the array : total number of coefficients (zero and non
  //! zero)
  ulong _size;

  //! @brief The array of values of type T (only non zero values will be coded
  //! for sparse arrays)
  T *_data;

  /*! @brief A flag that indicates whether the allocation of the field _data is
   * owned by the array or not
   */
  bool is_data_allocation_owned;

  //! @brief The size of the _data in case of a sparse array
  //! In order to save space it is also used as a flag that indicates
  //! whether the array is dense or sparse.
  //! If _size_sparse > 0 and _indices == nullPtr then it is dense
  ulong _size_sparse;

  //! @brief The array of non zero indices for Sparse arrays.
  //! For 1d-arrays : a strictly increasing array
  //! For 2d-arrays : each row is coded one after the other (an array of row
  //! pointers will be included for 2d-sparse-arrays). This array is of size
  //! _size_sparse
  INDICE_TYPE *_indices;

  /*! @brief A flag that indicates whether the allocation of the field _indices
   * is owned by the array or not
   */
  bool is_indices_allocation_owned;

 public:
  using value_type = T;

  //! @brief Returns true if array is dense
  inline bool is_dense() const {
    return (_indices == nullptr && _size_sparse > 0);
  }

  //! @brief Returns true if array is sparse
  inline bool is_sparse() const { return !is_dense(); }

  //! @brief Main constructor : builds an empty array
  //! \param flag_dense If true then creates a dense array otherwise it is
  //! sparse.
  explicit AbstractArray1d2d(bool flag_dense);

  //! &brief Copy constructor.
  //! \warning It copies the data creating new allocation owned by the new array
  AbstractArray1d2d(const AbstractArray1d2d<T> &other);

  //! @brief Move constructor.
  //! \warning No copy of the data (owner is kept).
  AbstractArray1d2d(AbstractArray1d2d<T> &&other);

  //! @brief Assignement operator.
  //! \warning It copies the data creating new allocation owned by the new array
  AbstractArray1d2d &operator=(const AbstractArray1d2d<T> &other);

  //! @brief Move assignement.
  //! \warning No copy of the data (owner is kept).
  AbstractArray1d2d &operator=(AbstractArray1d2d<T> &&other);

  //! @brief Destructor
  virtual ~AbstractArray1d2d();

  //! @brief Returns the size of the array
  inline ulong size() const { return _size; }

  //! @brief Returns the size of the _data array
  inline ulong size_data() const { return (is_dense() ? _size : _size_sparse); }

  //! @brief Returns the _data of the array
  inline T *data() const { return _data; }

  //! @brief Returns the _indices of the array
  inline INDICE_TYPE *indices() const { return _indices; }

  //! @brief Returns the _size_sparse of the array
  inline ulong size_sparse() const { return _size_sparse; }

  //! @brief Prints the array
  void print() const {
    if (is_sparse())
      _print_sparse();
    else
      _print_dense();
  }

 protected:
  virtual void _print_dense() const = 0;
  virtual void _print_sparse() const = 0;

  //! @brief Compare two arrays by value - ignores allocation methodology !)
  bool compare(const AbstractArray1d2d<T> &that) {
    bool are_equal =
        this->_size == that._size && this->_size_sparse == that._size_sparse;
    if (are_equal && this->_indices && that._indices) {
      for (size_t i = 0; i < this->_size_sparse; i++) {
        are_equal = this->_indices[i] == that._indices[i];
        if (!are_equal) break;
      }
    }
    if (are_equal) {
      for (size_t i = 0; i < size_data(); i++) {
        are_equal = this->_data[i] == that._data[i];
        if (!are_equal) break;
      }
    }
    return are_equal;
  }
  bool operator==(const AbstractArray1d2d<T> &that) { return compare(that); }

 public:
  //! @brief Fill array with zeros (in case of a sparse array we do not
  //! desallocate the various arrays... so it is a "lazy" but quick init !)

  void init_to_zero() {
    tick::vector_operations<T>{}.template set<K>(size_data(), T{0}, _data);
  }

  //! @brief Returns the sum of all the elements of the array
  template <typename Y = K>
  tick::promote_t<K> sum() const;

  //! @brief Returns the minimum element in the array
  template <typename Y = K>
  K min() const;

  //! @brief Returns the maximum element in the array
  template <typename Y = K>
  K max() const;

  //! @brief Compute the squared Euclidean norm of the array

  template <typename Y = K>
  tick::promote_t<K> norm_sq() const;

  //! @brief Multiplication in place with a scalar
  template <typename Y = K>
  typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
  operator*=(const K a);

  template <typename Y = K>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                          !std::is_same<Y, bool>::value &&
                          !std::is_same<T, std::atomic<Y>>::value>::type
  operator*=(const K a);

  template <typename Y = K>
  void multiply(const K a);

  //! @brief Division in place with a scalar
  template <typename Y = K>
  typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
  operator/=(const K a);

  template <typename Y = K>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                          !std::is_same<Y, bool>::value &&
                          !std::is_same<T, std::atomic<Y>>::value>::type
  operator/=(const K a);

  // // useful if data is atomic
  template <typename Y = K>
  typename std::enable_if<std::is_same<T, std::atomic<Y>>::value, Y>::type
  get_data_index(size_t index) const;

  template <typename Y = K>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                              !std::is_same<Y, bool>::value &&
                              !std::is_same<T, std::atomic<Y>>::value,
                          Y>::type
  get_data_index(size_t index) const;

 private:
  std::string type() const { return (is_dense() ? "Array" : "SparseArray"); }
};

// @brief Main constructor
template <typename T>
AbstractArray1d2d<T>::AbstractArray1d2d(bool flag_dense) {
  _size_sparse = 0;
  _size = 0;
  _data = nullptr;
  _indices = nullptr;
  if (flag_dense) _size_sparse = 1;
  is_data_allocation_owned = true;
  is_indices_allocation_owned = true;
}

//! @brief Destructor
template <typename T>
AbstractArray1d2d<T>::~AbstractArray1d2d() {
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
template <typename T>
AbstractArray1d2d<T>::AbstractArray1d2d(const AbstractArray1d2d<T> &other) {
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
template <typename T>
AbstractArray1d2d<T>::AbstractArray1d2d(AbstractArray1d2d<T> &&other) {
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

// Assignement operator : copies the data
template <typename T>
AbstractArray1d2d<T> &AbstractArray1d2d<T>::operator=(
    const AbstractArray1d2d<T> &other) {
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
      memcpy(_data, other._data, sizeof(T) * _size);
      _indices = nullptr;
    } else {
      if (_size_sparse > 0) {
        TICK_PYTHON_MALLOC(_data, T, _size_sparse);
        memcpy(_data, other._data, sizeof(T) * _size_sparse);
        TICK_PYTHON_MALLOC(_indices, INDICE_TYPE, _size_sparse);
        memcpy(_indices, other._indices, sizeof(INDICE_TYPE) * _size_sparse);
      }
    }
  }
  return *this;
}

// Move assignement operator : No copy
template <typename T>
AbstractArray1d2d<T> &AbstractArray1d2d<T>::operator=(
    AbstractArray1d2d<T> &&other) {
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

// @brief Returns the sum of all the elements of the array
template <typename T>
template <typename Y>
tick::promote_t<typename AbstractArray1d2d<T>::K> AbstractArray1d2d<T>::sum()
    const {
  if (_size == 0) TICK_ERROR("Cannot take the sum of an empty array");
  if (size_data() == 0) return 0;

  return tick::vector_operations<T>{}.template sum<Y>(size_data(), _data);
}

// @brief Returns the min
template <typename T>
template <typename Y>
typename AbstractArray1d2d<T>::K AbstractArray1d2d<T>::min() const {
  if (_size == 0) TICK_ERROR("Cannot take the min of an empty array");
  if (size_data() == 0) return 0;
  Y min = _data[0];
  for (ulong i = 1; i < size_data(); ++i) {
    if (_data[i] < min) min = _data[i];
  }

  if (is_sparse() && size_data() != _size)
    return (min > 0 ? 0 : min);
  else
    return min;
}

// @brief Returns the max
template <typename T>
template <typename Y>
typename AbstractArray1d2d<T>::K AbstractArray1d2d<T>::max() const {
  if (_size == 0) TICK_ERROR("Cannot take the max of an empty array");
  if (size_data() == 0) return 0;
  Y max = _data[0];
  for (ulong i = 1; i < size_data(); ++i) {
    if (_data[i] > max) max = _data[i];
  }

  if (is_sparse() && size_data() != _size)
    return (max < 0 ? 0 : max);
  else
    return max;
}

// @brief Compute the squared Euclidean norm of the array
template <typename T>
template <typename Y>
tick::promote_t<typename AbstractArray1d2d<T>::K>
AbstractArray1d2d<T>::norm_sq() const {
  if (_size == 0) TICK_ERROR("Cannot take the norm_sq of an empty array");
  if (size_data() == 0) return 0;

  tick::promote_t<Y> norm_sq{0};
  for (ulong i = 0; i < size_data(); ++i) {
    const Y x_i = get_data_index<Y>(i);
    norm_sq += x_i * x_i;
  }

  return norm_sq;
}

// @brief Multiplication in place with a scalar
template <typename T>
template <typename Y>
typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
AbstractArray1d2d<T>::operator*=(const typename AbstractArray1d2d<T>::K a) {
  if (_size == 0) TICK_ERROR("Cannot apply *= on an empty array");
  if (size_data() == 0) return;

  tick::vector_operations<T>{}.template scale<Y>(size_data(), a, _data);
}

template <typename T>
template <typename Y>
typename std::enable_if<!std::is_same<T, bool>::value &&
                        !std::is_same<Y, bool>::value &&
                        !std::is_same<T, std::atomic<Y>>::value>::type
AbstractArray1d2d<T>::operator*=(const typename AbstractArray1d2d<T>::K a) {
  if (_size == 0) TICK_ERROR("Cannot apply *= on an empty array");
  if (size_data() == 0) return;

  tick::vector_operations<T>{}.scale(size_data(), a, _data);
}

namespace tick {

template <typename T>
void fast_division(
    AbstractArray1d2d<T> &x,
    const typename std::enable_if<std::is_integral<T>::value, T>::type a) {
  for (ulong i = 0; i < x.size_data(); ++i) {
    x.data()[i] /= a;
  }
}

template <typename T>
void fast_division(
    AbstractArray1d2d<T> &x,
    const typename std::enable_if<std::is_floating_point<T>::value, T>::type
        a) {
  x *= (1.0 / double{a});
}

}  // namespace tick

// @brief Division in place with a scalar
template <typename T>
template <typename Y>
typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
AbstractArray1d2d<T>::operator/=(const typename AbstractArray1d2d<T>::K a) {
  if (_size == 0) TICK_ERROR("Cannot apply /= on an empty array");
  if (size_data() == 0) return;

  for (ulong i = 0; i < this->size_data(); ++i) {
    data()[i].store(this->template get_data_index<Y>(i) / a);
  }
}

template <typename T>
template <typename Y>
typename std::enable_if<!std::is_same<T, bool>::value &&
                        !std::is_same<Y, bool>::value &&
                        !std::is_same<T, std::atomic<Y>>::value>::type
AbstractArray1d2d<T>::operator/=(const typename AbstractArray1d2d<T>::K a) {
  if (_size == 0) TICK_ERROR("Cannot apply /= on an empty array");
  if (size_data() == 0) return;

  tick::fast_division<T>(*this, a);
}

template <typename T>
template <typename Y>
void AbstractArray1d2d<T>::multiply(const typename AbstractArray1d2d<T>::K a) {
  if (_size == 0) TICK_ERROR("Cannot apply *= on an empty array");
  if (size_data() == 0) return;

  tick::vector_operations<T>{}.template scale<Y>(size_data(), a, _data);
}

template <typename T>
template <typename Y>
typename std::enable_if<std::is_same<T, std::atomic<Y>>::value, Y>::type
AbstractArray1d2d<T>::get_data_index(size_t index) const {
  return _data[index].load();
}

template <typename T>
template <typename Y>
typename std::enable_if<!std::is_same<T, bool>::value &&
                            !std::is_same<Y, bool>::value &&
                            !std::is_same<T, std::atomic<Y>>::value,
                        Y>::type
AbstractArray1d2d<T>::get_data_index(size_t index) const {
  return _data[index];
}

#endif  // LIB_INCLUDE_TICK_ARRAY_ABSTRACTARRAY1D2D_H_
