#ifndef LIB_INCLUDE_TICK_ARRAY_BASEARRAY_H_
#define LIB_INCLUDE_TICK_ARRAY_BASEARRAY_H_

// License: BSD 3 clause

#include "abstractarray1d2d.h"

template <typename T>
class Array;

/*! \class BaseArray
 * \brief Base template class for all the 1d-array (dense and sparse) classes of
 * type `T`. Actually, this class can hold a Array or a SparseArray. So
 * depending on the return value of the method is_dense() (or is_sparse()) one
 * can safely cast an BaseArray to an Array (if is_dense() is true) or a
 * SparseArray (if is_dense() is false). Thus for instance, if a is an
 * BaseArrayDouble that is known to be dense, you can safely write : ArrayDouble
 * &r = static_cast<ArrayDouble &>(a);
 */
template <typename T>
class BaseArray : public AbstractArray1d2d<T> {
 protected:
  using K = typename AbstractArray1d2d<T>::K;
  using AbstractArray1d2d<T>::_size;
  using AbstractArray1d2d<T>::_size_sparse;
  using AbstractArray1d2d<T>::is_data_allocation_owned;
  using AbstractArray1d2d<T>::is_indices_allocation_owned;
  using AbstractArray1d2d<T>::_data;
  using AbstractArray1d2d<T>::_indices;

 public:
  using AbstractArray1d2d<T>::is_dense;
  using AbstractArray1d2d<T>::is_sparse;
  using AbstractArray1d2d<T>::init_to_zero;
  using AbstractArray1d2d<T>::get_data_index;

  //! @brief Main constructor : builds an empty array
  //! \param flag_dense If true then creates a dense array otherwise it is
  //! sparse.
  explicit BaseArray(bool flag_dense = true)
      : AbstractArray1d2d<T>(flag_dense) {}

  //! &brief Copy constructor.
  //! \warning It copies the data creating new allocation owned by the new array
  BaseArray(const BaseArray<T> &other) = default;

  //! @brief Move constructor.
  //! \warning No copy of the data.
  BaseArray(BaseArray<T> &&other) = default;

  //! @brief Assignement operator.
  //! \warning It copies the data creating new allocation owned by the new array
  BaseArray &operator=(const BaseArray<T> &other) = default;

  //! @brief Move assignement.
  //! \warning No copy of the data.
  BaseArray &operator=(BaseArray<T> &&other) = default;

  //! @brief Destructor
  virtual ~BaseArray() {}

 private:
  // Method for printing (called by print() AbstractArray1d2d method
  virtual void _print_dense() const;
  virtual void _print_sparse() const;

 public:
  //! @brief Returns the value of the `i`th element of the array
  inline T value(const ulong i) const {
#ifdef DEBUG_COSTLY_THROW
    if (i < 0 || i >= _size) TICK_BAD_INDEX(0, _size, i);
#endif
    return (is_sparse() ? _value_sparse(i) : _value_dense(i));
  }

  inline T _value_dense(const ulong i) const { return _data[i]; }

  inline T _value_sparse(const ulong j) const {
    ulong i;
    for (i = 0; i < _size_sparse; i++) {
      if (_indices[i] == j) return _data[i];
      if (_indices[i] > j) return 0;
    }

    return 0;
  }

  //! @brief Get last value
  // \return The last value (throw an error if empty)
  inline T last() const {
    if (_size == 0) TICK_ERROR("Array is empty");
    return (is_sparse() ? (_indices == nullptr ? 0 : _data[_size_sparse - 1])
                        : _data[_size - 1]);
  }

  //! @brief Returns the scalar product of the array with `array`
  // defined in file dot.h
  template <typename Y>
  typename std::enable_if<std::is_same<Y, std::atomic<T>>::value, T>::type dot(
      const BaseArray<Y> &array) const;

  template <typename Y = K>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                              !std::is_same<Y, bool>::value &&
                              !std::is_same<T, std::atomic<Y>>::value,
                          Y>::type
  dot(const BaseArray<K> &array) const;

  //! @brief Creates a dense Array from an BaseArray
  //! In terms of allocation owner, there are two cases
  //!     - If the BaseArray is an Array, then the created array is a view (so
  //!     it does not own allocation)
  //!     - If it is a SparseArray, then the created array owns its allocation
  // This method is defined in sparsearray.h
  Array<T> as_array();

  //! @brief Compare two arrays by value - ignores allocation methodology !)
  bool compare(const BaseArray<T> &that) {
    return AbstractArray1d2d<T>::compare(that);
  }
  bool operator==(const BaseArray<T> &that) { return compare(that); }

 private:
  std::string type() const { return (is_dense() ? "Array" : "SparseArray"); }

 public:
  template <class Y = K>
  typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
  set_data_index(size_t index, K value);

  template <class Y = K>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                          !std::is_same<Y, bool>::value &&
                          !std::is_same<T, std::atomic<Y>>::value>::type
  set_data_index(size_t index, K value);
};

// @brief Prints the array
template <typename T>
void BaseArray<T>::_print_dense() const {
  std::cout << "Array[size=" << _size << ",";
  if (_size < 20) {
    for (ulong i = 0; i < _size; ++i) {
      if (i > 0) std::cout << ",";
      std::cout << _data[i];
    }
  } else {
    for (ulong i = 0; i < 10; ++i) std::cout << _data[i] << ",";
    std::cout << "... ";
    for (ulong i = _size - 10; i < _size; ++i) std::cout << "," << _data[i];
  }
  std::cout << "]" << std::endl;
}

template <typename T>
void BaseArray<T>::_print_sparse() const {
  std::cout << "SparseArray[size=" << _size << ",size_sparse=" << _size_sparse
            << ",";
  if (_size_sparse < 20) {
    for (ulong i = 0; i < _size_sparse; ++i) {
      if (i > 0) std::cout << ",";
      std::cout << _indices[i] << "/" << _data[i];
    }
  } else {
    for (ulong i = 0; i < 10; ++i) std::cout << _data[i] << ",";
    std::cout << "... ";
    for (ulong i = _size_sparse - 10; i < _size_sparse; ++i)
      std::cout << _indices[i] << "/" << _data[i];
  }
  std::cout << "]" << std::endl;
}

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

#include <vector>

/**
 * \defgroup Array_typedefs_mod Array related typedef
 * \brief List of all the instantiations of the BaseArray template and 1d and 2d
 * List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup abstractarray_sub_mod The instantiations of the BaseArray template
 *  @ingroup Array_typedefs_mod
 * @{
 */

#define BASEARRAY_DEFINE_TYPE(TYPE, NAME)                       \
  typedef BaseArray<TYPE> BaseArray##NAME;                      \
  typedef std::vector<BaseArray##NAME> BaseArray##NAME##List1D; \
  typedef std::vector<BaseArray##NAME##List1D> BaseArray##NAME##List2D

BASEARRAY_DEFINE_TYPE(double, Double);
BASEARRAY_DEFINE_TYPE(float, Float);
BASEARRAY_DEFINE_TYPE(int32_t, Int);
BASEARRAY_DEFINE_TYPE(uint32_t, UInt);
BASEARRAY_DEFINE_TYPE(int16_t, Short);
BASEARRAY_DEFINE_TYPE(uint16_t, UShort);
BASEARRAY_DEFINE_TYPE(int64_t, Long);
BASEARRAY_DEFINE_TYPE(ulong, ULong);
BASEARRAY_DEFINE_TYPE(std::atomic<double>, AtomicDouble);
BASEARRAY_DEFINE_TYPE(std::atomic<float>, AtomicFloat);

#undef BASEARRAY_DEFINE_TYPE

/**
 * @}
 */

template <typename T>
template <typename Y>
typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
BaseArray<T>::set_data_index(size_t index, typename BaseArray<T>::K value) {
  _data[index].store(value);
}

template <typename T>
template <typename Y>
typename std::enable_if<!std::is_same<T, bool>::value &&
                        !std::is_same<Y, bool>::value &&
                        !std::is_same<T, std::atomic<Y>>::value>::type
BaseArray<T>::set_data_index(size_t index, typename BaseArray<T>::K value) {
  _data[index] = value;
}

#endif  // LIB_INCLUDE_TICK_ARRAY_BASEARRAY_H_
