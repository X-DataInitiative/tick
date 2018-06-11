#ifndef LIB_INCLUDE_TICK_ARRAY_ARRAY_H_
#define LIB_INCLUDE_TICK_ARRAY_ARRAY_H_

// License: BSD 3 clause

/** @file */

#include "alloc.h"
#include "tick/base/defs.h"

#include "basearray.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  1d Array
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class SArray;

/*! \class Array
 * \brief Template class for basic non sparse arrays of type `T`.
 *
 * It manages the fact that allocation can be owned by a different object. At
 * this level, if it is not self-owned, the owner could ony be another C
 * structure. It is important to understand that if you need the array to be
 * shared with the Python interpreter then this is ONLY handled by SArray
 * classes through their smart pointers `SArrayPtr`.
 *
 * In order to use this class, you have to understand clearly when copy of the
 * data is or is not performed. Here is a small example.
 *
 *      ArrayDouble c(10); // Creates an array of double of size 10
 *      ArrayDouble b = c; // Copies the data
 *      ArrayDouble e = view(c) // Does not copy the data
 *      b = view(c) // No copy
 *
 */
template <typename T>
class Array : public BaseArray<T> {
 protected:
  using BaseArray<T>::_size;
  using BaseArray<T>::is_data_allocation_owned;
  using BaseArray<T>::_data;

 public:
  using AbstractArray1d2d<T>::is_dense;
  using AbstractArray1d2d<T>::is_sparse;
  using AbstractArray1d2d<T>::init_to_zero;

  using BaseArray<T>::get_data_index;
  using BaseArray<T>::set_data_index;
  using K = typename BaseArray<T>::K;

  //! @brief Constructor for an empty array.
  Array() : BaseArray<T>(true) {}

  /**
   * @brief Constructor for constructing an array of size `size`
   * (and eventually some preallocated data).
   *
   * \param size The size of the corresponding array.
   * \param data A pointer to an array of elements of type `T`. Default is
   * `nullptr`. This pointer should point to allocation created by macro
   * PYSHARED_ALLOC_ARRAY. (and desallocated with PYSHARED_FREE_ARRAY)
   * \return If `data == nullptr`, then a zero-valued array of size `size` is
   * created. Otherwise `data` will be used for the data of the array. \warning
   * If data != nullptr then the created object does not own the allocation
   */
  explicit Array(ulong size, T *data = nullptr);

  /**
   * @brief Constructor for constructing an array from an
   * `std::initializer_list`
   *
   * \param data_list The data that will be stored in this array
   * \warning Data from the `std::initializer_list` is copied. This should not
   * be used on big arrays
   */
  explicit Array(std::initializer_list<K> data_list);

  //! @brief The copy constructor
  Array(const Array<T> &other) = default;

  //! @brief The move constructor
  Array(Array<T> &&other) = default;

  //! @brief The copy assignement operator
  Array<T> &operator=(const Array<T> &other) = default;

  //! @brief The move assignement operator
  Array<T> &operator=(Array<T> &&other) = default;

  //! @brief Destructor
  virtual ~Array() {}

  /**
   * @brief Bracket notation for extraction and assignments.
   *
   * \param i Index in the array
   * \return A reference to the element number \a i
   */
  inline T &operator[](const ulong i) {
#ifdef DEBUG_COSTLY_THROW
    if (i >= _size) TICK_BAD_INDEX(0, _size, i);
#endif
    return _data[i];
  }

  /**
   * @brief Bracket notation for extraction and assignments.
   *
   * \param i Index in the array
   * \return A reference to the element number \a i
   */
  inline const T &operator[](const ulong i) const {
#ifdef DEBUG_COSTLY_THROW
    if (i >= _size) TICK_BAD_INDEX(0, _size, i);
#endif
    return _data[i];
  }

  //! @brief Fill vector with given value
  void fill(const K value);

  /**
   * @brief Sort Array inplace
   * \param increasing : If true, array will be sorted in increasing order,
   * otherwise in decreasing order
   */
  template <typename Y = K>
  typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type sort(
      bool increasing = true);

  template <typename Y = K>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                          !std::is_same<Y, bool>::value &&
                          !std::is_same<T, std::atomic<Y>>::value>::type
  sort(bool increasing = true);

  /**
   * @brief Sort Array with respect to a given function inplace and keep track
   * of index \param index : Array in which the index will be stored \param
   * order_function : function that will be used to compare two values
   */
  template <typename Y = K, typename F>
  typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
  sort_function(Array<uint64_t> &index, F order_function);

  template <typename Y = K, typename F>
  typename std::enable_if<!std::is_same<T, bool>::value &&
                          !std::is_same<Y, bool>::value &&
                          !std::is_same<T, std::atomic<Y>>::value>::type
  sort_function(Array<uint64_t> &index, F order_function);

  /**
   * @brief Sort Array inplace and keep track of index
   * \param index : Array in which the index will be stored
   * \param increasing : If true, array will be sorted in increasing order,
   * otherwise in decreasing order
   */
  template <typename Y = K>
  void sort(Array<ulong> &index, bool increasing = true);

  /**
   * @brief Sort Array with respect to absolute values of entries inplace and
   * keep track of index \param index : Array in which the index will be stored
   * \param increasing : If true, array will be sorted in increasing order,
   * otherwise in decreasing order
   */
  template <typename Y = K>
  void sort_abs(Array<ulong> &index, bool increasing);

  //! @brief Multiply an array by a factor
  //! \param fact : the factor
  //! \param out : if null then the multiplication is performed in-place,
  //!              otherwise it is stored in `out`
  template <typename Y = K>
  void multiply(const K fact, Array<T> *out = nullptr);

  //! @brief Multiply vector x by factor a inplace and increment this by new
  //! vector
  //! @brief Namely, we perform the operator this += x * a
  //! \param x : an Array (can be sparse of dense)
  //! \param a : a scalar of type T
  //! @note Scalar is of type T, meaning that real values will get truncated
  //! before multiplication if T is an integer type
  template <typename Y>
  void mult_incr(const BaseArray<Y> &x, const K a);

  //! @brief Multiply vector x by factor a inplace and fill using this by new
  //! vector
  //! @brief Namely, we perform the operator this = x * a
  //! \param x : an Array (can be sparse of dense)
  //! \param a : a scalar
  //! @note Scalar is of type T, meaning that real values will get truncated
  //! before multiplication if T is an integer type
  void mult_fill(const BaseArray<T> &x, const T a) {
    if (this->size() == x.size()) {
      if (x.is_sparse()) {
        this->fill(0);
        for (ulong j = 0; j < x.size_sparse(); j++) {
          _data[x.indices()[j]] = x.data()[j] * a;
        }
      } else {
        for (ulong j = 0; j < this->size(); ++j) {
          _data[j] = x.data()[j] * a;
        }
      }
    } else {
      TICK_ERROR("Vectors don't have the same size.");
    }
  }

  //! @brief Multiply vector x by factor c inplace and fill using this by new
  //! vector
  //! @brief Namely, we perform the operator this += a * x + b * y
  //! \param x : an Array (can be sparse or dense)
  //! \param a : a scalar
  //! \param y : an Array (can be sparse or dense)
  //! \param b : a scalar
  void mult_add_mult_incr(const BaseArray<T> &x, const K a,
                          const BaseArray<T> &y, const K b) {
    if (this->size() == x.size() && x.size() == y.size()) {
      mult_incr(x, a);
      mult_incr(y, b);
    } else {
      TICK_ERROR("Vectors don't have the same size.");
    }
  }

  bool contains(T value);

  //! @brief Returns a shared pointer to a SArray encapsulating the array
  //! \warning : The ownership of the data is given to the returned structure
  //! THUS the array becomes a view.
  //! \warning : This method cannot be called on a view
  // The definition is in the file sarray.h
  std::shared_ptr<SArray<T>> as_sarray_ptr();
};

// Constructor
template <typename T>
Array<T>::Array(ulong size, T *data1) : BaseArray<T>(true) {
#ifdef DEBUG_ARRAY
  std::cout << "Array Constructor : Array(size=" << size << ",data=" << data1
            << ") --> " << this << std::endl;
#endif
  _size = size;
  // if no one gave us data we allocate it and are now responsible for it
  if (data1 == nullptr) {
    is_data_allocation_owned = true;
    TICK_PYTHON_MALLOC(_data, T, _size);
  } else {
    // Otherwise the one who gave the data is responsible for its allocation
    is_data_allocation_owned = false;
    _data = data1;
  }
}

// initializer_list constructor
template <typename T>
Array<T>::Array(std::initializer_list<typename Array<T>::K> data_list)
    : BaseArray<T>(true) {
  is_data_allocation_owned = true;
  _size = data_list.size();
  TICK_PYTHON_MALLOC(_data, T, _size);
  size_t index = 0;
  for (auto it = data_list.begin(); it != data_list.end(); ++it) {
    set_data_index(index, *it);
    index++;
  }
}

// fill with given value
template <typename T>
void Array<T>::fill(const typename Array<T>::K value) {
  tick::vector_operations<T>{}.set(_size, value, _data);
}

template <typename T>
using value_index = std::pair<T, ulong>;

template <typename T>
bool less_comparator(const value_index<T> &l, const value_index<T> &r) {
  return l.first < r.first;
}

template <typename T>
bool greater_comparator(const value_index<T> &l, const value_index<T> &r) {
  return l.first > r.first;
}

template <typename T>
bool less_comparator_abs(const value_index<T> &l, const value_index<T> &r) {
  return std::fabs(l.first) < std::fabs(r.first);
}

template <typename T>
bool greater_comparator_abs(const value_index<T> &l, const value_index<T> &r) {
  return std::fabs(l.first) > std::fabs(r.first);
}

// sort array inplace and keep track of index
template <typename T>
template <typename Y>
void Array<T>::sort(Array<ulong> &index, bool increasing) {
  if (increasing)
    sort_function<typename Array<T>::K>(index, less_comparator<T>);
  else
    sort_function<typename Array<T>::K>(index, greater_comparator<T>);
}

// sort array in absolute value inplace and keep track of index
template <typename T>
template <typename Y>
void Array<T>::sort_abs(Array<ulong> &index, bool increasing) {
  if (increasing)
    sort_function<typename Array<T>::K>(index, less_comparator_abs<T>);
  else
    sort_function<typename Array<T>::K>(index, greater_comparator_abs<T>);
}

// Multiply an array with a scalar
template <typename T>
template <typename Y>
void Array<T>::multiply(const K fact, Array<T> *out) {
  if (out == nullptr) {
    tick::vector_operations<T>{}.scale(_size, fact, _data);
  } else {
    for (ulong i = 0; i < _size; i++) (*out)[i] = _data[i] * fact;
  }
}

//! @brief Creates a range valued array from `min`to `max` not included (by
//! steps of 1)
template <typename T>
Array<T> arange(const std::int64_t min, const std::int64_t max) {
  ulong n = ulong(max - min >= 0 ? max - min : 0);
  Array<T> a(n);

  for (ulong i = 0; i < n; ++i) {
    a[i] = T(i) + T(min);
  }
  return a;
}

/**
 * @brief Create a sorted copy of the given array
 * \param increasing : If true, array will be sorted in increasing order,
 * otherwise in decreasing order \returns : A sorted copy of this array
 */
template <typename T, typename K = T>
Array<T> sort(Array<T> &array, bool increasing = true) {
  // Create a copy of array
  Array<T> sorted_array(array);
  // Sort the copy inplace
  sorted_array.template sort<K>(increasing);
  return sorted_array;
}

/**
 * @brief Create a sorted copy of the given array and keep track of index
 * \param index : Array in which the index will be stored
 * \param increasing : If true, array will be sorted in increasing order,
 * otherwise in decreasing order \returns : A sorted copy of this array
 */
template <typename T>
Array<T> sort(Array<T> &array, Array<ulong> &index, bool increasing = true) {
  // Create a copy of array
  Array<T> sorted_array(array);
  // Sort the copy inplace keeping track of the index
  sorted_array.sort(index, increasing);
  return sorted_array;
}

/**
 * @brief Create a sorted copy with respect to the absolute values of entries of
 * the given array and keep track of index \param index : Array in which the
 * index will be stored \param increasing : If true, array will be sorted in
 * increasing order, otherwise in decreasing order \returns : A sorted copy of
 * this array
 */
template <typename T, typename K = T>
Array<T> sort_abs(Array<T> &array, Array<ulong> &index,
                  bool increasing = true) {
  // Create a copy of array
  Array<T> sorted_array(array);
  // Sort the copy inplace keeping track of the index
  sorted_array.template sort_abs<K>(index, increasing);
  return sorted_array;
}


/**
 * Array serialization function for binary archives types
 */
template <class Archive, class T>
typename std::enable_if<
    cereal::traits::is_output_serializable<cereal::BinaryData<T>, Archive>::value, void>::type
CEREAL_SAVE_FUNCTION_NAME(Archive &ar, Array<T> const &arr) {
  const bool is_sparse = arr.is_sparse();

  if (is_sparse) {
    std::cerr << typeid(arr).name() << std::endl;
    throw std::runtime_error("this function shouldn't be called");
  }

  ar(CEREAL_NVP(is_sparse));

  ar(cereal::make_size_tag(arr.size()));
  ar(cereal::binary_data(arr.data(), arr.size() * sizeof(T)));

  if (is_sparse) {
    ar(cereal::make_size_tag(arr.size()));
    ar(cereal::binary_data(arr.indices(), arr.size() * sizeof(ulong)));
  }
}

/**
 * Array serialization function for text archives types (XML, JSON)
 */
template <class Archive, class T>
typename std::enable_if<
    !cereal::traits::is_output_serializable<cereal::BinaryData<T>, Archive>::value, void>::type
CEREAL_SAVE_FUNCTION_NAME(Archive &ar, Array<T> const &arr) {
  const bool is_sparse = arr.is_sparse();

  if (is_sparse) {
    std::cerr << typeid(arr).name() << std::endl;
    throw std::runtime_error("this function shouldn't be called");
  }

  ar(CEREAL_NVP(is_sparse));

  {
    ar.setNextName("values");
    ar.startNode();

    ar(cereal::make_size_tag(arr.size_data()));

    for (ulong i = 0; i < arr.size_data(); ++i) ar(arr.get_data_index(i));

    ar.finishNode();
  }

  if (is_sparse) {
    ar.setNextName("indices");
    ar.startNode();

    ar(cereal::make_size_tag(arr.size_sparse()));

    for (ulong i = 0; i < arr.size_sparse(); ++i) ar(arr.indices()[i]);

    ar.finishNode();
  }
}

/**
 * Array deserialization function for binary archives types
 */
template <class Archive, class T>
typename std::enable_if<cereal::traits::is_input_serializable<
                            cereal::BinaryData<T>, Archive>::value,
                        void>::type
CEREAL_LOAD_FUNCTION_NAME(Archive &ar, Array<T> &arr) {
  bool is_sparse = false;

  ar(CEREAL_NVP(is_sparse));

  ulong vectorSize = 0;
  ar(cereal::make_size_tag(vectorSize));

  arr = Array<T>(vectorSize);
  ar(cereal::binary_data(arr.data(),
                         static_cast<std::size_t>(vectorSize) * sizeof(T)));

  if (is_sparse)
    TICK_ERROR("Deserializing sparse arrays is not supported yet.");
}

/**
 * Array deserialization function for text archives types (XML, JSON)
 */
template <class Archive, class T>
typename std::enable_if<!cereal::traits::is_input_serializable<
                            cereal::BinaryData<T>, Archive>::value,
                        void>::type
CEREAL_LOAD_FUNCTION_NAME(Archive &ar, Array<T> &arr) {
  bool is_sparse = false;

  ar(CEREAL_NVP(is_sparse));

  {
    ar.setNextName("values");
    ar.startNode();

    ulong vectorSize = 0;
    ar(cereal::make_size_tag(vectorSize));

    arr = Array<T>(vectorSize);

    typename Array<T>::K k;
    for (ulong i = 0; i < arr.size_data(); ++i) {
      ar(k);
      arr.set_data_index(i, k);
    }

    ar.finishNode();
  }

  if (is_sparse)
    TICK_ERROR("Deserializing sparse arrays is not supported yet.");
}

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

#include <vector>

/**
 * \defgroup Array_typedefs_mod Array related typedef
 * \brief List of all the instantiations of the Array template and 1d and 2d
 * List of these classes
 * @{
 */

#define ARRAY_DEFINE_TYPE(TYPE, NAME)                   \
  typedef Array<TYPE> Array##NAME;                      \
  typedef std::vector<Array##NAME> Array##NAME##List1D; \
  typedef std::vector<Array##NAME##List1D> Array##NAME##List2D

// impacts compile time and not needed for all types (yet)
#define ARRAY_DEFINE_TYPE_SERIALIZE(TYPE, NAME) \
  ARRAY_DEFINE_TYPE(TYPE, NAME);                \
  CEREAL_REGISTER_TYPE(Array##NAME)

ARRAY_DEFINE_TYPE_SERIALIZE(double, Double);
ARRAY_DEFINE_TYPE_SERIALIZE(float, Float);
ARRAY_DEFINE_TYPE(int32_t, Int);
ARRAY_DEFINE_TYPE(uint32_t, UInt);
ARRAY_DEFINE_TYPE(int16_t, Short);
ARRAY_DEFINE_TYPE(uint16_t, UShort);
ARRAY_DEFINE_TYPE(int64_t, Long);
ARRAY_DEFINE_TYPE(ulong, ULong);
ARRAY_DEFINE_TYPE(std::atomic<double>, AtomicDouble);
ARRAY_DEFINE_TYPE(std::atomic<float>, AtomicFloat);

#undef ARRAY_DEFINE_TYPE
#undef ARRAY_DEFINE_TYPE_SERIALIZE

/**
 * @}
 */

/**
 * Output function to log.
 *
 * Usage example:
 *
 * ArrayDouble d(10);
 * TICK_DEBUG() << "MyArray: " << d;
 */
template <typename E, typename T>
tick::TemporaryLog<E> &operator<<(tick::TemporaryLog<E> &log,
                                  const Array<T> &arr) {
  const auto size = arr.size();

  log << "Array[size=" << size << ", " << typeid(T).name();

  if (size <= 20) {
    for (ulong i = 0; i < size; ++i) log << ", " << arr.data()[i];
  } else {
    for (ulong i = 0; i < 10; ++i) log << ", " << arr.data()[i];

    log << ", ...";

    for (ulong i = size - 10; i < size; ++i) log << ", " << arr.data()[i];
  }

  log << "]";

  return log;
}

template <typename T>
template <typename Y, typename F>
typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
Array<T>::sort_function(Array<uint64_t> &index, F order_function) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("Array<std::atomic<T>>");
}

template <typename T>
template <typename Y, typename F>
typename std::enable_if<!std::is_same<T, bool>::value &&
                        !std::is_same<Y, bool>::value &&
                        !std::is_same<T, std::atomic<Y>>::value>::type
Array<T>::sort_function(Array<uint64_t> &index, F order_function) {
  std::vector<value_index<T>> pairs(_size);
  for (uint64_t i = 0; i < _size; ++i) {
    pairs[i].first = _data[i];
    pairs[i].second = i;
  }

  std::sort(pairs.begin(), pairs.end(), order_function);

  for (uint64_t i = 0; i < _size; ++i) {
    _data[i] = pairs[i].first;
    index[i] = pairs[i].second;
  }
}

template <class T>
template <class Y>
void Array<T>::mult_incr(const BaseArray<Y> &x, const typename Array<T>::K a) {
  if (this->size() != x.size()) {
    TICK_ERROR("Vectors don't have the same size.");
  } else {
    if (x.is_sparse()) {
      for (uint64_t j = 0; j < x.size_sparse(); j++) {
        auto x_index_j = x.indices()[j];
        set_data_index(
            x_index_j,
            this->template get_data_index<typename Array<T>::K>(x_index_j) +
                (x.template get_data_index<typename Array<T>::K>(j) * a));
      }
    } else {
      tick::vector_operations<T>{}.mult_incr(this->size(), a, x.data(),
                                             this->data());
    }
  }
}

template<typename T>
bool Array<T>::contains(T value) {
  for (ulong i = 0; i < _size; ++i) {
    if (_data[i] == value) return true;
  }
  return false;
}

template <typename T>
template <typename Y>
typename std::enable_if<std::is_same<T, std::atomic<Y>>::value>::type
Array<T>::sort(bool increasing) {
  TICK_CLASS_DOES_NOT_IMPLEMENT("Array<std::atomic<T>>");
}
template <typename T>
template <typename Y>
typename std::enable_if<!std::is_same<T, bool>::value &&
                        !std::is_same<Y, bool>::value &&
                        !std::is_same<T, std::atomic<Y>>::value>::type
Array<T>::sort(bool increasing) {
  if (increasing)
    std::sort(_data, _data + _size);
  else
    std::sort(_data, _data + _size, std::greater<T>());
}

#endif  // LIB_INCLUDE_TICK_ARRAY_ARRAY_H_
