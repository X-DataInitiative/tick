#ifndef LIB_INCLUDE_TICK_ARRAY_ARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_ARRAY2D_H_

// License: BSD 3 clause

/** @file */

#include "array.h"
#include "basearray2d.h"

template <typename T, typename MAJ = RowMajor>
class SArray2d;
template <typename T, typename MAJ = RowMajor>
class SparseArray2d;

/*! \class Array2d
 * \brief Template class for basic non sparse 2d arrays of type `T`.
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
 *      ArrayDouble2d c(10, 10); // Creates an array of double of size 10x10
 *      ArrayDouble b = c; // Copies the data
 *      ArrayDouble e = view(c) // Does not copy the data
 *      b = view(c) // No copy
 *
 */
template <typename T, typename MAJ>
class Array2d : public BaseArray2d<T, MAJ> {
 protected:
  using BaseArray2d<T, MAJ>::_size;
  using BaseArray2d<T, MAJ>::is_data_allocation_owned;
  using BaseArray2d<T, MAJ>::is_indices_allocation_owned;
  using BaseArray2d<T, MAJ>::_data;
  using BaseArray2d<T, MAJ>::_indices;
  using BaseArray2d<T, MAJ>::_row_indices;
  using BaseArray2d<T, MAJ>::_n_cols;
  using BaseArray2d<T, MAJ>::_n_rows;
  using K = typename BaseArray2d<T, MAJ>::K;

  // NB: We always have _n_rows * _n_cols == _size

 public:
  using AbstractArray1d2d<T, MAJ>::is_dense;
  using AbstractArray1d2d<T, MAJ>::is_sparse;
  using AbstractArray1d2d<T, MAJ>::init_to_zero;

  //! @brief Constructor for an empty array.
  Array2d() : BaseArray2d<T, MAJ>(true) {}

  /**
   * @brief Constructor for constructing a 2d array of size `n_rows, n_cols`
   * (and eventually some preallocated data).
   *
   * \param n_rows The number of rows
   * \param n_cols The number of cols
   * \param data A pointer to an array of elements of type `T`. Default is
   * `nullptr`. This pointer should point to allocation created by macro
   * PYSHARED_ALLOC_ARRAY(and desallocated with PYSHARED_FREE_ARRAY)
   *
   * \return If `data == nullptr`, then a zero-valued array is created.
   * Otherwise `data` will be used for the data of the array.
   * \warning If data != nullptr then the created object does not own the
   * allocation
   */
  explicit Array2d(ulong n_rows, ulong n_cols, T* data = nullptr);

  //! @brief The copy constructor
  Array2d(const Array2d<T, MAJ>& other) = default;

  //! @brief The move constructor
  Array2d(Array2d<T, MAJ>&& other) = default;

  //! @brief The copy assignement operator
  Array2d<T, MAJ>& operator=(const Array2d<T, MAJ>& other) = default;

  //! @brief The move assignement operator
  Array2d<T, MAJ>& operator=(Array2d<T, MAJ>&& other) = default;

  //! @brief Destructor
  virtual ~Array2d() {}

  //! @brief Fill vector with given value
  void fill(T value);

  //! @brief Multiply matrix x by factor a inplace and increment this by new
  //! matrix
  //! @brief Namely, we perform the operator this += x * a
  //! \param x : an Array2d
  //! \param a : a scalar of type T
  //! @note Scalar is of type T, meaning that real values will get truncated
  //! before multiplication if T is an integer type
  void mult_incr(const Array2d<T, MAJ>& x, const T a);

  //! @brief Multiply matrix x by factor a inplace and fill using this by new
  //! matrix
  //! @brief Namely, we perform the operator this = x * a
  //! \param x : an Array2d
  //! \param a : a scalar
  //! @note Scalar is of type T, meaning that real values will get truncated
  //! before multiplication if T is an integer type
  void mult_fill(const Array2d<T, MAJ>& x, const T a);

  //! @brief Multiply matrix x by factor c inplace and fill using this by new
  //! matrix
  //! @brief Namely, we perform the operator this += a * x + b * y
  //! \param x : an Array2d
  //! \param a : a scalar
  //! \param y : an Array2d
  //! \param b : a scalar
  void mult_add_mult_incr(const Array2d<T, MAJ>& x, const T a, const Array2d<T, MAJ>& y,
                          const T b);

  /**
   * @brief Bracket notation for extraction and assignments. We use 1d index.
   *
   * \param i Index in the array (i = r*n_cols+c, where r (c) is the row
   * (column) number \return A reference to the element number \a i
   */
  T& operator[](const ulong i) {
#ifdef DEBUG_COSTLY_THROW
    if (i >= _size) TICK_BAD_INDEX(0, _size, i);
#endif

    return _data[i];
  }

  /**
   * @brief Bracket notation for extraction and assignments. We use 1d index.
   *
   * Const version
   *
   * \param i Index in the array (i = r*n_cols+c, where r (c) is the row
   * (column) number \return A reference to the element number \a i
   */
  const T& operator[](const ulong i) const {
#ifdef DEBUG_COSTLY_THROW
    if (i >= _size) TICK_BAD_INDEX(0, _size, i);
#endif

    return _data[i];
  }

  /**
   * Operator to get value at position (i, j)
   *
   * @param i Row index
   * @param j Column index
   * @return Value at position (i, j)
   */
  T& operator()(const ulong i, const ulong j) {
    const ulong idx = i * this->n_cols() + j;

#ifdef DEBUG_COSTLY_THROW
    if (i >= this->n_rows() || j >= this->n_cols())
      TICK_BAD_INDEX(0, _size, idx);
#endif

    return _data[idx];
  }

  /**
   * Operator to get value at position (i, j)
   *
   * Const version
   *
   * @param i Row index
   * @param j Column index
   * @return Value at position (i, j)
   */
  const T& operator()(const ulong i, const ulong j) const {
    const ulong idx = i * this->n_cols() + j;

#ifdef DEBUG_COSTLY_THROW
    if (i >= this->n_rows() || j >= this->n_cols())
      TICK_BAD_INDEX(0, _size, idx);
#endif

    return _data[idx];
  }

  //! @brief Returns a shared pointer to a SArray2d encapsulating the array
  //! \warning : The ownership of the data is given to the returned structure
  //! THUS the array *this becomes a view.
  //! \warning : This method cannot be called on a view
  // The definition is in the file sarray.h
  std::shared_ptr<SArray2d<T, MAJ>> as_sarray2d_ptr();

 public:
  bool compare(const Array2d<T, MAJ>& that) const {
    bool are_equal = BaseArray2d<T, MAJ>::compare(that);
    return are_equal;
  }

  bool operator==(const Array2d<T, MAJ>& that) const { return compare(that); }


  template <class Archive>
  void save(Archive& ar) const {
    const bool is_sparse = this->is_sparse();
    if (is_sparse)
      throw std::runtime_error("Array2d<T, MAJ>::save - this function shouldn't be called");
    ar(CEREAL_NVP(is_sparse));
    ar(_n_cols, _n_rows);
    ar(cereal::make_size_tag(_size));
    ar(cereal::binary_data(_data, _size * sizeof(T)));
  }

  template <class Archive>
  void load(Archive& ar) {
    bool is_sparse = false;
    ar(CEREAL_NVP(is_sparse));
    if (is_sparse)
      throw std::runtime_error("Array2d<T, MAJ>::load - this function shouldn't be called");
    ar(_n_cols, _n_rows);
    ar(cereal::make_size_tag(_size));
    this->is_data_allocation_owned = true;
    TICK_PYTHON_MALLOC(this->_data, T, _size);
    ar(cereal::binary_data(_data, _size * sizeof(T)));
  }
};

// Constructor
template <typename T, typename MAJ>
Array2d<T, MAJ>::Array2d(ulong n_rows, ulong n_cols, T* data)
    : BaseArray2d<T, MAJ>(true) {
#ifdef DEBUG_ARRAY
  std::cout << "Array2d Constructor : Array(n_rows=" << n_rows
            << ", n_cols=" << n_cols << ",data=" << data << ") --> " << this
            << std::endl;
#endif
  _n_cols = n_cols;
  _n_rows = n_rows;
  _size = n_cols * n_rows;
  // if no one gave us data we allocate it and are now responsible for it
  if (data == nullptr) {
    is_data_allocation_owned = true;
    TICK_PYTHON_MALLOC(_data, T, _size);
  } else {
    // Otherwise the one who gave the data is responsible for its allocation
    is_data_allocation_owned = false;
    _data = data;
  }
}

// fill with given value
template <typename T, typename MAJ>
void Array2d<T, MAJ>::fill(T value) {
  tick::vector_operations<T>{}.set(_size, value, _data);
}

template <typename T, typename MAJ>
void Array2d<T, MAJ>::mult_incr(const Array2d<T, MAJ>& x, const T a) {
  Array<T> this_array = Array<T>(this->size(), this->data());
  Array<T> x_array = Array<T>(x.size(), x.data());
  this_array.mult_incr(x_array, a);
}

template <typename T, typename MAJ>
void Array2d<T, MAJ>::mult_fill(const Array2d<T, MAJ>& x, const T a) {
  Array<T> this_array = Array<T>(this->size(), this->data());
  Array<T> x_array = Array<T>(x.size(), x.data());
  this_array.mult_fill(x_array, a);
}

template <typename T, typename MAJ>
void Array2d<T, MAJ>::mult_add_mult_incr(const Array2d<T, MAJ>& x, const T a,
                                    const Array2d<T, MAJ>& y, const T b) {
  Array<T> this_array = Array<T>(this->size(), this->data());
  Array<T> x_array = Array<T>(x.size(), x.data());
  Array<T> y_array = Array<T>(y.size(), y.data());
  this_array.mult_add_mult_incr(x_array, a, y_array, b);
}

/**
 * The various instances of this template
 */
#define ARRAY_DEFINE_TYPE(TYPE, NAME)                         \
  typedef Array2d<TYPE> Array##NAME##2d;                      \
  typedef Array2d<TYPE, ColMajor> ColMajArray##NAME##2d;      \
  typedef std::vector<Array##NAME##2d> Array##NAME##2dList1D; \
  typedef std::vector<Array##NAME##2dList1D> Array##NAME##2dList2D

#define ARRAY_DEFINE_TYPE_SERIALIZE(TYPE, NAME) \
  ARRAY_DEFINE_TYPE(TYPE, NAME);                \
  CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(Array##NAME##2d,            \
                                     cereal::specialization::member_load_save); \
  CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ColMajArray##NAME##2d,            \
                                     cereal::specialization::member_load_save);
  /*\
  CEREAL_REGISTER_TYPE(Array##NAME##2d);        \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(Array##NAME##2d, BaseArray##NAME##2d) \
  CEREAL_REGISTER_TYPE(ColMajArray##NAME##2d);        \
  CEREAL_REGISTER_POLYMORPHIC_RELATION(ColMajArray##NAME##2d, ColMajBaseArray##NAME##2d)*/

ARRAY_DEFINE_TYPE_SERIALIZE(double, Double);
ARRAY_DEFINE_TYPE_SERIALIZE(float, Float);

ARRAY_DEFINE_TYPE(std::atomic<double>, AtomicDouble);
ARRAY_DEFINE_TYPE(std::atomic<float>, AtomicFloat);

ARRAY_DEFINE_TYPE_SERIALIZE(int32_t, Int);
ARRAY_DEFINE_TYPE_SERIALIZE(uint32_t, UInt);
ARRAY_DEFINE_TYPE_SERIALIZE(int16_t, Short);
ARRAY_DEFINE_TYPE_SERIALIZE(uint16_t, UShort);
ARRAY_DEFINE_TYPE_SERIALIZE(int64_t, Long);
ARRAY_DEFINE_TYPE_SERIALIZE(ulong, ULong);

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
 * ArrayDouble2d d(10, 10);
 * TICK_DEBUG() << "MyArray: " << d;
 */
template <typename E, typename T, typename MAJ>
tick::TemporaryLog<E>& operator<<(tick::TemporaryLog<E>& log,
                                  const Array2d<T, MAJ>& arr) {
  const auto n_cols = arr.n_cols();
  const auto n_rows = arr.n_rows();

  log << "Array2d[nrows=" << arr.n_rows() << ", ncols=" << arr.n_cols() << ", "
      << typeid(T).name();

  log << "\n[";

  const ulong max_rows = 8;
  const ulong max_cols = max_rows;

  for (uint64_t r = 0; r < std::min(max_rows, n_rows); ++r) {
    log << "[";

    for (uint64_t c = 0; c < std::min(max_cols, n_cols); ++c) {
      log << arr.data()[r * n_cols + c] << ", ";
    }

    if (n_cols > max_cols) {
      log << "... ";
    }

    log << "]\n";
  }

  if (n_rows > max_rows) {
    log << "... ]]";
  } else {
    log << "]]";
  }

  return log;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &s, const std::vector<T> &p) {
  return s << typeid(p).name() << "<" << typeid(T).name() << ">";
}

#endif  // LIB_INCLUDE_TICK_ARRAY_ARRAY2D_H_
