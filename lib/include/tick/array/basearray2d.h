#ifndef LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_H_

// License: BSD 3 clause

#include "abstractarray1d2d.h"

template <typename T, typename MAJ = RowMajor>
class Array2d;

/*! \class BaseArray2d
 * \brief Base template class for all the 2d-array (dense and sparse) classes of
 * type `T`.
 *
 */
template <typename T, typename MAJ = RowMajor>
class BaseArray2d : public AbstractArray1d2d<T, MAJ> {
  template <typename T1, typename MAJ1>
  friend std::ostream &operator<<(std::ostream &, const BaseArray2d<T1, MAJ1> &);

 protected:
  using AbstractArray1d2d<T, MAJ>::_size;
  using AbstractArray1d2d<T, MAJ>::_size_sparse;
  using AbstractArray1d2d<T, MAJ>::is_data_allocation_owned;
  using AbstractArray1d2d<T, MAJ>::is_indices_allocation_owned;
  using AbstractArray1d2d<T, MAJ>::_data;
  using AbstractArray1d2d<T, MAJ>::_indices;
  using K = typename AbstractArray1d2d<T, MAJ>::K;

  //! \brief Number of rows
  ulong _n_rows;

  //! \brief Number of columns
  ulong _n_cols;

  //! \brief Pointer to the rows for SparseArray2d
  INDICE_TYPE *_row_indices;

  //! \brief A flag to indicate whether the allocation of
  //! _row_indices is owned by the object itself or not
  bool is_row_indices_allocation_owned;

 public:
  using AbstractArray1d2d<T, MAJ>::is_dense;
  using AbstractArray1d2d<T, MAJ>::is_sparse;
  using AbstractArray1d2d<T, MAJ>::init_to_zero;

  //! @brief Returns the number of rows of the array
  inline ulong n_rows() const { return _n_rows; }

  //! @brief Returns the number of columns of the array
  inline ulong n_cols() const { return _n_cols; }

  //! @brief Returns the number of rows of the array
  inline INDICE_TYPE *row_indices() const { return _row_indices; }

  //! @brief Main constructor : builds an empty array
  //! \param flag_dense If true then creates a dense array otherwise it is
  //! sparse.
  explicit BaseArray2d(bool flag_dense = true)
      : AbstractArray1d2d<T, MAJ>(flag_dense) {
    _row_indices = nullptr;
    is_row_indices_allocation_owned = true;
    _n_cols = 0;
    _n_rows = 0;
  }

  //! &brief Copy constructor.
  //! \warning It copies the data creating new allocation owned by the new array
  BaseArray2d(const BaseArray2d<T, MAJ> &other);

  //! @brief Move constructor.
  //! \warning No copy of the data.
  BaseArray2d(BaseArray2d<T, MAJ> &&other) : AbstractArray1d2d<T, MAJ>(std::move(other)) {
    is_row_indices_allocation_owned = other.is_row_indices_allocation_owned;
    _row_indices = other._row_indices;
    other._row_indices = nullptr;
    _n_cols = other._n_cols;
    _n_rows = other._n_rows;
    _size = _n_cols * _n_rows;
  }

  //! @brief Assignement operator.
  //! \warning It copies the data creating new allocation owned by the new array
  BaseArray2d<T, MAJ>& operator=(const BaseArray2d<T, MAJ> &that);

  //! @brief Assignement operator from row major to col major.
  //! \warning It copies the data creating new allocation owned by the new array
  template <typename RIGHT_MAJ>
  typename std::enable_if<std::is_same<MAJ, ColMajor>::value && std::is_same<RIGHT_MAJ, RowMajor>::value, BaseArray2d<T, MAJ>>::type&
  operator=(const BaseArray2d<T, RIGHT_MAJ> &other);

  //! @brief Assignement operator from col major to row major.
  //! \warning It copies the data creating new allocation owned by the new array
  template <typename RIGHT_MAJ>
  typename std::enable_if<std::is_same<MAJ, RowMajor>::value && std::is_same<RIGHT_MAJ, ColMajor>::value, BaseArray2d<T, MAJ>>::type&
  operator=(const BaseArray2d<T, RIGHT_MAJ> &other);

  //! @brief Move assignement.
  //! \warning No copy of the data.
  BaseArray2d &operator=(BaseArray2d<T, MAJ> &&other) {
    AbstractArray1d2d<T, MAJ>::operator=(std::move(other));
    if (is_row_indices_allocation_owned && _row_indices != nullptr)
      TICK_PYTHON_FREE(_row_indices);
    _row_indices = other._row_indices;
    other._row_indices = nullptr;
    is_row_indices_allocation_owned = other.is_row_indices_allocation_owned;
    _n_cols = other._n_cols;
    _n_rows = other._n_rows;
    _size = _n_cols * _n_rows;
    return *this;
  }

  //! @brief Destructor
  virtual ~BaseArray2d() {
    if (is_row_indices_allocation_owned && _row_indices != nullptr) TICK_PYTHON_FREE(_row_indices);
  }

  void dot(const Array<T> &array, Array<T> &out) const;
  void dot_incr(const Array<T> &array, const T a, Array<T> &out) const;

 private:
  template <typename THIS_MAJ = MAJ>
  typename std::enable_if<std::is_same<THIS_MAJ, ColMajor>::value, K>::type
  _value_at_index(ulong row, ulong col) const {
    // T{0} is added to cast from T to T when T = std::atomic<T>
    return _data[col * _n_rows + row];
  }

  template <typename THIS_MAJ = MAJ>
  typename std::enable_if<std::is_same<THIS_MAJ, RowMajor>::value, K>::type
  _value_at_index(ulong row, ulong col) const {
    return _data[row * _n_cols + col];
  }

  // Method for printing (called by print() AbstractArray1d2d method
  virtual void _print_dense() const;
  virtual void _print_sparse() const;

 public:
  //! @brief Creates a dense Array2d from a BaseArray2d
  //! In terms of allocation owner, there are two cases
  //!     - If the BaseArray is an Array2d, then the created array is a view (so
  //!     it does not own allocation)
  //!     - If it is a SparseArray2d, then the created array owns its allocation
  // This method is defined in sparsearray2d.h
  Array2d<T, MAJ> as_array2d();

  //! @brief Compare two arrays by value - ignores allocation methodology !)
  bool compare(const BaseArray2d<T, MAJ> &that) {
    bool are_equal = AbstractArray1d2d<T, MAJ>::compare(that) &&
                     (this->_n_rows == that._n_rows) &&
                     (this->_n_cols == that._n_cols);
    if (are_equal && this->_row_indices && that._row_indices) {
      for (size_t i = 0; i < _n_rows + 1; i++) {
        are_equal = (this->_row_indices[i] == that._row_indices[i]);
        if (!are_equal) break;
      }
    }
    return are_equal;
  }
  bool operator==(const BaseArray2d<T, MAJ> &that) { return compare(that); }

 private:
  std::string type() const {
    return (is_dense() ? "Array2d" : "SparseArray2d");
  }
};

#include "tick/array/basearray2d/assignment.h"

//! &brief Copy constructor.
//! \warning It copies the data creating new allocation owned by the new array
template <typename T, typename MAJ>
BaseArray2d<T, MAJ>::BaseArray2d(const BaseArray2d<T, MAJ> &other)
    : AbstractArray1d2d<T, MAJ>(other) {
  _n_cols = other._n_cols;
  _n_rows = other._n_rows;
  _size = _n_cols * _n_rows;
  is_row_indices_allocation_owned = true;
  _row_indices = nullptr;
  if (other.is_sparse()) {
    TICK_PYTHON_MALLOC(_row_indices, INDICE_TYPE, _n_rows + 1);
    memcpy(_row_indices, other._row_indices,
           sizeof(INDICE_TYPE) * (_n_rows + 1));
  }
}

// @brief Prints the array
template <typename T, typename MAJ>
void BaseArray2d<T, MAJ>::_print_dense() const {
  std::cout << "Array2d[nrows=" << _n_rows << ",ncols=" << _n_cols << ","
            << std::endl;
  if (_n_rows < 6) {
    for (ulong r = 0; r < _n_rows; r++) {
      if (_n_cols < 8) {
        for (ulong c = 0; c < _n_cols; c++) {
          if (c > 0) std::cout << ",";
          std::cout << _value_at_index(r, c);
        }
      } else {
        for (ulong c = 0; c < 4; ++c)
          std::cout << _value_at_index(r, c) << ",";
        std::cout << " ... ";
        for (ulong c = _size - 4; c < _n_cols; ++c)
          std::cout << "," << _value_at_index(r, c);
      }
      std::cout << std::endl;
    }
  } else {
    for (ulong r = 0; r < 3; r++) {
      if (_n_cols < 8) {
        for (ulong c = 0; c < _n_cols; c++) {
          if (c > 0) std::cout << ",";
          std::cout << _value_at_index(r, c);
        }
      } else {
        for (ulong c = 0; c < 4; ++c)
          std::cout << _value_at_index(r, c) << ",";
        std::cout << "...";
        for (ulong c = _n_cols - 4; c < _n_cols; ++c)
          std::cout << "," << _value_at_index(r, c);
      }
      std::cout << std::endl;
    }
    std::cout << " ... " << std::endl;
    std::cout << " ... " << std::endl;
    for (ulong r = _n_rows - 3; r < _n_rows; r++) {
      if (_n_cols < 8) {
        for (ulong c = 0; c < _n_cols; c++) {
          if (c > 0) std::cout << ",";
          std::cout << _value_at_index(r, c);
        }
      } else {
        for (ulong c = 0; c < 4; ++c)
          std::cout << _value_at_index(r, c) << ",";
        std::cout << "...";
        for (ulong c = _n_cols - 4; c < _n_cols; ++c)
          std::cout << "," <<_value_at_index(r, c);
      }
      std::cout << std::endl;
    }
  }
  std::cout << "]" << std::endl;
}

template <typename T, typename MAJ>
void BaseArray2d<T, MAJ>::_print_sparse() const {
  std::cout << "_print_sparse ... not implemented" << std::endl;
}

template <typename T, typename MAJ>
inline std::ostream &operator<<(std::ostream &s, const BaseArray2d<T, MAJ> &p) {
  return s << typeid(p).name();
}

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

#include <vector>

/**
 * \defgroup Array_typedefs_mod Array related typedef
 * \brief List of all the instantiations of the BaseArray2d template and 1d and
 * 2d List of these classes
 * @{
 */

#define BASE_ARRAY_DEFINE_TYPE_SERIALIZE(TYPE, NAME)                  \
  typedef BaseArray2d<TYPE> BaseArray##NAME##2d;                      \
  typedef BaseArray2d<TYPE, ColMajor> ColMajBaseArray##NAME##2d;      \
  typedef std::vector<BaseArray##NAME##2d> BaseArray##NAME##2dList1D; \
  typedef std::vector<BaseArray##NAME##2dList1D> BaseArray##NAME##2dList2D

BASE_ARRAY_DEFINE_TYPE_SERIALIZE(double, Double);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(float, Float);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(int32_t, Int);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(uint32_t, UInt);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(int16_t, Short);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(uint16_t, UShort);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(int64_t, Long);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(ulong, ULong);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(std::atomic<double>, AtomicDouble);
BASE_ARRAY_DEFINE_TYPE_SERIALIZE(std::atomic<float>, AtomicFloat);

#undef BASE_ARRAY2D_DEFINE_TYPE

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_H_
