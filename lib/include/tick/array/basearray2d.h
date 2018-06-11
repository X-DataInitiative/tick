#ifndef LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_H_

// License: BSD 3 clause

#include "abstractarray1d2d.h"

template <typename T>
class Array2d;

/*! \class BaseArray2d
 * \brief Base template class for all the 2d-array (dense and sparse) classes of
 * type `T`.
 *
 */
template <typename T>
class BaseArray2d : public AbstractArray1d2d<T> {
  template <class T1>
  friend std::ostream &operator<<(std::ostream &, const BaseArray2d<T1> &);

 protected:
  using AbstractArray1d2d<T>::_size;
  using AbstractArray1d2d<T>::_size_sparse;
  using AbstractArray1d2d<T>::is_data_allocation_owned;
  using AbstractArray1d2d<T>::is_indices_allocation_owned;
  using AbstractArray1d2d<T>::_data;
  using AbstractArray1d2d<T>::_indices;

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
  using AbstractArray1d2d<T>::is_dense;
  using AbstractArray1d2d<T>::is_sparse;
  using AbstractArray1d2d<T>::init_to_zero;

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
      : AbstractArray1d2d<T>(flag_dense) {
    _row_indices = nullptr;
    is_row_indices_allocation_owned = true;
    _n_cols = 0;
    _n_rows = 0;
  }

  //! &brief Copy constructor.
  //! \warning It copies the data creating new allocation owned by the new array
  BaseArray2d(const BaseArray2d<T> &other);

  //! @brief Move constructor.
  //! \warning No copy of the data.
  BaseArray2d(BaseArray2d<T> &&other) : AbstractArray1d2d<T>(std::move(other)) {
    is_row_indices_allocation_owned = other.is_row_indices_allocation_owned;
    _row_indices = other._row_indices;
    other._row_indices = nullptr;
    _n_cols = other._n_cols;
    _n_rows = other._n_rows;
    _size = _n_cols * _n_rows;
  }

  //! @brief Assignement operator.
  //! \warning It copies the data creating new allocation owned by the new array
  BaseArray2d &operator=(const BaseArray2d<T> &other) {
    if (this != &other) {
      AbstractArray1d2d<T>::operator=(other);
      if (is_row_indices_allocation_owned && _row_indices != nullptr)
        TICK_PYTHON_FREE(_row_indices);
      _row_indices = nullptr;
      is_row_indices_allocation_owned = true;
      _n_cols = other._n_cols;
      _n_rows = other._n_rows;
      _size = _n_cols * _n_rows;
      if (other.is_sparse()) {
        TICK_PYTHON_MALLOC(_row_indices, INDICE_TYPE, _n_rows + 1);
        memcpy(_row_indices, other._row_indices,
               sizeof(INDICE_TYPE) * (_n_rows + 1));
      }
    }
    return (*this);
  }

  //! @brief Move assignement.
  //! \warning No copy of the data.
  BaseArray2d &operator=(BaseArray2d<T> &&other) {
    AbstractArray1d2d<T>::operator=(std::move(other));
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
    if (is_row_indices_allocation_owned && _row_indices != nullptr)
      TICK_PYTHON_FREE(_row_indices);
  }

 private:
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
  Array2d<T> as_array2d();

  //! @brief Compare two arrays by value - ignores allocation methodology !)
  bool compare(const BaseArray2d<T> &that) {
    bool are_equal = AbstractArray1d2d<T>::compare(that) &&
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
  bool operator==(const BaseArray2d<T> &that) { return compare(that); }

 private:
  std::string type() const {
    return (is_dense() ? "Array2d" : "SparseArray2d");
  }
};

//! &brief Copy constructor.
//! \warning It copies the data creating new allocation owned by the new array
template <typename T>
BaseArray2d<T>::BaseArray2d(const BaseArray2d<T> &other)
    : AbstractArray1d2d<T>(other) {
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
template <typename T>
void BaseArray2d<T>::_print_dense() const {
  std::cout << "Array2d[nrows=" << _n_rows << ",ncols=" << _n_cols << ","
            << std::endl;
  if (_n_rows < 6) {
    for (ulong r = 0; r < _n_rows; r++) {
      if (_n_cols < 8) {
        for (ulong c = 0; c < _n_cols; c++) {
          if (c > 0) std::cout << ",";
          std::cout << _data[r * _n_cols + c];
        }
      } else {
        for (ulong c = 0; c < 4; ++c)
          std::cout << _data[r * _n_cols + c] << ",";
        std::cout << " ... ";
        for (ulong c = _size - 4; c < _n_cols; ++c)
          std::cout << "," << _data[r * _n_cols + c];
      }
      std::cout << std::endl;
    }
  } else {
    for (ulong r = 0; r < 3; r++) {
      if (_n_cols < 8) {
        for (ulong c = 0; c < _n_cols; c++) {
          if (c > 0) std::cout << ",";
          std::cout << _data[r * _n_cols + c];
        }
      } else {
        for (ulong c = 0; c < 4; ++c)
          std::cout << _data[r * _n_cols + c] << ",";
        std::cout << "...";
        for (ulong c = _n_cols - 4; c < _n_cols; ++c)
          std::cout << "," << _data[r * _n_cols + c];
      }
      std::cout << std::endl;
    }
    std::cout << " ... " << std::endl;
    std::cout << " ... " << std::endl;
    for (ulong r = _n_rows - 3; r < _n_rows; r++) {
      if (_n_cols < 8) {
        for (ulong c = 0; c < _n_cols; c++) {
          if (c > 0) std::cout << ",";
          std::cout << _data[r * _n_cols + c];
        }
      } else {
        for (ulong c = 0; c < 4; ++c)
          std::cout << _data[r * _n_cols + c] << ",";
        std::cout << "...";
        for (ulong c = _n_cols - 4; c < _n_cols; ++c)
          std::cout << "," << _data[r * _n_cols + c];
      }
      std::cout << std::endl;
    }
  }
  std::cout << "]" << std::endl;
}

template <typename T>
void BaseArray2d<T>::_print_sparse() const {
  std::cout << "_print_sparse ... not implemented" << std::endl;
}

template <typename T>
inline std::ostream &operator<<(std::ostream &s, const BaseArray2d<T> &p) {
  return s << typeid(p).name() << "<" << typeid(T).name() << ">";
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

/**
 * @}
 */

/** @defgroup BaseArray2d_sub_mod The instantiations of the BaseArray template
 *  @ingroup Array_typedefs_mod
 * @{
 */

#define BASE_ARRAY2D_DEFINE_TYPE(TYPE, NAME)\
  typedef BaseArray2d<TYPE> BaseArray##NAME##2d; \
  typedef std::vector<BaseArray##NAME##2d> BaseArray##NAME##2dList1D; \
  typedef std::vector<BaseArray##NAME##2dList1D> BaseArray##NAME##2dList2D

BASE_ARRAY2D_DEFINE_TYPE(double, Double);
BASE_ARRAY2D_DEFINE_TYPE(float, Float);
BASE_ARRAY2D_DEFINE_TYPE(int32_t, Int);
BASE_ARRAY2D_DEFINE_TYPE(uint32_t, UInt);
BASE_ARRAY2D_DEFINE_TYPE(int16_t, Short);
BASE_ARRAY2D_DEFINE_TYPE(uint16_t, UShort);
BASE_ARRAY2D_DEFINE_TYPE(int64_t, Long);
BASE_ARRAY2D_DEFINE_TYPE(ulong, ULong);
BASE_ARRAY2D_DEFINE_TYPE(std::atomic<double>, AtomicDouble);
BASE_ARRAY2D_DEFINE_TYPE(std::atomic<float>, AtomicFloat);

#undef BASE_ARRAY2D_DEFINE_TYPE

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_BASEARRAY2D_H_
