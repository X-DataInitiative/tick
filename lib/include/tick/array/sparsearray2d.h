
#ifndef LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY2D_H_

// License: BSD 3 clause

/** @file */

#include "basearray2d.h"

template <typename T>
class SSparseArray2d;

// Class forwarding to allow "friend class" declarations

/*! \class SparseArray2d
 * \brief Template class for basic sparse 2d-arrays of type `T`.
 * It is stored in Compressed Sparse Row matrix form :
 * The i-th row is stored in indices[row_indices[i], row_indices[i+1][`` and
 * their corresponding values are stored in ``data[row_indices[i],
 * row_indices[i+1][ So the size of the array row_indices is n_rows+1
 *
 * It manages the fact that allocation can be owned by a different object. At
 * this level, if it is not self-owned, the owner could ony be another C
 * structure. It is important to understand that if you need the array to be
 * shared with the Python interpreter then this is ONLY handled by
 * SSparseArray2d classes through their smart pointers `SSparseArray2dPtr`.
 *
 * In order to use this class, you have to understand clearly when copy of the
 * data is or is not performed. Here is a small example.
 *
 *      SparseArrayDouble2d b = c; // Copies the data
 *      SparseArrayDouble2d e = view(c) // Does not copy the data
 *      b = view(c) // No copy
 *
 */
template <typename T>
class SparseArray2d : public BaseArray2d<T> {
  // "friend class" declarations to allow private member access when
  // de/serializing
  friend class cereal::access;

 protected:
  using BaseArray2d<T>::_size;
  using BaseArray2d<T>::_size_sparse;
  using BaseArray2d<T>::_data;
  using BaseArray2d<T>::_indices;
  using BaseArray2d<T>::_row_indices;
  using BaseArray2d<T>::_n_cols;
  using BaseArray2d<T>::_n_rows;
  using BaseArray2d<T>::is_data_allocation_owned;
  using BaseArray2d<T>::is_indices_allocation_owned;
  using BaseArray2d<T>::is_row_indices_allocation_owned;

 public:
  //! @brief Constructor for zero sparsearray2d.
  //! \param n_rows Number of rows
  //! \param n_cols Number of columns
  //! \warning : No allocation is performed.
  //! Typically one should call set_data_indices_rowindices method right after
  explicit SparseArray2d(ulong n_rows = 0, ulong n_cols = 0)
      : BaseArray2d<T>(false) {
    _n_rows = n_rows;
    _n_cols = n_cols;
    _size = n_rows * n_cols;
  }

  //! @brief Constructor for constructing a 2d array of size `n_rows, n_cols`
  //! \param n_rows The number of rows
  //! \param n_cols The number of cols
  //! \param size Size of the total array
  //! \param row_indices The pointer to the rows (in the indices array)
  //! \param indices The indices of the non zero values
  //! \param data The non zero values
  //! \warning The allocations are not owned by the array
  SparseArray2d(ulong n_rows, ulong n_cols, INDICE_TYPE *row_indices,
                INDICE_TYPE *indices, T *data);

  //! @brief The copy constructor
  SparseArray2d(const SparseArray2d<T> &other) = default;

  //! @brief The move constructor
  SparseArray2d(SparseArray2d<T> &&other) = default;

  //! @brief The copy assignement operator
  SparseArray2d<T> &operator=(const SparseArray2d<T> &other) = default;

  //! @brief The move assignement operator
  SparseArray2d<T> &operator=(SparseArray2d<T> &&other) = default;

  //! @brief Destructor
  virtual ~SparseArray2d() {}

  //! @brief Returns a shared pointer to a SSparseArray2d encapsulating the
  //! sparse array \warning : The ownership of the data is given to the returned
  //! structure THUS the array becomes a view. \warning : This method cannot be
  //! called on a view
  std::shared_ptr<SSparseArray2d<T>> as_ssparsearray2d_ptr();

  template <class Archive>
  void save(Archive &ar) const {
    try {
      ar(this->_size_sparse);
      ar(this->_n_rows);
      ar(this->_n_cols);
      ar(this->_size);

      std::vector<T> v_data(this->_data, this->_data + this->_size_sparse);
      std::vector<INDICE_TYPE> v_indices(this->_indices, this->_indices + this->_size_sparse),
          v_row_indices(this->_row_indices, this->_row_indices + (this->_n_rows + 1));

      ar(v_data);
      ar(v_indices);
      ar(v_row_indices);

    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }

  template <class Archive>
  void load(Archive &ar) {
    if (this->_data || this->_indices || this->_row_indices)
      throw std::runtime_error(
          "SparseArray2d being used for deserializing may not have previous "
          "allocations");

    try {
      ar(this->_size_sparse);
      ar(this->_n_rows);
      ar(this->_n_cols);
      ar(this->_size);

      // using data structures directly on SparseArray2d<T> causes segfaults
      //  when deserializing, but using intermediary is ok, and
      //  doesn't (seem to) cause memory leaks

      T *s_data;
      std::vector<T> v_data(this->_size_sparse);

      INDICE_TYPE *s_indices, *s_row_indices;
      std::vector<INDICE_TYPE> v_indices(this->_size_sparse);
      std::vector<INDICE_TYPE> v_row_indices(this->_n_rows + 1);

      TICK_PYTHON_MALLOC(s_data, T, this->_size_sparse);
      TICK_PYTHON_MALLOC(s_indices, INDICE_TYPE, this->_size_sparse);
      TICK_PYTHON_MALLOC(s_row_indices, INDICE_TYPE, this->_n_rows + 1);

      ar(v_data);
      ar(v_indices);
      ar(v_row_indices);

      std::copy(v_data.begin(), v_data.end(), s_data);
      std::copy(v_indices.begin(), v_indices.end(), s_indices);
      std::copy(v_row_indices.begin(), v_row_indices.end(), s_row_indices);

      this->_data = s_data;
      this->_indices = s_indices;
      this->_row_indices = s_row_indices;

      this->is_data_allocation_owned = 1;
      this->is_indices_allocation_owned = 1;
      this->is_row_indices_allocation_owned = 1;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }
};

// Constructor
template <typename T>
SparseArray2d<T>::SparseArray2d(ulong n_rows, ulong n_cols,
                                INDICE_TYPE *row_indices, INDICE_TYPE *indices,
                                T *data)
    : BaseArray2d<T>(false) {
#ifdef DEBUG_ARRAY
  std::cout << "SparseArray2d Constructor : SparseArray2d("
            << "n_rows=" << n_rows << ", n_cols=" << n_cols
            << ",row_indices=" << _row_indices << ",indices=" << indices
            << ",data=" << data << ") --> " << this << std::endl;
#endif
  is_data_allocation_owned = false;
  is_indices_allocation_owned = false;
  is_row_indices_allocation_owned = false;
  _n_rows = n_rows;
  _row_indices = row_indices;
  _n_cols = n_cols;
  _indices = indices;
  _data = data;
  _size = _n_cols * _n_rows;

  _size_sparse = _row_indices[_n_rows];
}

// @brief Creates a dense Array2d from a SparseArray2d
// In terms of allocation owner, there are two cases
//     - If the BaseArray is an Array2d, then the created array is a view (so it
//     does not own allocation)
//     - If it is a SparseArray2d, then the created array owns its allocation
template <typename T>
Array2d<T> BaseArray2d<T>::as_array2d() {
  if (is_dense()) return view(*static_cast<Array2d<T> *>(this));
  Array2d<T> c(_n_rows, _n_cols);
  c.init_to_zero();

  for (ulong i = 0; i < _n_rows; i++)
    for (ulong j = _row_indices[i]; j < _row_indices[i + 1]; j++)
      c[i * _n_cols + _indices[j]] = _data[j];
  return c;
}

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

/** @defgroup array2d_sub_mod The instantiations of the Array2d template
 *  @ingroup Array_typedefs_mod
 * @{
 */

#define SPARSE_ARRAY2D_DEFINE_TYPE_BASIC(TYPE, NAME) \
  typedef SparseArray2d<TYPE> SparseArray##NAME##2d

#define SPARSE_ARRAY2D_DEFINE_TYPE(TYPE, NAME)                                  \
  SPARSE_ARRAY2D_DEFINE_TYPE_BASIC(TYPE, NAME);                                 \
  CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(SparseArray##NAME##2d,                     \
                                     cereal::specialization::member_load_save); \
  CEREAL_REGISTER_TYPE(SparseArray##NAME##2d)

SPARSE_ARRAY2D_DEFINE_TYPE(double, Double);
SPARSE_ARRAY2D_DEFINE_TYPE(float, Float);
SPARSE_ARRAY2D_DEFINE_TYPE(int32_t, Int);
SPARSE_ARRAY2D_DEFINE_TYPE(uint32_t, UInt);
SPARSE_ARRAY2D_DEFINE_TYPE(int16_t, Short);
SPARSE_ARRAY2D_DEFINE_TYPE(uint16_t, UShort);
SPARSE_ARRAY2D_DEFINE_TYPE(int64_t, Long);
SPARSE_ARRAY2D_DEFINE_TYPE(ulong, ULong);
SPARSE_ARRAY2D_DEFINE_TYPE_BASIC(std::atomic<double>, AtomicDouble);
SPARSE_ARRAY2D_DEFINE_TYPE_BASIC(std::atomic<float>, AtomicFloat);

#undef SPARSE_ARRAY2D_DEFINE_TYPE_BASIC
#undef SPARSE_ARRAY2D_DEFINE_TYPE
/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY2D_H_
