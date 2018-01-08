
#ifndef LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY2D_H_
#define LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY2D_H_

// License: BSD 3 clause

/** @file */

#include "basearray2d.h"
#include "array2d.h"

template<typename T>
class SSparseArray2d;

// Class forwarding to allow "friend class" declarations
namespace cereal {
  template <typename T, class Archive> void save(Archive& ar, const SparseArray2d<T>& s);
  template <typename T, class Archive> void load(Archive& ar, SparseArray2d<T>& s);
}

/*! \class SparseArray2d
 * \brief Template class for basic sparse 2d-arrays of type `T`.
 * It is stored in Compressed Sparse Row matrix form :
 * The i-th row is stored in indices[row_indices[i], row_indices[i+1][`` and their
 * corresponding values are stored in ``data[row_indices[i], row_indices[i+1][
 * So the size of the array row_indices is n_rows+1
 *
 * It manages the fact that allocation can be owned by a different object. At this level, if
 * it is not self-owned, the owner could ony be another C structure.
 * It is important to understand that if you need the array to be shared with the Python interpreter
 * then this is ONLY handled by SSparseArray2d classes through their smart pointers `SSparseArray2dPtr`.
 *
 * In order to use this class, you have to understand clearly when copy of the data is or is not
 * performed. Here is a small example.
 *
 *      SparseArrayDouble2d b = c; // Copies the data
 *      SparseArrayDouble2d e = view(c) // Does not copy the data
 *      b = view(c) // No copy
 *
 */
template<typename T>
class SparseArray2d : public BaseArray2d<T> {
  // "friend class" declarations to allow private member access when de/serializing
  template <typename T1, class Archive> friend void cereal::save(Archive& ar, const SparseArray2d<T1>& s);
  template <typename T1, class Archive> friend void cereal::load(Archive& ar, SparseArray2d<T1>& s);

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
    explicit SparseArray2d(ulong n_rows = 0, ulong n_cols = 0) : BaseArray2d<T>(false) {
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
    SparseArray2d(ulong n_rows, ulong n_cols,
                  INDICE_TYPE *row_indices, INDICE_TYPE *indices, T *data);

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

    //! @brief Returns a shared pointer to a SSparseArray2d encapsulating the sparse array
    //! \warning : The ownership of the data is given to the returned structure
    //! THUS the array becomes a view.
    //! \warning : This method cannot be called on a view
    std::shared_ptr<SSparseArray2d<T>> as_ssparsearray2d_ptr();
};

// Constructor
template<typename T>
SparseArray2d<T>::SparseArray2d(ulong n_rows, ulong n_cols,
                                INDICE_TYPE *row_indices, INDICE_TYPE *indices, T *data) :
    BaseArray2d<T>(false) {
#ifdef DEBUG_ARRAY
    std::cout << "SparseArray2d Constructor : SparseArray2d("
    << "n_rows=" << n_rows
    << ", n_cols=" << n_cols
    << ",row_indices=" << _row_indices
    << ",indices=" << indices
    << ",data=" << data
    << ") --> "
    << this << std::endl;
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
//     - If the BaseArray is an Array2d, then the created array is a view (so it does not own allocation)
//     - If it is a SparseArray2d, then the created array owns its allocation
template<typename T>
Array2d<T> BaseArray2d<T>::as_array2d() {
    if (is_dense()) return view(*static_cast<Array2d<T> *>(this));
    Array2d<T> c(_n_rows, _n_cols);
    c.init_to_zero();

    for (ulong i = 0; i < _n_rows; i++)
        for (ulong j = _row_indices[i]; j < _row_indices[i + 1]; j++)
            c[i * _n_cols + _indices[j]] = _data[j];
    return c;
}

// Using save/load methods directory on SparseArray2d<T> does not compile simply
namespace cereal {

  template <typename T, class Archive>
  void save(Archive& ar, const SparseArray2d<T>& s) {
    try {
      ar(s._size_sparse);
      ar(s._n_rows);
      ar(s._n_cols);
      ar(s._size);
      ar( cereal::binary_data(s._data, sizeof(T) * s._size_sparse));
      ar( cereal::binary_data(s._indices, sizeof(INDICE_TYPE) * s._size_sparse));
      ar( cereal::binary_data(s._row_indices, sizeof(INDICE_TYPE) * (s._n_rows + 1)));
    }catch(const std::exception& e) {
      std::cerr << e.what() << std::endl;
    }
  }

  template<typename T, class Archive>
  void load(Archive &ar, SparseArray2d<T> &s) {
    if (s._data || s._indices || s._row_indices)
      throw std::runtime_error(
        "SparseArray2d being used for deserializing may not have previous allocations");

    try {
      ar(s._size_sparse);
      ar(s._n_rows);
      ar(s._n_cols);
      ar(s._size);

      // using data structures directly on SparseArray2d<T> causes segfaults
      //  when deserializing, but using intermediary is ok, and
      //  doesn't (seem to) cause memory leaks

      T *s_data;
      TICK_PYTHON_MALLOC(s_data, T, s._size_sparse);
      INDICE_TYPE *s_indices;
      TICK_PYTHON_MALLOC(s_indices, INDICE_TYPE, s._size_sparse);
      INDICE_TYPE *s_row_indices;
      TICK_PYTHON_MALLOC(s_row_indices, INDICE_TYPE, s._n_rows + 1);

      ar(cereal::binary_data(s_data, sizeof(T) * s._size_sparse));
      ar(cereal::binary_data(s_indices, sizeof(INDICE_TYPE) * s._size_sparse));
      ar(cereal::binary_data(s_row_indices, sizeof(INDICE_TYPE) * (s._n_rows + 1)));

      s._data = s_data;
      s._indices = s_indices;
      s._row_indices = s_row_indices;

      s.is_data_allocation_owned = 1;
      s.is_indices_allocation_owned = 1;
      s.is_row_indices_allocation_owned = 1;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    }
  }
}  // namespace cereal

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

/** @defgroup array2d_sub_mod The instantiations of the Array2d template
 *  @ingroup Array_typedefs_mod
 * @{
 */

typedef SparseArray2d<double> SparseArrayDouble2d;
typedef SparseArray2d<float> SparseArrayFloat2d;
typedef SparseArray2d<int> SparseArrayInt2d;
typedef SparseArray2d<std::uint32_t> SparseArrayUInt2d;
typedef SparseArray2d<std::int16_t> SparseArrayShort2d;
typedef SparseArray2d<std::uint16_t> SparseArrayUShort2d;
typedef SparseArray2d<std::int64_t> SparseArrayLong2d;
typedef SparseArray2d<ulong> SparseArrayULong2d;

/**
 * @}
 */

#endif  // LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY2D_H_
