#ifndef LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY_H_
#define LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY_H_

// License: BSD 3 clause

/** @file */

#include "tick/base/defs.h"
#include "alloc.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  1d SparseArray
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class SSparseArray;

/*! \class SparseArray
 * \brief Template class for basic sparse arrays of type `T`.
 *
 * It manages the fact that allocation can be owned by a different object. At this level, if
 * it is not self-owned, the owner could ony be another C structure.
 * It is important to understand that if you need the array to be shared with the Python interpreter
 * then this is ONLY handled by SSparseArray classes through their smart pointers `SSparseArrayPtr`.
 *
 * In order to use this class, you have to understand clearly when copy of the data is or is not
 * performed. Here is a small example.
 *
 *      SparseArrayDouble b = c; // Copies the data
 *      SparseArrayDouble e = view(c) // Does not copy the data
 *      b = view(c) // No copy
 *
 */
template<typename T>
class SparseArray : public BaseArray<T> {
 protected:
    using BaseArray<T>::_size;
    using BaseArray<T>::_size_sparse;
    using BaseArray<T>::_data;
    using BaseArray<T>::_indices;
    using BaseArray<T>::is_data_allocation_owned;
    using BaseArray<T>::is_indices_allocation_owned;

 public :
    //! @brief Constructor for an empty sparse array.
    explicit SparseArray(ulong size = 0) : BaseArray<T>(false) {
        _size = size;
    }

    //! @brief Constructor for a sparse array.
    //
    //! \param size Size of the total array
    //! \param size_sparse Number of non zero values
    //! \param indices The indices of the non zero values
    //! \param data The non zero values
    //! \warning The allocations are not owned by the array
    SparseArray(ulong size, ulong size_sparse, INDICE_TYPE *indices, T *data);

    //! @brief The copy constructor
    SparseArray(const SparseArray<T> &other) = default;

    //! @brief The move constructor
    SparseArray(SparseArray<T> &&other) = default;

    //! @brief The copy assignement operator
    SparseArray<T> &operator=(const SparseArray<T> &other) = default;

    //! @brief The move assignement operator
    SparseArray<T> &operator=(SparseArray<T> &&other) = default;

    //! @brief Destructor
    virtual ~SparseArray() {}

    //! @brief Returns a shared pointer to a SSparseArray encapsulating the sparse array
    //! \warning : The ownership of the data is given to the returned structure
    //! THUS the array becomes a view.
    //! \warning : This method cannot be called on a view
    std::shared_ptr<SSparseArray<T>> as_ssparsearray_ptr();
};

// Constructor
template<typename T>
SparseArray<T>::SparseArray(ulong size, ulong size_sparse, INDICE_TYPE *indices, T *data) :
    BaseArray<T>(false) {
#ifdef DEBUG_ARRAY
    std::cout << "SparseArray Constructor : SparseArray(size=" << _size
        << ", size_sparse=" << size_sparse
        << ", indices=" << indices
        << ", data=" << data
        << ") --> "
        << this << std::endl;
#endif
    is_data_allocation_owned = false;
    is_indices_allocation_owned = false;
    _size = size;
    _size_sparse = size_sparse;
    _indices = indices;
    _data = data;
}

// @brief Creates a dense Array from an AbstractArray
// In terms of allocation owner, there are two cases
//     - If the BaseArray is an Array, then the created array is a view (so it does not own allocation)
//     - If it is a SparseArray, then the created array owns its allocation
// This method is defined in sparsearray.h
template<typename T>
Array<T> BaseArray<T>::as_array() {
    if (is_dense()) {
        return view(*static_cast<Array<T> *>(this));
    } else {
        Array<T> c(_size);
        c.init_to_zero();
        for (ulong i = 0; i < _size_sparse; i++) {
            c[_indices[i]] = _data[i];
        }
        return c;
    }
}

/////////////////////////////////////////////////////////////////
//
//  The various instances of this template
//
/////////////////////////////////////////////////////////////////

#include <vector>

/**
 * \defgroup Array_typedefs_mod Array related typedef
 * \brief List of all the instantiations of the Array template and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup array_sub_mod The instantiations of the Array template
 *  @ingroup Array_typedefs_mod
 * @{
 */
typedef SparseArray<double> SparseArrayDouble;
typedef SparseArray<float> SparseArrayFloat;
typedef SparseArray<std::int32_t> SparseArrayInt;
typedef SparseArray<std::uint32_t> SparseArrayUInt;
typedef SparseArray<std::int16_t> SparseArrayShort;
typedef SparseArray<std::uint16_t> SparseArrayUShort;
typedef SparseArray<std::int64_t> SparseArrayLong;
typedef SparseArray<ulong> SparseArrayULong;

/**
 * @}
 */

/** @defgroup arraylist1d_sub_mod The classes for dealing with 1d-list of Arrays
 *  @ingroup Array_typedefs_mod
 * @{
 */

typedef std::vector<SparseArray<float> > SparseArrayFloatList1D;
typedef std::vector<SparseArray<double> > SparseArrayDoubleList1D;
typedef std::vector<SparseArray<std::int32_t> > SparseArrayIntList1D;
typedef std::vector<SparseArray<std::uint32_t> > SparseArrayUIntList1D;
typedef std::vector<SparseArray<std::int16_t> > SparseArrayShortList1D;
typedef std::vector<SparseArray<std::uint16_t> > SparseArrayUShortList1D;
typedef std::vector<SparseArray<std::int64_t> > SparseArrayLongList1D;
typedef std::vector<SparseArray<ulong> > SparseArrayULongList1D;

/**
 * @}
 */

/** @defgroup arraylist2d_sub_mod The classes for dealing with 2d-list of Arrays
 *  @ingroup Array_typedefs_mod
 * @{
 */
typedef std::vector<SparseArrayFloatList1D> SparseArrayFloatList2D;
typedef std::vector<SparseArrayIntList1D> SparseArrayIntList2D;
typedef std::vector<SparseArrayUIntList1D> SparseArrayUIntList2D;
typedef std::vector<SparseArrayShortList1D> SparseArrayShortList2D;
typedef std::vector<SparseArrayUShortList1D> SparseArrayUShortList2D;
typedef std::vector<SparseArrayLongList1D> SparseArrayLongList2D;
typedef std::vector<SparseArrayULongList1D> SparseArrayULongList2D;
typedef std::vector<SparseArrayDoubleList1D> SparseArrayDoubleList2D;

/**
 * @}
 */

template <typename E, typename T>
tick::TemporaryLog<E>& operator<<(tick::TemporaryLog<E>& log, const SparseArray<T>& arr) {
    const auto size = arr.size();
    const auto size_sparse = arr.size_sparse();

    log << "SparseArray[size=" << size << ", size_sparse=" << size_sparse << ", " << typeid(T).name();

    if (size_sparse <= 20) {
        for (ulong i = 0; i < size_sparse; ++i) log <<  ", " << arr.indices()[i] << "/" << arr.data()[i];
    } else {
        for (ulong i = 0; i < 10; ++i) log <<  ", " << arr.indices()[i] << "/" << arr.data()[i];

        log << ", ...";

        for (ulong i = size - 10; i < size; ++i) log <<  ", " << arr.indices()[i] << "/" << arr.data()[i];
    }

    log << "]";

    return log;
}

#endif  // LIB_INCLUDE_TICK_ARRAY_SPARSEARRAY_H_
