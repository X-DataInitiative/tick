//
//  ssparsearray.h
//  tests
//
//  Created by bacry on 06/01/2016.
//  Copyright (c) 2016 bacry. All rights reserved.
//

#ifndef TICK_BASE_ARRAY_SRC_SSPARSEARRAY_H_
#define TICK_BASE_ARRAY_SRC_SSPARSEARRAY_H_

/** @file */

#include <memory>
#include "sparsearray.h"

///////////////////////////////////////////////////////////////////////////////////
//
//  SSparseArray
//
///////////////////////////////////////////////////////////////////////////////////

/*! \class SSparseArray
 * \brief Template class for 1d sparse arrays of type `T` fully shareable
 * with Python.
 * This is the array class that introduces the ability to share its allocation with Python.
 * \warning You should always use the associated smart pointer class
 * `SSparseArrayPtr` and never access this class directly. Thus the only constructor
 * to be used is the `SSparseArray<T>::new_ptr()` constructor. In that way you can also share the
 * allocations within C++ (using a reference counter).
 */

template<typename T>
class SSparseArray : public SparseArray<T> {
 protected:
    using SparseArray<T>::_size;
    using SparseArray<T>::is_data_allocation_owned;
    using SparseArray<T>::_data;
    using SparseArray<T>::_size_sparse;
    using SparseArray<T>::is_indices_allocation_owned;
    using SparseArray<T>::_indices;

#ifdef PYTHON_LINK
    //! @brief The (eventual) Python owner of the array _data;
    //! If ==nullptr then it is self-owned
    void *_data_owner;

    //! @brief The (eventual) Python owner of the array _indices;
    //! If ==nullptr then it is self-owned
    void *_indices_owner;
#endif

 public:
#ifdef PYTHON_LINK
    void give_data_indices_owners(void *data_owner, void* indices_owner);

    //! @brief returns the python object which is owner _data allocation
    inline void *data_owner() {return _data_owner;}

    //! @brief returns the python object which is owner _indices allocation
    inline void *indices_owner() {return _indices_owner;}

    //! @brief Sets the data/indices allocation owner with either a numpy array or nullptr
    //! \param data A pointer to the data array
    //! \param indices A pointer to the indices array
    //! \param size The size of the full array
    //! \param size_sparse The size of non zero values
    //! \param data_owner A pointer to the numpy array (if nullptr, then the data allocation is owned by the object itself) that owns _data allocation
    //! \param indices_owner A pointer to the numpy array (if nullptr, then the data allocation is owned by the object itself) that owns _indices allocation
    virtual void set_data_indices(T *data, INDICE_TYPE *indices, ulong size, ulong size_sparse, void *owner_data = nullptr, void *owner_indices = nullptr);
#else
    //! @brief Sets the data. After this call, allocation is owned by the SArray.
    //! \param data A pointer to the data array
    //! \param indices A pointer to the indices array
    //! \param size The size of the full array
    //! \param size_sparse The size of non zero values
    virtual void set_data_indices(T *data, INDICE_TYPE *indices, ulong size, ulong size_sparse);
#endif

#ifdef DEBUG_SHAREDARRAY
    static ulong n_allocs;
#endif

 public:
    // Constructors :
    //     One should only use the constructor
    //     SSparseArray<T>::new_ptr(size, size_sparse) which returns a shared pointer
    //     SSparseArray<T>::new_ptr(array) which returns a shared pointer
    //

    //! \cond
    // Constructor of an empty sparse array : not to be used directly
    // No allocation is performed whatever size is
    // Typically after this call one should call the set_data method.s
    explicit SSparseArray(ulong size = 0);
    //! \endcond

    //! @brief The only constructor to be used
    //! \return A shared pointer to the corresponding shared sparsearray of the given size
    //! and size_sparse. So if size_sparse != 0 then allocation is performed.
    //! \warning : Right after this call, you are supposed to fill up the indices array with
    //! valid values
    static std::shared_ptr<SSparseArray<T>> new_ptr(ulong size, ulong size_sparse);

    //! @brief The only constructor to be used from an SparseArray<T>
    //! \param a : The array the data/indices are copied from
    //! \return A shared pointer to the corresponding shared sparse array
    //! \warning The data/indices arrays are copied
    static std::shared_ptr<SSparseArray<T>> new_ptr(SparseArray<T> &a);

 protected:
    /**
     * @brief Clears the array without desallocating the data pointer.
     * Returns this pointer if it should be desallocated otherwise returns
     * nullptr
     */
    virtual void _clear(bool &flag_desallocate_data, bool &flag_desallocate_indices);

 public:
    //! @brief clears the corresponding allocation (size becomes 0)
    virtual void clear();

    //! @brief Destructor
    virtual ~SSparseArray();

    // The following constructors should not be used as we only use SSparseArray with the
    // associated shared_ptr
    SSparseArray(const SSparseArray<T> &other) = delete;
    SSparseArray(const SSparseArray<T> &&other) = delete;
};

#ifdef PYTHON_LINK
template <typename T>
void SSparseArray<T>::set_data_indices(T *data, INDICE_TYPE *indices, ulong size, ulong size_sparse, void *data_owner, void * indices_owner) {
    clear();
    _data = data;
    _indices = indices;
    _size = size;
    _size_sparse = size_sparse;
    give_data_indices_owners(data_owner, indices_owner);
}
#else
template<typename T>
void SSparseArray<T>::set_data_indices(T *data, INDICE_TYPE *indices, ulong size, ulong size_sparse) {
    clear();
    _data = data;
    _indices = indices;
    _size = size;
    _size_sparse = size_sparse;
    is_data_allocation_owned = true;
    is_indices_allocation_owned = true;
}
#endif

#ifdef PYTHON_LINK
template <typename T>
void SSparseArray<T>::give_data_indices_owners(void *data_owner, void *indices_owner) {
#ifdef DEBUG_SHAREDARRAY
    if (_data_owner == nullptr)
        std::cout << "SSparseArray : SetOwner data_owner=" << data_owner<< " on " << this << std::endl;
    else
        std::cout << "SSparseArray : ChangeDataOwner data_owner=" << _data_owner <<" -> " << data_owner << " on " << this << std::endl;

    if (_indices_owner == nullptr)
        std::cout << "SSparseArray : SetOwner indices_owner=" << data_owner << " on " << this << std::endl;
    else
        std::cout << "SSparseArray : ChangeIndicesOwner indices_owner=" << _indices_owner <<" -> " << indices_owner << " on " << this << std::endl;
#endif
    _data_owner = data_owner;
    if (_data_owner) {
        PYINCREF(_data_owner);
        is_data_allocation_owned = false;
    } else {
        is_data_allocation_owned = true;
    }

    _indices_owner = indices_owner;
    if (_indices_owner) {
        PYINCREF(_indices_owner);
        is_indices_allocation_owned = false;
    } else {
        is_indices_allocation_owned = true;
    }
}
#endif

#ifdef DEBUG_SHAREDARRAY
template <typename T>
ulong SSparseArray<T>::n_allocs = 0;
#endif

// In order to create a Shared SparseArray
template<typename T>
SSparseArray<T>::SSparseArray(ulong size) : SparseArray<T>(size) {
#ifdef PYTHON_LINK
    _data_owner = nullptr;
    _indices_owner = nullptr;
#endif
#ifdef DEBUG_SHAREDARRAY
    n_allocs++;
    std::cout << "SSparseArray Constructor (->#" << n_allocs << ") : SArray(size=" << _size << ") --> "
    << this << std::endl;
#endif
}

// @brief The only constructor to be used
template<typename T>
std::shared_ptr<SSparseArray<T>> SSparseArray<T>::new_ptr(ulong size, ulong size_sparse) {
    std::shared_ptr<SSparseArray<T>> aptr = std::make_shared<SSparseArray<T>>(size);
    if (size == 0 || size_sparse == 0) {
        return aptr;
    }
    T *data;
    TICK_PYTHON_MALLOC(data, T, size_sparse);
    INDICE_TYPE *indices;
    TICK_PYTHON_MALLOC(indices, INDICE_TYPE, size_sparse);
    aptr->set_data_indices(data, indices, size, size_sparse);
    return aptr;
}

// Constructor from a SparseArray (copy is made)
template<typename T>
std::shared_ptr<SSparseArray<T>> SSparseArray<T>::new_ptr(SparseArray<T> &a) {
    std::shared_ptr<SSparseArray<T>> aptr = SSparseArray<T>::new_ptr(a.size(), a.size_sparse());
    if (a.size_sparse() != 0) {
        memcpy(aptr->data(), a.data(), sizeof(T) * a.size_sparse());
        memcpy(aptr->indices(), a.indices(), sizeof(INDICE_TYPE) * a.size_sparse());
    }
    return aptr;
}

// Clears the array without desallocation
// returns some flags to understand what should be desallocated
template<typename T>
void SSparseArray<T>::_clear(bool &flag_desallocate_data, bool &flag_desallocate_indices) {
    flag_desallocate_data = flag_desallocate_indices = false;

    // Case not empty
    if (_data) {
#ifdef PYTHON_LINK
        // We have to deal with Python owner if any
        if (_data_owner != nullptr) {
            PYDECREF(_data_owner);
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray Clear : " << this
                << " decided not to free data=" << _data << " since owned by owner=" << _data_owner << std::endl;
#endif
            _data_owner = nullptr;
        } else {
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray Clear : " << this << " decided to free data="
                << _data << std::endl;
#endif
            flag_desallocate_data = true;
        }
        if (_indices_owner != nullptr) {
            PYDECREF(_indices_owner);
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray Clear : " << this
            << " decided not to free indices=" << _indices << " since owned by owner=" << _indices_owner << std::endl;
#endif
            _indices_owner = nullptr;
        } else {
#ifdef DEBUG_SHAREDARRAY
            std::cout << "SSparseArray Clear : " << this << " decided to free indices="
            << _indices << std::endl;
            flag_desallocate_indices = true;
#endif
        }
#else
#ifdef DEBUG_SHAREDARRAY
        std::cout << "SSparseArray Clear : " << this << " decided to free data=" << _data << std::endl;
        std::cout << "SSparseArray Clear : " << this << " decided to free indices=" << _indices << std::endl;
#endif
        flag_desallocate_data = true;
        flag_desallocate_indices = true;
#endif
    }
    _size = 0;
    _size_sparse = 0;

    is_indices_allocation_owned = true;
    is_data_allocation_owned = true;
}

// clear the array (including data if necessary)
template<typename T>
void SSparseArray<T>::clear() {
    bool flag_desallocate_data;
    bool flag_desallocate_indices;
    _clear(flag_desallocate_data, flag_desallocate_indices);
    if (flag_desallocate_data) TICK_PYTHON_FREE(_data);
    if (flag_desallocate_indices) TICK_PYTHON_FREE(_indices);
    _data = nullptr;
    _indices = nullptr;
}

// Destructor
template<typename T>
SSparseArray<T>::~SSparseArray() {
#ifdef DEBUG_SHAREDARRAY
    n_allocs--;
    std::cout << "SSparseArray Destructor (->#" << n_allocs << ") : ~SSparseArray on " <<  this << std::endl;
#endif
    clear();
}

// @brief Returns a shared pointer to a SSparseArray encapsulating the sparse array
// \warning : The ownership of the data is given to the returned structure
// THUS the array becomes a view.
// \warning : This method cannot be called on a view
template<typename T>
std::shared_ptr<SSparseArray<T>> SparseArray<T>::as_ssparsearray_ptr() {
    if (!is_data_allocation_owned || !is_indices_allocation_owned)
        TICK_ERROR("This method cannot be called on an object that does not own its allocations");

    std::shared_ptr<SSparseArray<T>> arrayptr = SSparseArray<T>::new_ptr(0, 0);
    arrayptr->set_data_indices(_data, _indices, _size, _size_sparse);
    is_data_allocation_owned = false;
    is_indices_allocation_owned = false;
    return arrayptr;
}

// Instanciations

/**
 * \defgroup SArray_typedefs_mod SArray related typedef
 * \brief List of all the instantiations of the SArray template and associated
 *  shared pointers and 1d and 2d List of these classes
 * @{
 */

/**
 * @}
 */

/** @defgroup sarray_sub_mod The instantiations of the SArray template
 *  @ingroup SArray_typedefs_mod
 * @{
 */

typedef SSparseArray<double> SSparseArrayDouble;
typedef SSparseArray<float> SSparseArrayFloat;
typedef SSparseArray<std::int32_t> SSparseArrayInt;
typedef SSparseArray<std::uint32_t> SSparseArrayUInt;
typedef SSparseArray<std::int16_t> SSparseArrayShort;
typedef SSparseArray<std::uint16_t> SSparseArrayUShort;
typedef SSparseArray<std::int64_t> SSparseArrayLong;
typedef SSparseArray<ulong> SSparseArrayULong;

/**
 * @}
 */

/** @defgroup sarrayptr_sub_mod The shared pointer array classes
 *  @ingroup SArray_typedefs_mod
 * @{
 */

typedef std::shared_ptr<SSparseArrayFloat> SSparseArrayFloatPtr;
typedef std::shared_ptr<SSparseArrayInt> SSparseArrayIntPtr;
typedef std::shared_ptr<SSparseArrayUInt> SSparseArrayUIntPtr;
typedef std::shared_ptr<SSparseArrayShort> SSparseArrayShortPtr;
typedef std::shared_ptr<SSparseArrayUShort> SSparseArrayUShortPtr;
typedef std::shared_ptr<SSparseArrayLong> SSparseArrayLongPtr;
typedef std::shared_ptr<SSparseArrayULong> SSparseArrayULongPtr;
typedef std::shared_ptr<SSparseArrayDouble> SSparseArrayDoublePtr;

/**
 * @}
 */

/** @defgroup sarrayptrlist1d_sub_mod The classes for dealing with 1d-list of shared pointer arrays
 *  @ingroup SArray_typedefs_mod
 * @{
 */

// @brief The basic SSparseArrayList1D classes
typedef std::vector<SSparseArrayFloatPtr> SSparseArrayFloatPtrList1D;
typedef std::vector<SSparseArrayIntPtr> SSparseArrayIntPtrList1D;
typedef std::vector<SSparseArrayUIntPtr> SSparseArrayUIntPtrList1D;
typedef std::vector<SSparseArrayShortPtr> SSparseArrayShortPtrList1D;
typedef std::vector<SSparseArrayUShortPtr> SSparseArrayUShortPtrList1D;
typedef std::vector<SSparseArrayLongPtr> SSparseArrayLongPtrList1D;
typedef std::vector<SSparseArrayULongPtr> SSparseArrayULongPtrList1D;
typedef std::vector<SSparseArrayDoublePtr> SSparseArrayDoublePtrList1D;

/**
 * @}
 */

/** @defgroup sarrayptrlist2d_sub_mod The classes for dealing with 2d-list of shared pointer arrays
 *  @ingroup SSparseArray_typedefs_mod
 * @{
 */

// @brief The basic SSparseArrayList2D classes
typedef std::vector<SSparseArrayFloatPtrList1D> SSparseArrayFloatPtrList2D;
typedef std::vector<SSparseArrayIntPtrList1D> SSparseArrayIntPtrList2D;
typedef std::vector<SSparseArrayUIntPtrList1D> SSparseArrayUIntPtrList2D;
typedef std::vector<SSparseArrayShortPtrList1D> SSparseArrayShortPtrList2D;
typedef std::vector<SSparseArrayUShortPtrList1D> SSparseArrayUShortPtrList2D;
typedef std::vector<SSparseArrayLongPtrList1D> SSparseArrayLongPtrList2D;
typedef std::vector<SSparseArrayULongPtrList1D> SSparseArrayULongPtrList2D;
typedef std::vector<SSparseArrayDoublePtrList1D> SSparseArrayDoublePtrList2D;

/**
 * @}
 */

#endif  // TICK_BASE_ARRAY_SRC_SSPARSEARRAY_H_
